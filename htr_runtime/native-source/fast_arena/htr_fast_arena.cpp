// Zero-pickle, GIL-free C++ MoE arena loader for HTR-W1-P2-PACKED-1 chunks.
// Parses uncompressed (STORE) PyTorch ZIP archives directly from .htr shard fds
// via pread() and scatters into the exact e12 v2 serving arena layout.
#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <vector>

namespace {

static inline uint16_t rd16(const uint8_t* p) {
  uint16_t v; std::memcpy(&v, p, 2); return v;
}
static inline uint32_t rd32(const uint8_t* p) {
  uint32_t v; std::memcpy(&v, p, 4); return v;
}

static bool pread_all(int fd, uint8_t* dst, size_t n, int64_t off) {
  size_t done = 0;
  while (done < n) {
    ssize_t r = ::pread(fd, dst + done, n - done, off + (int64_t)done);
    if (r <= 0) return false;
    done += (size_t)r;
  }
  return true;
}

// Parse ZIP central directory for entries named "archive/data/<0..191>" (comp=0)
// and fill data_off[i] & data_len[i]. Returns false on any unexpected layout.
static bool parse_chunk_zip(const uint8_t* buf, size_t len,
                            size_t data_off[192], uint32_t data_len[192]) {
  if (len < 22) return false;
  // Locate End of Central Directory record (0x06054b50) near end of chunk
  size_t search_start = (len > 65557) ? (len - 65557) : 0;
  size_t eocd = SIZE_MAX;
  for (size_t pos = len - 22; ; --pos) {
    if (rd32(buf + pos) == 0x06054b50u) {
      uint16_t clen = rd16(buf + pos + 20);
      if (pos + 22 + clen == len) { eocd = pos; break; }
    }
    if (pos == search_start) break;
  }
  if (eocd == SIZE_MAX) return false;

  uint16_t num_entries = rd16(buf + eocd + 10);
  uint32_t cd_size = rd32(buf + eocd + 12);
  uint32_t cd_off = rd32(buf + eocd + 16);
  if ((size_t)cd_off + cd_size > eocd) return false;

  bool seen[192] = {false};
  int found = 0;
  size_t cur = cd_off;
  constexpr const char* kPrefix = "archive/data/";
  constexpr size_t kPrefixLen = 13;

  for (uint16_t i = 0; i < num_entries; ++i) {
    if (cur + 46 > (size_t)cd_off + cd_size) return false;
    if (rd32(buf + cur) != 0x02014b50u) return false;
    uint16_t comp = rd16(buf + cur + 10);
    uint32_t csize = rd32(buf + cur + 20);
    uint32_t usize = rd32(buf + cur + 24);
    uint16_t nlen = rd16(buf + cur + 28);
    uint16_t elen = rd16(buf + cur + 30);
    uint16_t clen = rd16(buf + cur + 32);
    uint32_t lhdr_off = rd32(buf + cur + 42);
    const char* name = reinterpret_cast<const char*>(buf + cur + 46);
    cur += 46 + nlen + elen + clen;

    if (nlen > kPrefixLen && std::memcmp(name, kPrefix, kPrefixLen) == 0) {
      int idx = 0;
      bool valid = true;
      for (size_t j = kPrefixLen; j < nlen; ++j) {
        char ch = name[j];
        if (ch < '0' || ch > '9') { valid = false; break; }
        idx = idx * 10 + (ch - '0');
      }
      if (!valid || idx < 0 || idx >= 192) continue;
      if (comp != 0 || csize != usize) return false;
      if ((size_t)lhdr_off + 30 > len) return false;
      if (rd32(buf + lhdr_off) != 0x04034b50u) return false;
      uint16_t lnlen = rd16(buf + lhdr_off + 26);
      uint16_t lelen = rd16(buf + lhdr_off + 28);
      size_t doff = (size_t)lhdr_off + 30 + lnlen + lelen;
      if (doff + usize > len) return false;
      data_off[idx] = doff;
      data_len[idx] = usize;
      if (!seen[idx]) { seen[idx] = true; ++found; }
    }
  }
  return found == 192;
}

static inline void copy_qweight_padded(int32_t* dst, const uint8_t* src,
                                       int rows, int raw_cols, int spad) {
  const size_t raw_bytes = (size_t)raw_cols * 4;
  if (spad == 0) {
    std::memcpy(dst, src, (size_t)rows * raw_bytes);
  } else {
    const int cols = raw_cols + spad;
    const size_t pad_bytes = (size_t)spad * 4;
    for (int r = 0; r < rows; ++r) {
      int32_t* row_dst = dst + (size_t)r * cols;
      std::memcpy(row_dst, src + (size_t)r * raw_bytes, raw_bytes);
      std::memset(row_dst + raw_cols, 0, pad_bytes);
    }
  }
}

}  // namespace

// Load all 32 chunks (512 experts, uniform K, H=2560, I=640) into pre-allocated CPU tensors.
// Releases GIL so Python ThreadPoolExecutor loads 12 layers concurrently at full NVMe bandwidth!
bool load_layer_fast(
    std::vector<int> fds,
    std::vector<int64_t> offsets,
    std::vector<int64_t> sizes,
    std::vector<int> e_bases,
    int K, int spad,
    torch::Tensor qw,
    torch::Tensor qoff,
    torch::Tensor scales,
    torch::Tensor sv,
    torch::Tensor su) {
  if (fds.size() != 32 || offsets.size() != 32 || sizes.size() != 32 || e_bases.size() != 32) {
    return false;
  }
  constexpr int H = 2560, I = 640;
  constexpr int SC_ES = 2 * I + H;             // 3840 f16 = 7680 bytes
  constexpr int SV_ES = (2 * I + H) / 8;       // 480 u8
  constexpr int SU_ES = (H + I) / 8;           // 400 u8
  const uint32_t qw_bytes = (uint32_t)(40 * 160 * 8 * K * 4);  // same for gate/up/down!
  const uint32_t exp_sizes[12] = {
    qw_bytes, (uint32_t)(I * 2), (uint32_t)(H / 8), (uint32_t)(I / 8),
    qw_bytes, (uint32_t)(I * 2), (uint32_t)(H / 8), (uint32_t)(I / 8),
    qw_bytes, (uint32_t)(H * 2), (uint32_t)(I / 8), (uint32_t)(H / 8),
  };

  int32_t* qw_ptr = qw.data_ptr<int32_t>();
  const int64_t* qoff_ptr = qoff.data_ptr<int64_t>();
  uint8_t* sc_ptr = reinterpret_cast<uint8_t*>(scales.data_ptr());
  uint8_t* sv_ptr = sv.data_ptr<uint8_t>();
  uint8_t* su_ptr = su.data_ptr<uint8_t>();

  py::gil_scoped_release nogil;
  // Thread-local reusable chunk read buffer (~40 MB, zero realloc across layers!)
  thread_local std::vector<uint8_t> buf;

  if (32 > 0) {
    ::posix_fadvise(fds[0], offsets[0], (off_t)sizes[0], POSIX_FADV_WILLNEED);
    if (32 > 1) ::posix_fadvise(fds[1], offsets[1], (off_t)sizes[1], POSIX_FADV_WILLNEED);
  }
  for (int ci = 0; ci < 32; ++ci) {
    if (ci + 2 < 32) {
      ::posix_fadvise(fds[ci + 2], offsets[ci + 2], (off_t)sizes[ci + 2], POSIX_FADV_WILLNEED);
    }
    const size_t csz = (size_t)sizes[ci];
    if (buf.size() < csz) buf.resize(csz);
    if (!pread_all(fds[ci], buf.data(), csz, offsets[ci])) return false;
    ::posix_fadvise(fds[ci], offsets[ci], (off_t)csz, POSIX_FADV_DONTNEED);

    size_t data_off[192];
    uint32_t data_len[192];
    if (!parse_chunk_zip(buf.data(), csz, data_off, data_len)) return false;

    const int e0 = e_bases[ci];
    for (int le = 0; le < 16; ++le) {
      const int e = e0 + le;
      const int base = le * 12;
      for (int k = 0; k < 12; ++k) {
        if (data_len[base + k] != exp_sizes[k]) return false;
      }
      // Verify gate.su == up.su (matches build_arena_cpu invariant!)
      if (std::memcmp(buf.data() + data_off[base + 2],
                      buf.data() + data_off[base + 6], H / 8) != 0) {
        return false;
      }

      // gate: (40, 160*8K + spad), up: (40, 160*8K + spad), down: (160, 40*8K + spad)
      copy_qweight_padded(qw_ptr + qoff_ptr[e * 3 + 0], buf.data() + data_off[base + 0], 40, 160 * 8 * K, spad);
      copy_qweight_padded(qw_ptr + qoff_ptr[e * 3 + 1], buf.data() + data_off[base + 4], 40, 160 * 8 * K, spad);
      copy_qweight_padded(qw_ptr + qoff_ptr[e * 3 + 2], buf.data() + data_off[base + 8], 160, 40 * 8 * K, spad);

      uint8_t* sb = sc_ptr + (size_t)e * SC_ES * 2;
      std::memcpy(sb,                 buf.data() + data_off[base + 1], I * 2);
      std::memcpy(sb + I * 2,         buf.data() + data_off[base + 5], I * 2);
      std::memcpy(sb + 2 * I * 2,     buf.data() + data_off[base + 9], H * 2);

      uint8_t* vb = sv_ptr + (size_t)e * SV_ES;
      std::memcpy(vb,                 buf.data() + data_off[base + 3], I / 8);
      std::memcpy(vb + I / 8,         buf.data() + data_off[base + 7], I / 8);
      std::memcpy(vb + 2 * I / 8,     buf.data() + data_off[base + 11], H / 8);

      uint8_t* ub = su_ptr + (size_t)e * SU_ES;
      std::memcpy(ub,                 buf.data() + data_off[base + 2], H / 8);
      std::memcpy(ub + H / 8,         buf.data() + data_off[base + 10], I / 8);
    }
  }
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load_layer_fast", &load_layer_fast, "Zero-pickle GIL-free HTR MoE arena loader");
}
