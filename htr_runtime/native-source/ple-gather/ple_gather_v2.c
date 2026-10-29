// E27 PLE gather v2: exact batched pread gather for PLE rows with (1) a caller-side RWF_NOWAIT page-cache pass and (2) a
// persistent worker pool for the rows that would block (cold NVMe reads). Same contract and bytes as the image's
// ple_gather.c (ple_gather(fds, offs, lens, outs, out, n, nthreads) -> 0 / errno / -1 short read / -2 bad args); no caching,
// no reordering of bytes. nthreads bounds the number of pool workers used for the cold remainder of this call.
//
// Why: decode looks up 16 rows/token. The image library runs them on one thread when runs < MIN_RUNS_PER_THREAD (512), so
// cold rows cost 16 x ~225 us serially (E25 timing: 3.3-3.9 ms/token on the GPU-idle critical path); spawning 16 pthreads per
// token instead costs ~15 pthread_create/join. Here warm rows complete inline (no thread hop) and cold rows go to already
// running workers (one futex wake each), so a cold token costs ~one NVMe latency instead of sixteen.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#ifndef RWF_NOWAIT
#define RWF_NOWAIT 0x00000008
#endif
#define MAXT 128
#define SPIN_ITERS 4000

typedef struct {
  const int32_t *fds;
  const int64_t *offs, *lens, *outs;
  char *out;
  const int32_t *items;  // cold item indices
  const int64_t *done0;  // bytes already read by the nowait pass, per cold slot
  int64_t n;             // number of cold items
} job_t;

typedef struct {
  _Atomic uint32_t gen;  // futex word: incremented by the caller to start a job on this worker
  char pad[60];
} slot_t;

static job_t g_job;
static slot_t g_slot[MAXT];
static _Atomic int32_t g_left;  // workers still running the current job (futex word for the caller)
static _Atomic int32_t g_err;
static pthread_t g_th[MAXT];
static uint32_t g_seen0[MAXT];
static int g_pool;  // workers created
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;  // one gather at a time (pool is a singleton)
static _Atomic uint64_t g_stat[4];  // calls, items, cold items, pool jobs

static long fwait(_Atomic uint32_t *addr, uint32_t val) {
  return syscall(SYS_futex, (uint32_t *)addr, FUTEX_WAIT_PRIVATE, val, NULL, NULL, 0);
}
static long fwake(_Atomic uint32_t *addr, int n) {
  return syscall(SYS_futex, (uint32_t *)addr, FUTEX_WAKE_PRIVATE, n, NULL, NULL, 0);
}

// cold-item strategy: 0 = persistent pool; 1 = caller-only: POSIX_FADV_WILLNEED every cold range first (the kernel submits the
// reads asynchronously, the NVMe serves them in parallel), then blocking preads in order (they wait on in-flight I/O);
// 2 = WILLNEED hints + pool. Set with ple_gather_set_mode (default from env PLE_GATHER_V2_MODE, else 1 for n <= 64, 2 above).
static int g_mode = -1;
void ple_gather_set_mode(int mode) { g_mode = mode; }

static int read_rest(int fd, char *dst, int64_t len, int64_t off, int64_t done) {
  while (done < len) {
    ssize_t r = pread(fd, dst + done, (size_t)(len - done), (off_t)(off + done));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return r < 0 ? errno : -1;
    done += r;
  }
  return 0;
}

static int g_nparts;  // participants of the current job (caller + workers); static contiguous slices like the image library
static void run_cold(int part) {
  const int64_t b = g_job.n * part / g_nparts, e = g_job.n * (part + 1) / g_nparts;
  for (int64_t k = b; k < e; k++) {
    const int32_t i = g_job.items[k];
    const int r = read_rest(g_job.fds[i], g_job.out + g_job.outs[i], g_job.lens[i], g_job.offs[i], g_job.done0[k]);
    if (r) atomic_store_explicit(&g_err, r, memory_order_relaxed);
  }
}

static void *worker(void *arg) {
  const int id = (int)(intptr_t)arg;
  slot_t *s = &g_slot[id];
  uint32_t seen = g_seen0[id];  // gen value BEFORE this worker's first job (set before pthread_create; a late-starting thread
                                // must not mistake an already-posted job for the baseline -> deadlock)
  for (;;) {
    uint32_t g;
    int spins = 0;
    while ((g = atomic_load_explicit(&s->gen, memory_order_acquire)) == seen) {
      if (spins < SPIN_ITERS) { ++spins; __builtin_ia32_pause(); continue; }
      fwait(&s->gen, seen);
    }
    seen = g;
    run_cold(id + 1);
    if (atomic_fetch_sub_explicit(&g_left, 1, memory_order_acq_rel) == 1) fwake((_Atomic uint32_t *)&g_left, 1);
  }
  return NULL;
}

static void ensure_pool(int want) {
  if (want > MAXT) want = MAXT;
  while (g_pool < want) {
    g_seen0[g_pool] = atomic_load_explicit(&g_slot[g_pool].gen, memory_order_acquire);
    if (pthread_create(&g_th[g_pool], NULL, worker, (void *)(intptr_t)g_pool) != 0) break;
    pthread_detach(g_th[g_pool]);
    ++g_pool;
  }
}

int ple_gather(const int32_t *fds, const int64_t *offs, const int64_t *lens, const int64_t *outs, char *out, int64_t n,
               int nthreads) {
  if (n < 0 || nthreads < 1 || nthreads > 256) return -2;
  if (n == 0) return 0;
  if (n > (int64_t)1 << 30) return -2;
  pthread_mutex_lock(&g_mu);
  atomic_fetch_add_explicit(&g_stat[0], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&g_stat[1], (uint64_t)n, memory_order_relaxed);
  // pass 1 (caller): page-cache hits complete without blocking; misses (EAGAIN / short) become cold items
  int32_t stack_items[256];
  int64_t stack_done[256];
  int32_t *items = stack_items;
  int64_t *done0 = stack_done;
  void *heap = NULL;
  if (n > 256) {
    heap = malloc((size_t)n * (sizeof(int32_t) + sizeof(int64_t)));
    if (!heap) { pthread_mutex_unlock(&g_mu); return ENOMEM; }
    done0 = (int64_t *)heap;
    items = (int32_t *)(done0 + n);
  }
  int64_t nc = 0;
  int err = 0;
  const int big = n > 64 && g_mode < 0;  // large lookups (prefill): no serial caller pass; everything to the pool (= image
                                         // library parallelism without per-call pthread_create); small (decode): nowait pass
  for (int64_t i = 0; i < n && !err; i++) {
    int64_t done = 0;
    while (!big && done < lens[i]) {
      struct iovec v = {out + outs[i] + done, (size_t)(lens[i] - done)};
      ssize_t r = preadv2(fds[i], &v, 1, (off_t)(offs[i] + done), RWF_NOWAIT);
      if (r < 0 && errno == EINTR) continue;
      if (r > 0) { done += r; continue; }
      if (r < 0 && errno != EAGAIN && errno != EOPNOTSUPP && errno != ENOSYS) { err = errno; }
      break;  // would block (or 0 = EOF -> let the blocking path report the short read)
    }
    if (!err && done < lens[i]) { items[nc] = (int32_t)i; done0[nc] = done; ++nc; }
  }
  if (!err && nc) {
    atomic_fetch_add_explicit(&g_stat[2], (uint64_t)nc, memory_order_relaxed);
    int mode = g_mode;
    if (mode < 0) {
      const char *e = getenv("PLE_GATHER_V2_MODE");
      mode = e ? atoi(e) : -2;
    }
    if (mode == -2) mode = big ? 0 : 1;
    if (mode == 1 || mode == 2) {
      for (int64_t c = 0; c < nc; c++) {
        const int32_t i = items[c];
        posix_fadvise(fds[i], (off_t)(offs[i] + done0[c]), (off_t)(lens[i] - done0[c]), POSIX_FADV_WILLNEED);
      }
    }
    if (mode == 1) {
      for (int64_t c = 0; c < nc && !err; c++) {
        const int32_t i = items[c];
        err = read_rest(fds[i], out + outs[i], lens[i], offs[i], done0[c]);
      }
      goto done;
    }
    int k = nthreads - 1;  // pool workers in addition to the caller
    if (k > nc - 1) k = (int)(nc - 1);
    if (k > MAXT) k = MAXT;
    if (k > 0) ensure_pool(k);
    if (k > g_pool) k = g_pool;
    g_job.fds = fds; g_job.offs = offs; g_job.lens = lens; g_job.outs = outs; g_job.out = out;
    g_job.items = items; g_job.done0 = done0; g_job.n = nc;
    g_nparts = k + 1;
    atomic_store_explicit(&g_err, 0, memory_order_relaxed);
    atomic_store_explicit(&g_left, k, memory_order_release);
    if (k > 0) {
      atomic_fetch_add_explicit(&g_stat[3], 1, memory_order_relaxed);
      for (int t = 0; t < k; t++) {
        atomic_fetch_add_explicit(&g_slot[t].gen, 1, memory_order_release);
        fwake(&g_slot[t].gen, 1);
      }
    }
    run_cold(0);
    int spins = 0;
    int32_t left;
    while ((left = atomic_load_explicit(&g_left, memory_order_acquire)) != 0) {
      if (spins < SPIN_ITERS) { ++spins; __builtin_ia32_pause(); continue; }
      fwait((_Atomic uint32_t *)&g_left, (uint32_t)left);
    }
    err = atomic_load_explicit(&g_err, memory_order_relaxed);
  }
done:
  free(heap);
  pthread_mutex_unlock(&g_mu);
  return err;
}

// diagnostics: calls, items, cold items, pool jobs
void ple_gather_stats(uint64_t out4[4]) {
  for (int i = 0; i < 4; i++) out4[i] = atomic_load_explicit(&g_stat[i], memory_order_relaxed);
}

// ---------------------------------------------------------------------------------------------------------------------
// ple_gather_ids: the whole direct_ple_ab_v1.gather_rows_native body in C (one ctypes call per lookup instead of ~20 numpy
// ops + a per-call fstat loop). Same mapping as the Python path: shard = ceil(num_embeddings / nparts), s = id / shard,
// rin = id - s*shard (must be < rows[s]), offset = base[s] + rin*row_bytes; unique ids are read once (ascending order, runs
// merged when adjacent rows of one shard are contiguous: off[k] == off[k-1] + row_bytes) and scattered to every position.
// nthreads = max(1, min(max_threads, runs / min_runs_per_thread)) exactly as _threads(runs). stats3 = {unique, runs, bytes}.
// Returns 0 / errno / -1 short read / -2 bad args / -3 id out of range / -4 row beyond shard rows.
typedef struct { int64_t id; int64_t pos; } idpos_t;
static int cmp_idpos(const void *a, const void *b) {
  const idpos_t *x = (const idpos_t *)a, *y = (const idpos_t *)b;
  if (x->id != y->id) return x->id < y->id ? -1 : 1;
  return x->pos < y->pos ? -1 : (x->pos > y->pos);
}

int ple_gather_ids(const int64_t *ids, int64_t n, const int32_t *pfd, const int64_t *pbase, const int64_t *prows,
                   int64_t nparts, int64_t shard, int64_t row_bytes, int64_t num_embeddings, char *out, int max_threads,
                   int min_runs_per_thread, int64_t *stats3) {
  if (n < 0 || nparts < 1 || shard < 1 || row_bytes < 1 || max_threads < 1 || max_threads > 256 || min_runs_per_thread < 1)
    return -2;
  if (n == 0) { if (stats3) { stats3[0] = stats3[1] = stats3[2] = 0; } return 0; }
  for (int64_t i = 0; i < n; i++)
    if (ids[i] < 0 || ids[i] >= num_embeddings) return -3;
  idpos_t sbuf[64];
  idpos_t *ip = n <= 64 ? sbuf : (idpos_t *)malloc((size_t)n * sizeof(idpos_t));
  if (!ip) return ENOMEM;
  for (int64_t i = 0; i < n; i++) { ip[i].id = ids[i]; ip[i].pos = i; }
  if (n <= 64) {  // insertion sort (decode: 16 ids)
    for (int64_t i = 1; i < n; i++) {
      idpos_t v = ip[i]; int64_t j = i - 1;
      while (j >= 0 && (ip[j].id > v.id || (ip[j].id == v.id && ip[j].pos > v.pos))) { ip[j + 1] = ip[j]; --j; }
      ip[j + 1] = v;
    }
  } else {
    qsort(ip, (size_t)n, sizeof(idpos_t), cmp_idpos);
  }
  // unique rows -> ubuf slots; runs over unique rows
  int64_t u = 0;
  for (int64_t i = 0; i < n; i++) if (i == 0 || ip[i].id != ip[i - 1].id) ++u;
  // per unique: offset/fd ; per run: start unique index + count
  size_t need = (size_t)u * (sizeof(int64_t) * 5 + sizeof(int32_t) * 2) + (size_t)u * (size_t)row_bytes;
  char *mem = (char *)malloc(need);
  if (!mem) { if (ip != sbuf) free(ip); return ENOMEM; }
  int64_t *uoff = (int64_t *)mem, *roffs = uoff + u, *rlens = roffs + u, *routs = rlens + u, *ush = routs + u;
  int32_t *ufd = (int32_t *)(ush + u), *rfds = ufd + u;
  char *ubuf = (char *)(rfds + u);
  int64_t k = -1, nr = 0;
  int err = 0;
  for (int64_t i = 0; i < n && !err; i++) {
    if (i == 0 || ip[i].id != ip[i - 1].id) {
      ++k;
      const int64_t s = ip[i].id / shard, rin = ip[i].id - s * shard;
      if (s >= nparts || rin >= prows[s]) { err = -4; break; }
      uoff[k] = pbase[s] + rin * row_bytes;
      ufd[k] = pfd[s];
      ush[k] = s;
      if (k > 0 && ush[k] == ush[k - 1] && uoff[k] == uoff[k - 1] + row_bytes) {  // == Python brk rule
        rlens[nr - 1] += row_bytes;
      } else {
        rfds[nr] = ufd[k]; roffs[nr] = uoff[k]; rlens[nr] = row_bytes; routs[nr] = k * row_bytes; ++nr;
      }
    }
  }
  if (!err) {
    int nt = (int)(nr / min_runs_per_thread);
    if (nt > max_threads) nt = max_threads;
    if (nt < 1) nt = 1;
    err = ple_gather(rfds, roffs, rlens, routs, ubuf, nr, nt);
  }
  if (!err) {
    k = -1;
    for (int64_t i = 0; i < n; i++) {
      if (i == 0 || ip[i].id != ip[i - 1].id) ++k;
      memcpy(out + ip[i].pos * row_bytes, ubuf + k * row_bytes, (size_t)row_bytes);
    }
    if (stats3) {
      int64_t bytes = 0;
      for (int64_t r = 0; r < nr; r++) bytes += rlens[r];
      stats3[0] = u; stats3[1] = nr; stats3[2] = bytes;
    }
  }
  free(mem);
  if (ip != sbuf) free(ip);
  return err;
}
