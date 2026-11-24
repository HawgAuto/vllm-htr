// ple_gather_v25.c â€” Clean-slate V5-P5 Unified PLE Gather + Optional Bounded Hot-Row LRU Cache
// Combines:
//   1. Caller-side RWF_NOWAIT page-cache pass + persistent futex worker pool (PLE_GATHER_SPIN_ITERS, PLE_GATHER_NOWAIT_MAX)
//   2. Persistent reusable scratch buffer in ple_gather_ids (eliminates per-chunk 42 MB mmap/munmap + minor page faults!)
//   3. APU PPT thread cap (PLE_GATHER_MAX_THREADS) so 128-thread wakes don't steal 85W PPT Watts from gfx1151
//   4. Parallel worker-pool scatter memcpy for large prefills (n >= 256)
//   5. Optional bounded O(1) hot-row LRU cache (HTR_PLE_ROW_CACHE_BYTES; 0 = disabled, exact ABI compat with libple_row_cache.so!)
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#ifndef RWF_NOWAIT
#define RWF_NOWAIT 0x00000008
#endif
#define MAXT 128
#define SPIN_ITERS 4000

static int g_spin_iters = -1;
static int g_nowait_max = -1;
static int g_max_threads_cap = -1;

static inline int get_spin_iters(void) {
  int v = g_spin_iters;
  if (v < 0) {
    const char *e = getenv("PLE_GATHER_SPIN_ITERS");
    v = e ? atoi(e) : SPIN_ITERS;
    if (v < 0) v = 0;
    if (v > 100000) v = 100000;
    g_spin_iters = v;
  }
  return v;
}
static inline int get_nowait_max(void) {
  int v = g_nowait_max;
  if (v < 0) {
    const char *e = getenv("PLE_GATHER_NOWAIT_MAX");
    v = e ? atoi(e) : 64;
    if (v < 0) v = 0;
    if (v > 65536) v = 65536;
    g_nowait_max = v;
  }
  return v;
}
static inline int get_max_threads_cap(void) {
  int v = g_max_threads_cap;
  if (v < 0) {
    const char *e = getenv("PLE_GATHER_MAX_THREADS");
    v = (e && *e) ? atoi(e) : MAXT;
    if (v < 1) v = 1;
    if (v > MAXT) v = MAXT;
    g_max_threads_cap = v;
  }
  return v;
}

typedef struct { int64_t id; int64_t pos; } idpos_t;

typedef struct {
  int kind;  // 0 = cold pread, 1 = parallel scatter memcpy
  const int32_t *fds;
  const int64_t *offs, *lens, *outs;
  char *out;
  const int32_t *items;
  const int64_t *done0;
  int64_t n;
  // Scatter fields (kind == 1)
  const idpos_t *sc_ip;
  const int64_t *sc_uidx;
  const char *sc_ubuf;
  int64_t sc_rb;
} job_t;

typedef struct {
  _Atomic uint32_t gen;
  char pad[60];
} slot_t;

static job_t g_job;
static slot_t g_slot[MAXT];
static _Atomic int32_t g_left;
static _Atomic int32_t g_err;
static pthread_t g_th[MAXT];
static uint32_t g_seen0[MAXT];
static int g_pool;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t g_stat[4];

// Persistent reusable scratch buffers (eliminates per-call mmap/munmap + minor page faults!)
static char *g_scratch = NULL;
static size_t g_scratch_cap = 0;
static idpos_t *g_ip_buf = NULL;
static size_t g_ip_cap = 0;
static int64_t *g_uidx_buf = NULL;
static size_t g_uidx_cap = 0;
static int32_t *g_cold_items = NULL;
static int64_t *g_cold_done0 = NULL;
static size_t g_cold_cap = 0;

static char *ensure_scratch(size_t need) {
  if (need <= g_scratch_cap) return g_scratch;
  size_t cap = g_scratch_cap ? g_scratch_cap * 2 : (4ULL << 20);
  while (cap < need) cap *= 2;
  char *p = (char *)realloc(g_scratch, cap);
  if (!p) return NULL;
  g_scratch = p;
  g_scratch_cap = cap;
  return g_scratch;
}
static int ensure_ip_uidx(size_t n) {
  if (n > g_ip_cap) {
    size_t cap = g_ip_cap ? g_ip_cap * 2 : 1024;
    while (cap < n) cap *= 2;
    idpos_t *p = (idpos_t *)realloc(g_ip_buf, cap * sizeof(idpos_t));
    if (!p) return ENOMEM;
    g_ip_buf = p; g_ip_cap = cap;
  }
  if (n > g_uidx_cap) {
    size_t cap = g_uidx_cap ? g_uidx_cap * 2 : 1024;
    while (cap < n) cap *= 2;
    int64_t *p = (int64_t *)realloc(g_uidx_buf, cap * sizeof(int64_t));
    if (!p) return ENOMEM;
    g_uidx_buf = p; g_uidx_cap = cap;
  }
  return 0;
}
static int ensure_cold(size_t n) {
  if (n <= g_cold_cap) return 0;
  size_t cap = g_cold_cap ? g_cold_cap * 2 : 1024;
  while (cap < n) cap *= 2;
  int32_t *pi = (int32_t *)realloc(g_cold_items, cap * sizeof(int32_t));
  if (!pi) return ENOMEM;
  g_cold_items = pi;
  int64_t *pd = (int64_t *)realloc(g_cold_done0, cap * sizeof(int64_t));
  if (!pd) return ENOMEM;
  g_cold_done0 = pd;
  g_cold_cap = cap;
  return 0;
}

static long fwait(_Atomic uint32_t *addr, uint32_t val) {
  return syscall(SYS_futex, (uint32_t *)addr, FUTEX_WAIT_PRIVATE, val, NULL, NULL, 0);
}
static long fwake(_Atomic uint32_t *addr, int n) {
  return syscall(SYS_futex, (uint32_t *)addr, FUTEX_WAKE_PRIVATE, n, NULL, NULL, 0);
}

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

static int g_nparts;
static void run_part(int part) {
  const int64_t b = g_job.n * part / g_nparts, e = g_job.n * (part + 1) / g_nparts;
  if (g_job.kind == 0) {
    for (int64_t k = b; k < e; k++) {
      const int32_t i = g_job.items[k];
      const int r = read_rest(g_job.fds[i], g_job.out + g_job.outs[i], g_job.lens[i], g_job.offs[i], g_job.done0[k]);
      if (r) atomic_store_explicit(&g_err, r, memory_order_relaxed);
    }
  } else {
    const int64_t rb = g_job.sc_rb;
    for (int64_t i = b; i < e; i++) {
      int64_t ui = g_job.sc_uidx[i];
      if (ui >= 0) memcpy(g_job.out + g_job.sc_ip[i].pos * rb, g_job.sc_ubuf + ui * rb, (size_t)rb);
    }
  }
}

static void *worker(void *arg) {
  const int id = (int)(intptr_t)arg;
  slot_t *s = &g_slot[id];
  uint32_t seen = g_seen0[id];
  for (;;) {
    uint32_t g;
    int spins = 0;
    while ((g = atomic_load_explicit(&s->gen, memory_order_acquire)) == seen) {
      if (spins < get_spin_iters()) { ++spins; __builtin_ia32_pause(); continue; }
      fwait(&s->gen, seen);
    }
    seen = g;
    run_part(id + 1);
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

static void dispatch_pool(int k) {
  if (k > MAXT) k = MAXT;
  if (k > 0) ensure_pool(k);
  if (k > g_pool) k = g_pool;
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
  run_part(0);
  int spins = 0;
  int32_t left;
  while ((left = atomic_load_explicit(&g_left, memory_order_acquire)) != 0) {
    if (spins < get_spin_iters()) { ++spins; __builtin_ia32_pause(); continue; }
    fwait((_Atomic uint32_t *)&g_left, (uint32_t)left);
  }
}

static int ple_gather_locked(const int32_t *fds, const int64_t *offs, const int64_t *lens, const int64_t *outs,
                             char *out, int64_t n, int nthreads) {
  int cap = get_max_threads_cap();
  if (nthreads > cap) nthreads = cap;
  atomic_fetch_add_explicit(&g_stat[0], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&g_stat[1], (uint64_t)n, memory_order_relaxed);
  int32_t stack_items[256];
  int64_t stack_done[256];
  int32_t *items = stack_items;
  int64_t *done0 = stack_done;
  if (n > 256) {
    int er = ensure_cold((size_t)n);
    if (er) return er;
    items = g_cold_items;
    done0 = g_cold_done0;
  }
  int64_t nc = 0;
  int err = 0;
  const int big = n > get_nowait_max() && g_mode < 0;
  for (int64_t i = 0; i < n && !err; i++) {
    int64_t done = 0;
    while (!big && done < lens[i]) {
      struct iovec v = {out + outs[i] + done, (size_t)(lens[i] - done)};
      ssize_t r = preadv2(fds[i], &v, 1, (off_t)(offs[i] + done), RWF_NOWAIT);
      if (r < 0 && errno == EINTR) continue;
      if (r > 0) { done += r; continue; }
      if (r < 0 && errno != EAGAIN && errno != EOPNOTSUPP && errno != ENOSYS) { err = errno; }
      break;
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
      return err;
    }
    int k = nthreads - 1;
    if (k > nc - 1) k = (int)(nc - 1);
    g_job.kind = 0;
    g_job.fds = fds; g_job.offs = offs; g_job.lens = lens; g_job.outs = outs; g_job.out = out;
    g_job.items = items; g_job.done0 = done0; g_job.n = nc;
    dispatch_pool(k);
    err = atomic_load_explicit(&g_err, memory_order_relaxed);
  }
  return err;
}

int ple_gather(const int32_t *fds, const int64_t *offs, const int64_t *lens, const int64_t *outs, char *out, int64_t n,
               int nthreads) {
  if (n < 0 || nthreads < 1 || nthreads > 256) return -2;
  if (n == 0) return 0;
  if (n > (int64_t)1 << 30) return -2;
  pthread_mutex_lock(&g_mu);
  int err = ple_gather_locked(fds, offs, lens, outs, out, n, nthreads);
  pthread_mutex_unlock(&g_mu);
  return err;
}

void ple_gather_stats(uint64_t out4[4]) {
  for (int i = 0; i < 4; i++) out4[i] = atomic_load_explicit(&g_stat[i], memory_order_relaxed);
}

// ---------------------------------------------------------------------------------------------------------------------
// Optional Bounded O(1) Hot-Row LRU Cache (HTR_PLE_ROW_CACHE_BYTES; 0 = disabled, exact libple_row_cache.so ABI!)
// ---------------------------------------------------------------------------------------------------------------------
#define RC_NIL UINT32_MAX
#define RC_PART_MAX 256
#define RC_STAT_N 18
#define RC_MAX_BUDGET (64ULL << 30)

typedef struct { int64_t id; uint32_t hash_next, prev, next; } RCEntry;
typedef struct { int32_t fd; int64_t base, rows; dev_t dev; ino_t ino; off_t size; } RCSource;
typedef struct {
  uint64_t s[RC_STAT_N];  // [0]=budget,[1]=cap,[2]=used,[4]=calls,[5]=lookups,[6]=hits,[7]=misses,[8]=evictions
  void *arena; size_t arena_bytes;
  RCEntry *entries; uint32_t *buckets; char *data;
  uint32_t cap, nb, used, head, tail;
  int64_t nparts, shard, rb, total;
  RCSource src[RC_PART_MAX];
  int bound;
} RCCache;

static RCCache g_rc;
static pthread_once_t g_rc_once = PTHREAD_ONCE_INIT;

static uint64_t rc_hash(uint64_t x) {
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27; x *= 0x94d049bb133111ebULL; return x ^ (x >> 31);
}
static void rc_free_arena(void) {
  if (g_rc.arena) munmap(g_rc.arena, g_rc.arena_bytes);
  g_rc.arena = NULL; g_rc.entries = NULL; g_rc.buckets = NULL; g_rc.data = NULL;
  g_rc.arena_bytes = 0; g_rc.cap = g_rc.nb = g_rc.used = 0; g_rc.head = g_rc.tail = RC_NIL; g_rc.bound = 0;
  g_rc.s[1] = g_rc.s[2] = 0;
}
static int rc_configure_locked(uint64_t budget) {
  if (budget > RC_MAX_BUDGET) return EINVAL;
  rc_free_arena();
  memset(g_rc.s, 0, sizeof(g_rc.s));
  g_rc.s[0] = budget;
  return 0;
}
static void rc_init_once(void) {
  const char *e = getenv("HTR_PLE_ROW_CACHE_BYTES");
  if (e && *e) {
    unsigned long long b = strtoull(e, NULL, 10);
    if (b > 0 && b <= RC_MAX_BUDGET) rc_configure_locked((uint64_t)b);
  }
}
int ple_cache_configure(uint64_t budget) {
  pthread_once(&g_rc_once, rc_init_once);
  pthread_mutex_lock(&g_mu);
  int e = rc_configure_locked(budget);
  pthread_mutex_unlock(&g_mu);
  return e;
}
int ple_cache_stats(uint64_t *out, uint64_t n) {
  if (!out || n != RC_STAT_N) return EINVAL;
  pthread_once(&g_rc_once, rc_init_once);
  pthread_mutex_lock(&g_mu);
  memcpy(out, g_rc.s, sizeof(g_rc.s));
  pthread_mutex_unlock(&g_mu);
  return 0;
}
static int rc_bind_locked(const int32_t *pfd, const int64_t *pbase, const int64_t *prows,
                          int64_t nparts, int64_t shard, int64_t rb, int64_t total) {
  if (g_rc.s[0] == 0 || nparts > RC_PART_MAX) return 0;
  if (g_rc.bound && g_rc.nparts == nparts && g_rc.shard == shard && g_rc.rb == rb && g_rc.total == total) {
    int same = 1;
    for (int64_t i = 0; i < nparts; i++) {
      if (g_rc.src[i].fd != pfd[i] || g_rc.src[i].base != pbase[i] || g_rc.src[i].rows != prows[i]) { same = 0; break; }
    }
    if (same) return 1;
  }
  rc_free_arena();
  uint64_t budget = g_rc.s[0];
  uint64_t per_row = (uint64_t)rb + sizeof(RCEntry) + 8;
  uint64_t cap64 = budget / per_row;
  if (cap64 < 64) return 0;
  if (cap64 > (uint64_t)total) cap64 = (uint64_t)total;
  if (cap64 > 2000000ULL) cap64 = 2000000ULL;
  uint32_t cap = (uint32_t)cap64;
  uint32_t nb = 64;
  while (nb < cap * 2 && nb < (1U << 22)) nb <<= 1;
  size_t e_bytes = (size_t)cap * sizeof(RCEntry);
  size_t b_bytes = (size_t)nb * sizeof(uint32_t);
  size_t d_off = (e_bytes + b_bytes + 63) & ~63ULL;
  size_t tot_bytes = d_off + (size_t)cap * (size_t)rb;
  void *ar = mmap(NULL, tot_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ar == MAP_FAILED) return 0;
  g_rc.arena = ar; g_rc.arena_bytes = tot_bytes;
  g_rc.entries = (RCEntry *)ar;
  g_rc.buckets = (uint32_t *)((char *)ar + e_bytes);
  g_rc.data = (char *)ar + d_off;
  for (uint32_t i = 0; i < nb; i++) g_rc.buckets[i] = RC_NIL;
  g_rc.cap = cap; g_rc.nb = nb; g_rc.used = 0; g_rc.head = g_rc.tail = RC_NIL;
  g_rc.nparts = nparts; g_rc.shard = shard; g_rc.rb = rb; g_rc.total = total;
  for (int64_t i = 0; i < nparts; i++) {
    struct stat st;
    if (fstat(pfd[i], &st) != 0) { rc_free_arena(); return 0; }
    g_rc.src[i].fd = pfd[i]; g_rc.src[i].base = pbase[i]; g_rc.src[i].rows = prows[i];
    g_rc.src[i].dev = st.st_dev; g_rc.src[i].ino = st.st_ino; g_rc.src[i].size = st.st_size;
  }
  g_rc.bound = 1;
  g_rc.s[1] = cap;
  return 1;
}
static inline void rc_unlink(uint32_t i) {
  RCEntry *e = g_rc.entries + i;
  if (e->prev != RC_NIL) g_rc.entries[e->prev].next = e->next; else g_rc.head = e->next;
  if (e->next != RC_NIL) g_rc.entries[e->next].prev = e->prev; else g_rc.tail = e->prev;
}
static inline void rc_front(uint32_t i) {
  RCEntry *e = g_rc.entries + i;
  e->prev = RC_NIL; e->next = g_rc.head;
  if (g_rc.head != RC_NIL) g_rc.entries[g_rc.head].prev = i; else g_rc.tail = i;
  g_rc.head = i;
}
static inline uint32_t rc_lookup(int64_t id) {
  uint32_t b = (uint32_t)rc_hash((uint64_t)id) & (g_rc.nb - 1);
  for (uint32_t i = g_rc.buckets[b]; i != RC_NIL; i = g_rc.entries[i].hash_next) {
    if (g_rc.entries[i].id == id) { rc_unlink(i); rc_front(i); return i; }
  }
  return RC_NIL;
}
static inline void rc_insert(int64_t id, const char *row_data) {
  uint32_t slot;
  if (g_rc.used < g_rc.cap) {
    slot = g_rc.used++;
    g_rc.s[2] = g_rc.used;
  } else {
    slot = g_rc.tail;
    rc_unlink(slot);
    int64_t old_id = g_rc.entries[slot].id;
    uint32_t ob = (uint32_t)rc_hash((uint64_t)old_id) & (g_rc.nb - 1);
    uint32_t *pp = &g_rc.buckets[ob];
    while (*pp != RC_NIL && *pp != slot) pp = &g_rc.entries[*pp].hash_next;
    if (*pp == slot) *pp = g_rc.entries[slot].hash_next;
    g_rc.s[8]++;
  }
  uint32_t b = (uint32_t)rc_hash((uint64_t)id) & (g_rc.nb - 1);
  g_rc.entries[slot].id = id;
  g_rc.entries[slot].hash_next = g_rc.buckets[b];
  g_rc.buckets[b] = slot;
  rc_front(slot);
  memcpy(g_rc.data + (size_t)slot * (size_t)g_rc.rb, row_data, (size_t)g_rc.rb);
}

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

  pthread_once(&g_rc_once, rc_init_once);
  pthread_mutex_lock(&g_mu);

  int use_rc = rc_bind_locked(pfd, pbase, prows, nparts, shard, row_bytes, num_embeddings);
  if (use_rc) g_rc.s[4]++;

  idpos_t sbuf[64];
  int64_t suidx[64];
  idpos_t *ip = sbuf;
  int64_t *uidx = suidx;
  if (n > 64) {
    int er = ensure_ip_uidx((size_t)n);
    if (er) { pthread_mutex_unlock(&g_mu); return er; }
    ip = g_ip_buf;
    uidx = g_uidx_buf;
  }

  for (int64_t i = 0; i < n; i++) { ip[i].id = ids[i]; ip[i].pos = i; }
  if (n <= 64) {
    for (int64_t i = 1; i < n; i++) {
      idpos_t v = ip[i]; int64_t j = i - 1;
      while (j >= 0 && (ip[j].id > v.id || (ip[j].id == v.id && ip[j].pos > v.pos))) { ip[j + 1] = ip[j]; --j; }
      ip[j + 1] = v;
    }
  } else {
    qsort(ip, (size_t)n, sizeof(idpos_t), cmp_idpos);
  }

  // Count unique ids and resolve cache hits inline (zero disk I/O for cached rows!)
  int64_t u_total = 0, u_miss = 0;
  for (int64_t i = 0; i < n; i++) {
    if (i == 0 || ip[i].id != ip[i - 1].id) ++u_total;
  }

  size_t meta_bytes = (size_t)u_total * (sizeof(int64_t) * 6 + sizeof(int32_t) * 2);
  size_t need = meta_bytes + (size_t)u_total * (size_t)row_bytes;
  char *mem = ensure_scratch(need);
  if (!mem) { pthread_mutex_unlock(&g_mu); return ENOMEM; }

  int64_t *uoff = (int64_t *)mem;
  int64_t *roffs = uoff + u_total;
  int64_t *rlens = roffs + u_total;
  int64_t *routs = rlens + u_total;
  int64_t *ush = routs + u_total;
  int64_t *miss_ids = ush + u_total;
  int32_t *ufd = (int32_t *)(miss_ids + u_total);
  int32_t *rfds = ufd + u_total;
  char *ubuf = (char *)(rfds + u_total);

  int64_t nr = 0;
  int err = 0;
  int64_t cur_slot = -1;
  uint32_t cur_rc_slot = RC_NIL;

  for (int64_t i = 0; i < n && !err; i++) {
    if (i == 0 || ip[i].id != ip[i - 1].id) {
      int64_t id = ip[i].id;
      if (use_rc) {
        g_rc.s[5]++;
        cur_rc_slot = rc_lookup(id);
      } else {
        cur_rc_slot = RC_NIL;
      }
      if (cur_rc_slot != RC_NIL) {
        g_rc.s[6]++;
        cur_slot = -1;
      } else {
        if (use_rc) g_rc.s[7]++;
        int64_t k = u_miss++;
        cur_slot = k;
        miss_ids[k] = id;
        const int64_t s = id / shard, rin = id - s * shard;
        if (s >= nparts || rin >= prows[s]) { err = -4; break; }
        uoff[k] = pbase[s] + rin * row_bytes;
        ufd[k] = pfd[s];
        ush[k] = s;
        if (k > 0 && ush[k] == ush[k - 1] && uoff[k] == uoff[k - 1] + row_bytes) {
          rlens[nr - 1] += row_bytes;
        } else {
          rfds[nr] = ufd[k]; roffs[nr] = uoff[k]; rlens[nr] = row_bytes; routs[nr] = k * row_bytes; ++nr;
        }
      }
    }
    if (cur_rc_slot != RC_NIL) {
      memcpy(out + ip[i].pos * row_bytes, g_rc.data + (size_t)cur_rc_slot * (size_t)row_bytes, (size_t)row_bytes);
      uidx[i] = -1;
    } else {
      uidx[i] = cur_slot;
    }
  }

  if (!err && nr > 0) {
    int nt = (int)(nr / min_runs_per_thread);
    if (nt > max_threads) nt = max_threads;
    int cap = get_max_threads_cap();
    if (nt > cap) nt = cap;
    if (nt < 1) nt = 1;
    err = ple_gather_locked(rfds, roffs, rlens, routs, ubuf, nr, nt);
  }

  if (!err && u_miss > 0) {
    // Insert newly read unique rows into LRU cache if enabled
    if (use_rc) {
      for (int64_t k = 0; k < u_miss; k++) {
        rc_insert(miss_ids[k], ubuf + k * row_bytes);
      }
    }
    // Scatter ubuf -> out (parallel worker pool for large prefills n >= 256, both cache=off and cache=on!)
    if (n >= 256) {
      int k_sc = get_max_threads_cap() - 1;
      if (k_sc > 15) k_sc = 15;
      if (k_sc > (int)(n / 64) - 1) k_sc = (int)(n / 64) - 1;
      if (k_sc > 0) {
        g_job.kind = 1;
        g_job.out = out;
        g_job.n = n;
        g_job.sc_ip = ip;
        g_job.sc_uidx = uidx;
        g_job.sc_ubuf = ubuf;
        g_job.sc_rb = row_bytes;
        dispatch_pool(k_sc);
      } else {
        for (int64_t i = 0; i < n; i++)
          if (uidx[i] >= 0)
            memcpy(out + ip[i].pos * row_bytes, ubuf + uidx[i] * row_bytes, (size_t)row_bytes);
      }
    } else {
      for (int64_t i = 0; i < n; i++) {
        if (uidx[i] >= 0)
          memcpy(out + ip[i].pos * row_bytes, ubuf + uidx[i] * row_bytes, (size_t)row_bytes);
      }
    }
  }

  if (!err && stats3) {
    int64_t bytes = 0;
    for (int64_t r = 0; r < nr; r++) bytes += rlens[r];
    stats3[0] = u_total; stats3[1] = nr; stats3[2] = bytes;
  }

  pthread_mutex_unlock(&g_mu);
  return err;
}

