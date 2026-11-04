// SPDX-License-Identifier: Apache-2.0
// Optional, byte-preserving PLE row LRU. The original gather remains byte-preserved
// in vendor/; only its public gather_ids entry is wrapped. No GPU code changes.
#define _GNU_SOURCE
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <limits.h>
#include <stdio.h>

static _Atomic uint64_t rc_io[4];
static ssize_t rc_pread(int fd, void *buf, size_t n, off_t off) {
    ssize_t r = pread(fd, buf, n, off);
    atomic_fetch_add(&rc_io[0], 1);
    if (r > 0) atomic_fetch_add(&rc_io[1], (uint64_t)r);
    return r;
}
static ssize_t rc_preadv2(int fd, const struct iovec *v, int n, off_t off, int flags) {
    ssize_t r = preadv2(fd, v, n, off, flags);
    atomic_fetch_add(&rc_io[2], 1);
    if (r > 0) atomic_fetch_add(&rc_io[3], (uint64_t)r);
    return r;
}
#define pread rc_pread
#define preadv2 rc_preadv2
#define ple_gather_ids ple_uncached_gather_ids
#pragma GCC diagnostic push
// GCC's zero-run path warning belongs to the unchanged vendor implementation.
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "vendor/ple_gather_v2.c"
#pragma GCC diagnostic pop
#undef pread
#undef preadv2
#undef ple_gather_ids

#define NIL UINT32_MAX
#define PART_MAX 256
#define STAT_N 18
#define MAX_BUDGET (64ULL << 30)
#define MAX_ROWS (4LL << 20)
typedef struct { int64_t id; uint32_t hash_next, prev, next; } Entry;
typedef struct {
    int32_t fd;
    int64_t base, rows;
    dev_t dev; ino_t ino; off_t size;
    struct timespec mt, ct;
} Source;
typedef struct {
    uint64_t s[STAT_N];
    void *arena; size_t arena_bytes;
    Entry *entries; uint32_t *buckets; char *data;
    uint32_t cap, nb, used, head, tail;
    int64_t nparts, shard, rb, total;
    Source source[PART_MAX];
    int bound;
} Cache;
static Cache rc;
static pthread_mutex_t rc_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t rc_once = PTHREAD_ONCE_INIT;
static int rc_init_error;
static uint64_t *rc_public;
static uint64_t rc_sequence;
static uint64_t page_up(uint64_t n) { return (n + 4095) & ~4095ULL; }
// Conservative fixed charge includes static cache state and optional shared stats page.
static uint64_t fixed_charge(void) { return page_up(sizeof(Cache)) + 4096; }
static void publish(void) {
    for (int i=0;i<4;i++) rc.s[13+i]=atomic_load(&rc_io[i]);
    if (rc_public) {
        __atomic_store_n(rc_public, ++rc_sequence, __ATOMIC_RELEASE);
        for (int i=0;i<STAT_N;i++) __atomic_store_n(rc_public+1+i, rc.s[i], __ATOMIC_RELAXED);
        __atomic_store_n(rc_public, ++rc_sequence, __ATOMIC_RELEASE);
    }
}
static void free_arena(void) {
    if (rc.arena) munmap(rc.arena,rc.arena_bytes);
    rc.arena=NULL;rc.entries=NULL;rc.buckets=NULL;rc.data=NULL;
    rc.arena_bytes=0;rc.cap=rc.nb=rc.used=0;rc.head=rc.tail=NIL;rc.bound=0;
    rc.s[1]=rc.s[2]=rc.s[3]=0;
}
static int configure_inner(uint64_t budget) {
    if (budget && (budget < fixed_charge()+4096 || budget > MAX_BUDGET)) return EINVAL;
    free_arena();memset(rc.s,0,sizeof(rc.s));rc.s[0]=budget;
    for (int i=0;i<4;i++) atomic_store(&rc_io[i],0);
    publish();return 0;
}
static void initialize(void) {
    const char *e=getenv("HTR_PLE_ROW_CACHE_BYTES");
    uint64_t budget=0;
    if(e && *e) {
        for(const char *p=e;*p;p++) if(*p<'0'||*p>'9') {rc_init_error=EINVAL;return;}
        errno=0;char *end=NULL;unsigned long long b=strtoull(e,&end,10);
        if(errno || !end || *end) {rc_init_error=EINVAL;return;}
        budget=b;
    }
    rc_init_error=configure_inner(budget);
    if(rc_init_error)return;
    const char *path=getenv("HTR_PLE_ROW_CACHE_STATS");
    if(path && *path) {
        int fd=open(path,O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
        if(fd<0){rc_init_error=errno;return;}
        if(ftruncate(fd,4096)){rc_init_error=errno;close(fd);return;}
        void *p=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
        if(p==MAP_FAILED){rc_init_error=errno;return;}
        rc_public=p;publish();
    }
}
int ple_cache_configure(uint64_t budget) {
    pthread_once(&rc_once,initialize);pthread_mutex_lock(&rc_mu);
    int e=configure_inner(budget);
    if(!e)rc_init_error=0;
    pthread_mutex_unlock(&rc_mu);return e;
}
int ple_cache_stats(uint64_t *out,uint64_t n) {
    if(!out || n!=STAT_N)return EINVAL;
    pthread_once(&rc_once,initialize);pthread_mutex_lock(&rc_mu);
    publish();memcpy(out,rc.s,sizeof(rc.s));pthread_mutex_unlock(&rc_mu);return 0;
}
static uint64_t hash_id(uint64_t x) {
    x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);
}
static uint32_t bucket(int64_t id){return (uint32_t)hash_id((uint64_t)id)&(rc.nb-1);}
static void unlink_lru(uint32_t i) {
    Entry *e=rc.entries+i;
    if(e->prev!=NIL)rc.entries[e->prev].next=e->next;else rc.head=e->next;
    if(e->next!=NIL)rc.entries[e->next].prev=e->prev;else rc.tail=e->prev;
}
static void front(uint32_t i) {
    Entry *e=rc.entries+i;e->prev=NIL;e->next=rc.head;
    if(rc.head!=NIL)rc.entries[rc.head].prev=i;else rc.tail=i;
    rc.head=i;
}
static uint32_t lookup(int64_t id) {
    for(uint32_t i=rc.buckets[bucket(id)];i!=NIL;i=rc.entries[i].hash_next)
        if(rc.entries[i].id==id){unlink_lru(i);front(i);return i;}
    return NIL;
}
static void insert(int64_t id,const char *bytes) {
    uint32_t i;
    if(rc.used<rc.cap)i=rc.used++;
    else {
        i=rc.tail;Entry *e=rc.entries+i;uint32_t *h=rc.buckets+bucket(e->id);
        while(*h!=i)h=&rc.entries[*h].hash_next;
        *h=e->hash_next;unlink_lru(i);rc.s[7]++;
    }
    Entry *e=rc.entries+i;uint32_t b=bucket(id);
    e->id=id;e->hash_next=rc.buckets[b];rc.buckets[b]=i;front(i);
    memcpy(rc.data+(uint64_t)i*rc.rb,bytes,(size_t)rc.rb);rc.s[3]=rc.used;
}
static int geometry(uint32_t n,int64_t rb,uint32_t *nb,uint64_t *bytes) {
    uint64_t b=1;while(b<(uint64_t)n*2)b<<=1;
    if(b>UINT32_MAX)return 0;
    *nb=(uint32_t)b;*bytes=page_up((uint64_t)n*(sizeof(Entry)+(uint64_t)rb)+b*sizeof(uint32_t));
    return *bytes+fixed_charge()<=rc.s[0];
}
static int allocate(int64_t rb) {
    free_arena();uint64_t budget=rc.s[0];
    uint64_t max=(budget-fixed_charge())/(rb+sizeof(Entry)+8);
    if(max>UINT32_MAX/2)max=UINT32_MAX/2;
    uint32_t lo=0,hi=(uint32_t)max,nb=0;uint64_t bytes=0;
    while(lo<hi){uint32_t mid=lo+(hi-lo+1)/2;if(geometry(mid,rb,&nb,&bytes))lo=mid;else hi=mid-1;}
    if(!lo || !geometry(lo,rb,&nb,&bytes))return EINVAL;
    void *p=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED)return errno;
    rc.arena=p;rc.arena_bytes=bytes;rc.cap=lo;rc.nb=nb;
    rc.entries=p;rc.buckets=(uint32_t*)(rc.entries+lo);rc.data=(char*)(rc.buckets+nb);
    memset(rc.buckets,0xff,(size_t)nb*sizeof(uint32_t));
    rc.rb=rb;rc.s[1]=bytes+fixed_charge();rc.s[2]=lo;
    return 0;
}
static int bind_sources(const int32_t *fds,const int64_t *base,const int64_t *rows,
                        int64_t nparts,int64_t shard,int64_t rb,int64_t total,int post) {
    int same=rc.bound && rc.nparts==nparts && rc.shard==shard && rc.rb==rb && rc.total==total;
    for(int64_t i=0;same && i<nparts;i++)
        same=rc.source[i].fd==fds[i] && rc.source[i].base==base[i] && rc.source[i].rows==rows[i];
    if(post && !same)return ESTALE;
    Source snapshot[PART_MAX];
    for(int64_t i=0;i<nparts;i++) {
        struct stat st;if(fstat(fds[i],&st))return errno;
        if(!S_ISREG(st.st_mode)||base[i]<0||rows[i]<0||base[i]>st.st_size||rows[i]>(st.st_size-base[i])/rb)return EINVAL;
        Source v={.fd=fds[i],.base=base[i],.rows=rows[i],.dev=st.st_dev,.ino=st.st_ino,.size=st.st_size,
                  .mt=st.st_mtim,.ct=st.st_ctim};
        if(same) {
            Source *o=rc.source+i;
            if(o->dev!=v.dev||o->ino!=v.ino||o->size!=v.size||o->mt.tv_sec!=v.mt.tv_sec||o->mt.tv_nsec!=v.mt.tv_nsec||
               o->ct.tv_sec!=v.ct.tv_sec||o->ct.tv_nsec!=v.ct.tv_nsec)return ESTALE;
        }
        snapshot[i]=v;
    }
    if(!same) {
        int was=rc.bound;
        if(!rc.arena || rc.rb!=rb){int e=allocate(rb);if(e)return e;}
        else {memset(rc.buckets,0xff,(size_t)rc.nb*sizeof(uint32_t));rc.used=0;rc.head=rc.tail=NIL;rc.s[3]=0;}
        if(was)rc.s[12]++;
        memcpy(rc.source,snapshot,(size_t)nparts*sizeof(Source));
        rc.nparts=nparts;rc.shard=shard;rc.rb=rb;rc.total=total;rc.bound=1;
    }
    return 0;
}
int ple_gather_ids(const int64_t *ids,int64_t n,const int32_t *fds,const int64_t *base,const int64_t *rows,
                  int64_t nparts,int64_t shard,int64_t rb,int64_t total,char *out,int max_threads,int min_runs,int64_t *stats3) {
    pthread_once(&rc_once,initialize);pthread_mutex_lock(&rc_mu);
    int err=rc_init_error;idpos_t *ip=NULL;char *mem=NULL;
    rc.s[10]++;
    if(err)goto finish;
    if(n<0||n>MAX_ROWS||nparts<1||nparts>PART_MAX||shard<1||rb<1||rb>(1<<20)||total<1||
       max_threads<1||max_threads>256||min_runs<1||!fds||!base||!rows||(n&&(!ids||!out))) {err=EINVAL;goto finish;}
    if(!n){if(stats3)memset(stats3,0,3*sizeof(int64_t));goto finish;}
    rc.s[17]+=(uint64_t)n;
    if(!rc.s[0]) {
        int64_t s[3];err=ple_uncached_gather_ids(ids,n,fds,base,rows,nparts,shard,rb,total,out,max_threads,min_runs,s);
        if(!err){rc.s[4]+=s[0];rc.s[6]+=s[0];rc.s[8]+=s[1];rc.s[9]+=s[2];if(stats3)memcpy(stats3,s,sizeof(s));}
        goto finish;
    }
    // Validate the entire call before publishing bytes or changing cache contents.
    for(int64_t i=0;i<n;i++) {
        if(ids[i]<0||ids[i]>=total){err=-3;goto finish;}
        int64_t s=ids[i]/shard;if(s>=nparts||ids[i]%shard>=rows[s]){err=-4;goto finish;}
    }
    err=bind_sources(fds,base,rows,nparts,shard,rb,total,0);if(err)goto finish;
    ip=malloc((size_t)n*sizeof(idpos_t));if(!ip){err=ENOMEM;goto finish;}
    for(int64_t i=0;i<n;i++){ip[i].id=ids[i];ip[i].pos=i;}
    qsort(ip,(size_t)n,sizeof(idpos_t),cmp_idpos);
    int64_t u=0;for(int64_t i=0;i<n;i++)if(!i||ip[i].id!=ip[i-1].id)u++;
    // Scratch is call-scoped (like native gather), never retained in the cache.
    uint64_t need=(uint64_t)u*(2*(uint64_t)rb+3*sizeof(int64_t));
    if(need>SIZE_MAX){err=ENOMEM;goto finish;}
    mem=malloc((size_t)need);if(!mem){err=ENOMEM;goto finish;}
    int64_t *uid=(int64_t*)mem,*miss=uid+u,*mpos=miss+u;
    char *ubuf=(char*)(mpos+u),*mbuf=ubuf+(uint64_t)u*rb;
    int64_t k=0,m=0,hits=0;
    for(int64_t i=0;i<n;i++)if(!i||ip[i].id!=ip[i-1].id){
        int64_t id=ip[i].id;uid[k]=id;uint32_t slot=lookup(id);
        if(slot==NIL){miss[m]=id;mpos[m++]=k;}
        else{memcpy(ubuf+(uint64_t)k*rb,rc.data+(uint64_t)slot*rb,(size_t)rb);hits++;}
        k++;
    }
    int64_t s[3]={0,0,0};
    if(m)err=ple_uncached_gather_ids(miss,m,fds,base,rows,nparts,shard,rb,total,mbuf,max_threads,min_runs,s);
    if(err)goto finish;
    err=bind_sources(fds,base,rows,nparts,shard,rb,total,1);if(err)goto finish;
    for(int64_t j=0;j<m;j++) {
        char *data=mbuf+(uint64_t)j*rb;
        memcpy(ubuf+(uint64_t)mpos[j]*rb,data,(size_t)rb);insert(miss[j],data);
    }
    k=-1;
    for(int64_t i=0;i<n;i++){
        if(!i||ip[i].id!=ip[i-1].id)k++;
        memcpy(out+(uint64_t)ip[i].pos*rb,ubuf+(uint64_t)k*rb,(size_t)rb);
    }
    rc.s[4]+=u;rc.s[5]+=hits;rc.s[6]+=m;rc.s[8]+=s[1];rc.s[9]+=s[2];
    if(stats3){stats3[0]=u;stats3[1]=s[1];stats3[2]=s[2];}
finish:
    if(err)rc.s[11]++;
    free(mem);free(ip);publish();pthread_mutex_unlock(&rc_mu);return err;
}
