// kernels.cuh — CUDA-ядра pearlhash: шум низкого ранга и NoisyGEMM с jackpot.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "blake3.h"
#include "pearl.h"

namespace kern {

struct Seed { uint32_t w[8]; };      // ключ blake3 (LE слова)
struct Bound { uint32_t w[8]; };     // граница, LE слова
static constexpr uint32_t SINGLE_KEYED = b3::CHUNK_START | b3::CHUNK_END | b3::ROOT | b3::KEYED;

// метка "A_tensor"/"B_tensor", дополненная нулями до 32 байт → 8 LE слов
__device__ __forceinline__ void label_words(bool isA, uint32_t *m) {
  m[0] = isA ? 0x65745f41u : 0x65745f42u;  // "A_te" / "B_te"
  m[1] = 0x726f736eu;                       // "nsor"
  m[2] = m[3] = m[4] = m[5] = m[6] = m[7] = 0;
}

// get_random_hash(index, label, key, slot): msg = 8×i32 (slot = 1+index) | label
__device__ __forceinline__ void rand_hash(const Seed &key, bool isA, int slot, uint32_t index, uint32_t out[8]) {
  uint32_t m[16] = {0};
  m[slot] = 1 + index;
  label_words(isA, m + 8);
  b3::compress(key.w, m, 0, 64, SINGLE_KEYED, out);
}

// E (rows × r): (byte & 63) - 32; поток на хэш (32 элемента)
__global__ void k_uniform(int8_t *E, int rows, int r, Seed key, bool isA) {
  uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= (uint32_t)rows * r / 32) return;
  uint32_t h[8];
  rand_hash(key, isA, 0, b, h);
  uint32_t *dst = (uint32_t *)(E + (size_t)b * 32);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    // побайтно x-32 для x = byte&63 ∈ [0,63] без межбайтовых переносов:
    // x>=32 → x^0x20; x<32 → (x^0x20)|0xC0
    uint32_t v = h[i] & 0x3f3f3f3fu, lo = ~v & 0x20202020u;
    dst[i] = (v ^ 0x20202020u) | (lo << 1) | (lo << 2);
  }
}

// перестановка: k пар (first, second), поток на хэш (8 строк)
__global__ void k_perm(uint8_t *first, uint8_t *second, int k, int r, Seed key, bool isA,
                       unsigned *reset = nullptr) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (reset && i == 0) *reset = 0;  // счётчик удач попытки обнуляется здесь (без memset)
  if (i >= (uint32_t)(k + 7) / 8) return;
  uint32_t h[8];
  rand_hash(key, isA, 1, i, h);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    uint32_t l = i * 8 + j;
    if (l >= (uint32_t)k) break;
    uint32_t u = h[j], f = u & (r - 1);
    uint32_t s = f ^ (1 + __umulhi((uint32_t)(r - 1), u));
    first[l] = (uint8_t)f;
    second[l] = (uint8_t)s;
  }
}

// Панельная раскладка A' и B' (v0.21): k режется на панели по PANEL = 64 байта (= BK), панель
// хранит все строки подряд: элемент (row, col) лежит в (col/64)·rows·64 + row·64 + col%64.
// Тогда стадия GEMM (BM строк × 64 байта) — один непрерывный кусок, и каждая загрузка warp
// (32 × 16 Б = 512 Б) читает 4 целые строки кэша вместо 8 половинок — вдвое меньше запросов к L1/TEX.
// Базовые A и B (для доказательств) остаются построчными на хосте.
static constexpr int PANEL = 64;
// v0.24: внутри строки панели 16-байтные сегменты уже переставлены так же, как в smem
// (seg ^ ((row >> 1) & 3), см. sw): стадия копируется в smem как есть — потоками (cp.async /
// регистры) или одной массовой копией TMA (cp.async.bulk) — и ldmatrix читает её без конфликтов.
// индекс char4 номер q строки row (row уже учтён в базе dst + row·PANEL) при rows строках
__device__ __forceinline__ size_t panel_q(int q, int rows, int row) {
  const int c = q % (PANEL / 4), seg = (c >> 2) ^ ((row >> 1) & 3);
  return (size_t)(q / (PANEL / 4)) * rows * (PANEL / 4) + seg * 4 + (c & 3);
}

// dst[i][l] = src[i][l] + E[i][first[l]] - E[i][second[l]]; блок на строку
__global__ void k_apply(int8_t *dst, const int8_t *src, const int8_t *E, const uint8_t *first,
                        const uint8_t *second, int k, int r) {
  extern __shared__ int8_t e_row[];
  size_t row = blockIdx.x;
  for (int c = threadIdx.x; c < r; c += blockDim.x) e_row[c] = E[row * r + c];
  __syncthreads();
  const char4 *s4 = (const char4 *)(src + row * k);
  char4 *d4 = (char4 *)(dst + row * PANEL);  // панельная раскладка (v0.21), см. panel_at
  const uchar4 *f4 = (const uchar4 *)first, *g4 = (const uchar4 *)second;
  for (int q = threadIdx.x; q < k / 4; q += blockDim.x) {
    char4 a = s4[q];
    uchar4 f = f4[q], g = g4[q];
    char4 o;
    o.x = (int8_t)(a.x + e_row[f.x] - e_row[g.x]);
    o.y = (int8_t)(a.y + e_row[f.y] - e_row[g.y]);
    o.z = (int8_t)(a.z + e_row[f.z] - e_row[g.z]);
    o.w = (int8_t)(a.w + e_row[f.w] - e_row[g.w]);
    d4[panel_q(q, gridDim.x, (int)row)] = o;
  }
}

// A' = A + шум одной строкой на блок, всё в одном ядре:
// - строка E_AL (r значений) считается здесь же (r/32 хэшей blake3);
// - A не хранится: чанк 0 (idx < 1024) = rnd_val(seedN, nonce*1024+idx), остальное gen_val(seedA, idx).
// dense=false — «разреженные» A: нули, кроме CHUNK0_RAND значений чанка 0 (доказательство жмётся gzip)
__global__ void k_apply_fused(int8_t *dst, uint32_t seedA, uint64_t seedN, uint64_t nonce, Seed key,
                              const uint8_t *first, const uint8_t *second, int k, int r, bool dense) {
  extern __shared__ int8_t e_row[];
  const size_t row = blockIdx.x;
  for (int hb = threadIdx.x; hb < r / 32; hb += blockDim.x) {
    uint32_t h[8];
    rand_hash(key, true, 0, (uint32_t)(row * r / 32 + hb), h);
    uint32_t *d = (uint32_t *)(e_row + hb * 32);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      uint32_t v = h[i] & 0x3f3f3f3fu, lo = ~v & 0x20202020u;
      d[i] = (v ^ 0x20202020u) | (lo << 1) | (lo << 2);
    }
  }
  __syncthreads();
  char4 *d4 = (char4 *)(dst + row * PANEL);  // панельная раскладка (v0.21)
  const uchar4 *f4 = (const uchar4 *)first, *g4 = (const uchar4 *)second;
  for (int q = threadIdx.x; q < k / 4; q += blockDim.x) {
    uint32_t idx = (uint32_t)(row * k) + 4 * q;
    int8_t a[4];
    if (idx < 1024) {
#pragma unroll
      for (int e = 0; e < 4; e++) a[e] = pearl::chunk0_val(seedN, nonce, idx + e, dense);
    } else {
#pragma unroll
      for (int e = 0; e < 4; e++) a[e] = dense ? pearl::gen_val(seedA, idx + e) : 0;
    }
    uchar4 f = f4[q], g = g4[q];
    char4 o;
    o.x = (int8_t)(a[0] + e_row[f.x] - e_row[g.x]);
    o.y = (int8_t)(a[1] + e_row[f.y] - e_row[g.y]);
    o.z = (int8_t)(a[2] + e_row[f.z] - e_row[g.z]);
    o.w = (int8_t)(a[3] + e_row[f.w] - e_row[g.w]);
    d4[panel_q(q, gridDim.x, (int)row)] = o;
  }
}

// ---------------- NoisyGEMM ----------------
// Блок BM×128 (BM = 128 или 256), warp по 64×64 ((BM/64)×2 warp), шаг K 64 байта.
// Поток warp держит 4×8 фрагментов m16n8 = 128 аккумуляторов = ровно один тайл
// rows {g+8a}, cols {2(l%4)+8b+{0,1}} — шаблоны ROWS/COLS_PATTERN из pearl.h.
static constexpr int BN = 128, BK = 64;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
static constexpr int STAGES = 2;  // Turing: загрузки через регистры
#else
static constexpr int STAGES = 3;  // cp.async
#endif
// размер dynamic smem на хосте (STAGES зависит от архитектуры)
inline int gemm_smem(int bm, int major) { return (major >= 8 ? 3 : 2) * (bm + BN) * BK; }
inline int gemm_threads_rt(int bm, int wn) { return 32 * (bm / 64) * (BN / wn); }
struct Hit { uint32_t row0, col0; };

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}
// TMA (sm_90+, на нашем риге — RTX 5060 sm_120): массовая копия global → smem одним потоком,
// завершение — по mbarrier (ожидаемое число байт). На старых архитектурах не вызывается.
__device__ __forceinline__ void mbar_init(uint32_t bar) {
#if __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(bar) : "memory");
#endif
}
__device__ __forceinline__ void tma_load(uint32_t dst, const void *src, uint32_t bytes, uint32_t bar) {
#if __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
#endif
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t bytes) {
#if __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
#endif
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
               "@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}\n" ::"r"(bar), "r"(parity) : "memory");
#endif
}
__device__ __forceinline__ void fence_async_smem() {
#if __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

template <int N> __device__ __forceinline__ void cp_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// D(16×8,s32) += A(16×32,s8) · B(32×8,s8). Без volatile: чистая операция над регистрами,
// компилятор может переставлять mma относительно ldmatrix (те остаются volatile)
__device__ __forceinline__ void mma16832(int *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else  // Turing: 4 × m8n8k16 с теми же фрагментами
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(b0));
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a[2]), "r"(b1));
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};\n"
               : "+r"(c[2]), "+r"(c[3]) : "r"(a[1]), "r"(b0));
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};\n"
               : "+r"(c[2]), "+r"(c[3]) : "r"(a[3]), "r"(b1));
#endif
}

// Turing: половина m16n8k32 — шаг k16 (h = 0: a[0], a[1] × b0; h = 1: a[2], a[3] × b1).
// Цикл по фрагментам делает сначала все h = 0, потом все h = 1: зависимые IMMA одного аккумулятора
// разнесены на десятки инструкций (профиль v0.20: простой «fixed latency dependency» 31%).
__device__ __forceinline__ void mma_half(int *c, const uint32_t *a, uint32_t b0, uint32_t b1, int h) {
#if __CUDA_ARCH__ < 800
  const uint32_t bb = h ? b1 : b0;
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};\n"
      : "+r"(c[0]), "+r"(c[1]) : "r"(a[h ? 2 : 0]), "r"(bb));
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};\n"
      : "+r"(c[2]), "+r"(c[3]) : "r"(a[h ? 3 : 1]), "r"(bb));
#else
  if (h == 0) mma16832(c, a, b0, b1);
#endif
}
// все фрагменты warp: acc[4][NJ] += af[4] × bf[NJ]
template <int NJ>
__device__ __forceinline__ void mma_all(int (*acc)[NJ][4], const uint32_t (*af)[4], const uint32_t (*bf)[2]) {
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int j = 0; j < NJ; j++) mma_half(acc[i][j], af[i], bf[j][0], bf[j][1], h);
}

// Микробенчмарк тензорных ядер: 4 warp в блоке, 8 независимых цепочек mma на warp, данные в
// регистрах. RND=false — постоянные операнды (как бенчмарк v0.1), RND=true — операнды меняются
// каждые 8 mma (по энергии ближе к настоящим данным). Число warp на SM задаётся сеткой.
// mma без volatile: при CH=1 цепочка зависимых mma всё равно не сворачивается (asm с "+r")
// CH — число независимых цепочек аккумуляторов на warp (ILP)
template <bool RND, int CH = 8>
__global__ void __launch_bounds__(128) k_bench(int iters, int *out, uint32_t seed) {
  uint32_t a[4], b0, b1;
  a[0] = (threadIdx.x * 0x01010101u) ^ seed ^ (blockIdx.x * 0x9E3779B9u);
  a[1] = a[0] * 3; a[2] = a[0] * 5; a[3] = a[0] * 7;
  b0 = a[0] ^ 0x5a5a5a5au; b1 = b0 * 9;
  int acc[CH][4] = {};
#pragma unroll 1
  for (int i = 0; i < iters; i++) {
    if (RND) {
      a[0] += 0x9E3779B9u; a[1] += 0x7F4A7C15u; a[2] += 0x85EBCA6Bu; a[3] += 0xC2B2AE35u;
      b0 += 0x27D4EB2Fu; b1 += 0x165667B1u;
    }
#pragma unroll
    for (int r = 0; r < 8 / CH; r++)  // всего 8 mma на итерацию при любом CH
#pragma unroll
      for (int c = 0; c < CH; c++) mma16832(acc[c], a, b0, b1);
  }
  int x = 0;
#pragma unroll
  for (int c = 0; c < CH; c++) x ^= acc[c][0] ^ acc[c][1] ^ acc[c][2] ^ acc[c][3];
  if (x == 0x7fffffff) out[0] = x;  // не даём компилятору выкинуть работу
}

// смещение 16-байтного сегмента в строке 64 байта со swizzle (без конфликтов банков)
__device__ __forceinline__ int sw(int row, int seg) { return row * BK + 16 * (seg ^ ((row >> 1) & 3)); }

// Адресация загрузок (v0.19): поток t берёт сегмент t%4 строки t/4, следующий проход — на
// NTHR/4 строк ниже (кратно 8 → swizzle тот же), поэтому смещения в smem и global считаются
// один раз на поток, а в цикле остаются константы (k = pearl::K известно при компиляции).
__device__ __forceinline__ void cp16u(uint32_t s, const void *g) {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g));
#else
  uint4 v = *(const uint4 *)g;
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(s), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
#endif
}
__device__ __forceinline__ void sts16(uint32_t s, const uint4 &v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(s), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void ldsm4u(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// загрузка тайла ROWS×64 байт: s — адрес сегмента потока в стадии, g — его адрес в global
template <int ROWS, int NTHR>
__device__ __forceinline__ void load_tile(uint32_t s, const int8_t *g) {
  constexpr int RS = NTHR / 4;
#pragma unroll
  for (int i = 0; i < ROWS / RS; i++) cp16u(s + i * RS * BK, g + i * RS * PANEL);
}

// Turing: следующая стадия грузится в регистры до вычислений и кладётся в smem после
template <int SA, int SB> struct RegStage { uint4 a[SA], b[SB]; };
template <int NTHR, int SA, int SB>
__device__ __forceinline__ void load_regs(RegStage<SA, SB> &rg, const int8_t *gA, const int8_t *gB) {
  constexpr int RS = NTHR / 4;
#pragma unroll
  for (int i = 0; i < SA; i++) rg.a[i] = *(const uint4 *)(gA + i * RS * PANEL);
#pragma unroll
  for (int i = 0; i < SB; i++) rg.b[i] = *(const uint4 *)(gB + i * RS * PANEL);
}
template <int NTHR, int SA, int SB>
__device__ __forceinline__ void store_regs(const RegStage<SA, SB> &rg, uint32_t sA, uint32_t sB) {
  constexpr int RS = NTHR / 4;
#pragma unroll
  for (int i = 0; i < SA; i++) sts16(sA + i * RS * BK, rg.a[i]);
#pragma unroll
  for (int i = 0; i < SB; i++) sts16(sB + i * RS * BK, rg.b[i]);
}

// A' (m×k), B'^T (n×k): int8, k-непрерывные. k кратно BK, r кратно BK.
// Сетка одномерная, порядок блоков сгруппирован по gm строк блоков (v0.20 — параметр, подбирается) —
// одновременно работающие блоки делят панели A и B в L2.
// MODE — диагностика узких мест (0 = рабочий режим):
// 1 — без загрузок global→smem в цикле; 2 — ещё и без барьеров; 3 — ещё и без ldmatrix
// (фрагменты загружены один раз). Результат в режимах 1–3 бессмысленный, важна только скорость.
// WN — ширина куска warp: 64 (тайл jackpot 8×16, 128 аккумуляторов на поток) или
// 32 (тайл 8×8, 64 аккумулятора, вдвое больше warp на SM). Кусок warp — 64×WN.
// PIPE — конвейер фрагментов через границу стадий (v0.15); false — прежний цикл (v0.7–v0.13)
// TMA (v0.24) — стадии грузит один поток массовыми копиями TMA с mbarrier (только sm_90+ и MODE 0)
template <int BM, int WN = 64, int MODE = 0, bool PIPE = true, bool TMA = false>
__global__ void __launch_bounds__(32 * (BM / 64) * (BN / WN), BM == 128 ? 2 : 1)
k_gemm(const int8_t *__restrict__ A, const int8_t *__restrict__ Bt, int m, int n, int gm,
       Seed a_seed, Bound bnd, unsigned *hit_count, Hit *hits, int max_hits, uint32_t *digest) {
  constexpr int WARPS_N = BN / WN, NTHR = 32 * (BM / 64) * WARPS_N, NJ = WN / 8;
  constexpr int k = pearl::K, r = pearl::R;  // v0.19: константы — без деления и умножений в цикле
  extern __shared__ __align__(128) int8_t smem[];  // STAGES × A BM×BK | STAGES × B BN×BK
  const uint32_t sbase = smem_u32(smem);
  auto sA = [&](int st) { return sbase + st * (BM * BK); };
  auto sB = [&](int st) { return sbase + STAGES * BM * BK + st * (BN * BK); };
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int wm = (warp / WARPS_N) * 64, wn = (warp % WARPS_N) * WN;
  const int nbm = m / BM, nbn = n / BN, gsz = gm * nbn, pid = blockIdx.x;
  const int first = (pid / gsz) * gm, gs = min(nbm - first, gm), loc = pid % gsz;
  const int bm = first + loc % gs, bn = loc / gs;
  // сегмент загрузки потока: копия линейная — swizzle уже в глобальной раскладке (v0.24)
  const int lrow = t >> 2, lseg = t & 3, loff = lrow * BK + lseg * 16;
  // панельная раскладка: стадия s блока — непрерывный кусок BM×64 по адресу s·(m·64) + bm·BM·64
  static_assert(PANEL == BK, "панель = шаг k");
  const size_t pA = (size_t)m * BK, pB = (size_t)n * BK;  // байт на панель
  const int8_t *gA = A + ((size_t)bm * BM + lrow) * BK + lseg * 16;
  const int8_t *gB = Bt + ((size_t)bn * BN + lrow) * BK + lseg * 16;

  int acc[4][NJ][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[i][j][c] = 0;
  uint32_t jp[16];
#pragma unroll
  for (int i = 0; i < 16; i++) jp[i] = 0;

  constexpr int KT = k / BK, per_chunk = r / BK;
  static_assert(k % BK == 0 && r % BK == 0 && KT >= STAGES, "k, r кратны BK");
#if __CUDA_ARCH__ >= 900
  constexpr bool use_tma = TMA && MODE == 0;
#else
  constexpr bool use_tma = false;
#endif
  __shared__ __align__(8) uint64_t mbar[STAGES];
  const uint32_t bar0 = smem_u32(mbar);
  const int8_t *tA = A + (size_t)bm * BM * BK, *tB = Bt + (size_t)bn * BN * BK;  // начало стадии блока
  // загрузить стадию номер kidx в буфер buf
  auto stage_load = [&](int buf, int kidx) {
    if constexpr (use_tma) {
      if (t == 0) {
        fence_async_smem();  // буфер только что читали ldmatrix (обычный доступ) — порядок с TMA
        mbar_expect(bar0 + 8 * buf, (BM + BN) * BK);
        tma_load(sA(buf), tA + kidx * pA, BM * BK, bar0 + 8 * buf);
        tma_load(sB(buf), tB + kidx * pB, BN * BK, bar0 + 8 * buf);
      }
    } else {
      load_tile<BM, NTHR>(sA(buf) + loff, gA + kidx * pA);
      load_tile<BN, NTHR>(sB(buf) + loff, gB + kidx * pB);
    }
  };
  auto stage_commit = [&]() { if constexpr (!use_tma) cp_commit(); };
  // дождаться стадии kidx (в буфере buf); после — __syncthreads у вызывающего
  auto stage_wait = [&](int buf, int kidx) {
    if constexpr (use_tma) mbar_wait(bar0 + 8 * buf, (kidx / STAGES) & 1);
    else cp_wait<STAGES - 2>();
  };
  if constexpr (use_tma) {
    if (t == 0) {
#pragma unroll
      for (int s = 0; s < STAGES; s++) mbar_init(bar0 + 8 * s);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
  }
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KT) stage_load(s, s);
    stage_commit();
  }

  // адреса ldmatrix: A — строка (lane%8)+8*((lane/8)%2), сегмент lane/16;
  //                  B — строка (lane%8)+8*(lane/16),    сегмент (lane/8)%2
  const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1), a_seg = lane >> 4;
  const int b_row = (lane & 7) + 8 * (lane >> 4), b_seg = (lane >> 3) & 1;
  // смещения фрагментов шагов k32 (ks = 0, 1); фрагмент i/j на 16·i строк ниже — swizzle тот же
  const int aoff[2] = {sw(wm + a_row, a_seg), sw(wm + a_row, 2 + a_seg)};
  const int boff[2] = {sw(wn + b_row, b_seg), sw(wn + b_row, 2 + b_seg)};

  if constexpr (PIPE) {
  // Конвейер по шагам k32 (2 шага на стадию BK=64), фрагменты в двух буферах:
  //   шаг 0 стадии: ldmatrix шага 1 → загрузка будущей стадии → mma шага 0;
  //   шаг 1 стадии: ожидание + барьер → ldmatrix шага 0 СЛЕДУЮЩЕЙ стадии → mma шага 1.
  // Так барьер и задержка ldmatrix перекрываются уже выданными mma, а не стоят перед ними.
  uint32_t af[2][4][4], bf[2][NJ][2];
  auto ldfr = [&](int b, int st, int ks) {
#pragma unroll
    for (int i = 0; i < 4; i++)
      ldsm4u(af[b][i][0], af[b][i][1], af[b][i][2], af[b][i][3], sA(st) + aoff[ks] + 16 * i * BK);
#pragma unroll
    for (int j = 0; j < NJ / 2; j++)
      ldsm4u(bf[b][2 * j][0], bf[b][2 * j][1], bf[b][2 * j + 1][0], bf[b][2 * j + 1][1],
             sB(st) + boff[ks] + 16 * j * BK);
  };
  auto mmas = [&](int b) { mma_all<NJ>(acc, af[b], bf[b]); };

  stage_wait(0, 0);
  __syncthreads();  // стадия 0 готова
  ldfr(0, 0, 0);
  if constexpr (MODE >= 3) ldfr(1, 0, 1);

  int st = 0;
  for (int kt = 0; kt < KT; kt++) {
    const int nk = kt + STAGES - 1, ns = st == 0 ? STAGES - 1 : st - 1;  // ns — буфер стадии kt-1
    const int st1 = st == STAGES - 1 ? 0 : st + 1;
    if constexpr (MODE < 3) ldfr(1, st, 1);
#if __CUDA_ARCH__ >= 800
    if constexpr (MODE < 1)
      if (nk < KT) stage_load(ns, nk);
    stage_commit();
    mmas(0);
#else
    constexpr int SA = BM * 4 / NTHR, SB = BN * 4 / NTHR;
    [[maybe_unused]] RegStage<SA, SB> rg;
    if constexpr (MODE < 1)
      if (nk < KT) load_regs<NTHR>(rg, gA + nk * pA, gB + nk * pB);
    mmas(0);
    if constexpr (MODE < 1)
      if (nk < KT) store_regs<NTHR>(rg, sA(ns) + loff, sB(ns) + loff);
#endif
    if (kt + 1 < KT) {
      if constexpr (MODE < 2) {
        stage_wait(st1, kt + 1);
        __syncthreads();  // стадия kt+1 готова; стадию kt все уже прочитали в ldfr выше
      }
      if constexpr (MODE < 3) ldfr(0, st1, 0);
    }
    mmas(1);
    if ((kt + 1) % per_chunk == 0) {  // конец чанка r: XOR тайла, rotl13, сдвиг jackpot
      uint32_t xp[4];  // XOR деревом: 4 независимые цепочки вместо одной длинной
#pragma unroll
      for (int c = 0; c < 4; c++) {
        xp[c] = 0;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int j = 0; j < NJ; j++) xp[c] ^= (uint32_t)acc[i][j][c];
      }
      uint32_t x = (xp[0] ^ xp[1]) ^ (xp[2] ^ xp[3]);
      uint32_t nv = ((jp[0] << 13) | (jp[0] >> 19)) ^ x;
#pragma unroll
      for (int i = 0; i < 15; i++) jp[i] = jp[i + 1];
      jp[15] = nv;
    }
    st = st1;
  }

  } else {
  uint32_t af[2][4][4], bf[2][NJ][2];  // фрагменты двух шагов k32
  auto load_frags = [&](int st) {
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
#pragma unroll
        for (int i = 0; i < 4; i++)
          ldsm4u(af[ks][i][0], af[ks][i][1], af[ks][i][2], af[ks][i][3], sA(st) + aoff[ks] + 16 * i * BK);
#pragma unroll
        for (int j = 0; j < NJ / 2; j++)
          ldsm4u(bf[ks][2 * j][0], bf[ks][2 * j][1], bf[ks][2 * j + 1][0], bf[ks][2 * j + 1][1],
                 sB(st) + boff[ks] + 16 * j * BK);
      }
  };
  if constexpr (MODE >= 3) { cp_wait<0>(); __syncthreads(); load_frags(0); }

  int st = 0;
  for (int kt = 0; kt < KT; kt++) {
    if constexpr (MODE < 2) {
      stage_wait(st, kt);
      __syncthreads();  // стадия kt готова; буфер стадии kt-1 больше никто не читает
    }
    const int nk = kt + STAGES - 1, ns = st == 0 ? STAGES - 1 : st - 1;
#if __CUDA_ARCH__ >= 800
    if constexpr (MODE < 1)
      if (nk < KT) stage_load(ns, nk);
    stage_commit();
#else
    constexpr int SA = BM * 4 / NTHR, SB = BN * 4 / NTHR;
    [[maybe_unused]] RegStage<SA, SB> rg;
    if constexpr (MODE < 1)
      if (nk < KT) load_regs<NTHR>(rg, gA + nk * pA, gB + nk * pB);
#endif
    {  // два шага k32: сначала все ldmatrix обоих шагов, потом все mma
      if constexpr (MODE < 3) load_frags(st);
#pragma unroll
      for (int ks = 0; ks < 2; ks++) mma_all<NJ>(acc, af[ks], bf[ks]);
    }
    if ((kt + 1) % per_chunk == 0) {  // конец чанка r: XOR тайла, rotl13, сдвиг jackpot
      uint32_t xp[4];  // XOR деревом: 4 независимые цепочки вместо одной длинной
#pragma unroll
      for (int c = 0; c < 4; c++) {
        xp[c] = 0;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int j = 0; j < NJ; j++) xp[c] ^= (uint32_t)acc[i][j][c];
      }
      uint32_t x = (xp[0] ^ xp[1]) ^ (xp[2] ^ xp[3]);
      uint32_t nv = ((jp[0] << 13) | (jp[0] >> 19)) ^ x;
#pragma unroll
      for (int i = 0; i < 15; i++) jp[i] = jp[i + 1];
      jp[15] = nv;
    }
#if __CUDA_ARCH__ < 800
    if constexpr (MODE < 1)
      if (nk < KT) store_regs<NTHR>(rg, sA(ns) + loff, sB(ns) + loff);
#endif
    st = st == STAGES - 1 ? 0 : st + 1;
  }

  }

  // позиция i после N сдвигов хранит слово (i+N)%16
  const int nch = k / r;
  uint32_t msg[16];
#pragma unroll
  for (int i = 0; i < 16; i++) msg[i] = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    int idx = (i + nch) & 15;
#pragma unroll
    for (int q = 0; q < 16; q++) if (q == idx) msg[q] = jp[i];
  }
  uint32_t h[8];
  b3::compress(a_seed.w, msg, 0, 64, SINGLE_KEYED, h);
  bool le = true;  // h (LE 256 бит) <= bnd
#pragma unroll
  for (int w = 7; w >= 0; w--) {
    if (h[w] != bnd.w[w]) { le = h[w] < bnd.w[w]; break; }
  }
  if (digest) {
#pragma unroll
    for (int w = 0; w < 8; w++) atomicXor(&digest[w], h[w]);
  }
  if (le) {
    unsigned i = atomicAdd(hit_count, 1u);
    if ((int)i < max_hits) {
      hits[i].row0 = bm * BM + wm + (lane >> 2);
      hits[i].col0 = bn * BN + wn + 2 * (lane & 3);
    }
  }
}

// ---------------- v0.25: k_gemm3 — 3 блока на SM (12 warp) ----------------
// Тот же блок 128×128, warp 64×64, тайл jackpot 8×16 и та же раскладка A'/B', но:
// - __launch_bounds__(128, 3): не больше ~168 регистров на поток (128 аккумуляторов + 16 jackpot
//   + фрагменты); фрагменты B грузятся по два столбца j прямо перед своими mma, а не все сразу;
// - 2 стадии по BK=64 (32 КБ smem на блок): 3 блока × 33 КБ влезают в 100 КБ SM (sm_86/89/120).
// Смысл: у основного ядра 2 warp на планировщик, и простои (ожидание ldmatrix/барьера) нечем
// закрыть; здесь их 3. Только sm_80+ (cp.async); на Turing smem SM 64 КБ — не помещается.
static constexpr int STAGES3 = 2;
inline int gemm3_smem() { return STAGES3 * (128 + BN) * BK; }
__global__ void __launch_bounds__(128, 3)
k_gemm3(const int8_t *__restrict__ A, const int8_t *__restrict__ Bt, int m, int n, int gm,
        Seed a_seed, Bound bnd, unsigned *hit_count, Hit *hits, int max_hits, uint32_t *digest) {
#if __CUDA_ARCH__ >= 800
  constexpr int BM = 128, NTHR = 128, k = pearl::K, r = pearl::R, KT = k / BK, per_chunk = r / BK;
  extern __shared__ __align__(128) int8_t smem[];
  const uint32_t sbase = smem_u32(smem);
  auto sA = [&](int st) { return sbase + st * (BM * BK); };
  auto sB = [&](int st) { return sbase + STAGES3 * BM * BK + st * (BN * BK); };
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int wm = (warp >> 1) * 64, wn = (warp & 1) * 64;
  const int nbm = m / BM, nbn = n / BN, gsz = gm * nbn, pid = blockIdx.x;
  const int first = (pid / gsz) * gm, gs = min(nbm - first, gm), loc = pid % gsz;
  const int bm = first + loc % gs, bn = loc / gs;
  const int loff = t * 16;  // линейная копия: swizzle уже в глобальной раскладке
  const uint32_t pA = (uint32_t)m * BK, pB = (uint32_t)n * BK;  // байт на панель (< 4 ГБ на матрицу)
  const int8_t *gA = A + (size_t)bm * BM * BK + loff;
  const int8_t *gB = Bt + (size_t)bn * BN * BK + loff;
  uint32_t oA = 0, oB = 0;  // смещение текущей панели

  int acc[4][8][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[i][j][c] = 0;
  // jackpot — кольцо в локальной памяти (динамический индекс): слово t%16 обновляется раз в 2 шага
  // k, а 16 регистров уходят на фрагменты. jackpot[t%16] = rotl13(jackpot[t%16]) ^ xor(тайл).
  uint32_t jp[16];
#pragma unroll
  for (int i = 0; i < 16; i++) jp[i] = 0;
  int ch = 0;

  const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1), a_seg = lane >> 4;
  const int b_row = (lane & 7) + 8 * (lane >> 4), b_seg = (lane >> 3) & 1;
  const int aoff[2] = {sw(wm + a_row, a_seg), sw(wm + a_row, 2 + a_seg)};
  const int boff[2] = {sw(wn + b_row, b_seg), sw(wn + b_row, 2 + b_seg)};

  load_tile<BM, NTHR>(sA(0) + loff, gA);
  load_tile<BN, NTHR>(sB(0) + loff, gB);
  cp_commit();
  int st = 0;
#pragma unroll 1
  for (int kt = 0; kt < KT; kt++) {
    cp_wait<0>();
    __syncthreads();  // стадия kt готова; буфер st^1 (стадия kt-1) все дочитали
    oA += pA; oB += pB;
    if (kt + 1 < KT) {
      load_tile<BM, NTHR>(sA(st ^ 1) + loff, gA + oA);
      load_tile<BN, NTHR>(sB(st ^ 1) + loff, gB + oB);
    }
    cp_commit();
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      uint32_t af[4][4];
#pragma unroll
      for (int i = 0; i < 4; i++)
        ldsm4u(af[i][0], af[i][1], af[i][2], af[i][3], sA(st) + aoff[ks] + 16 * i * BK);
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {  // два столбца j = 2jj, 2jj+1
        uint32_t b0, b1, b2, b3;
        ldsm4u(b0, b1, b2, b3, sB(st) + boff[ks] + 16 * jj * BK);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          mma16832(acc[i][2 * jj], af[i], b0, b1);
          mma16832(acc[i][2 * jj + 1], af[i], b2, b3);
        }
      }
    }
    if ((kt + 1) % per_chunk == 0) {  // конец чанка r: XOR тайла, rotl13, сдвиг jackpot
      uint32_t xp[4];
#pragma unroll
      for (int c = 0; c < 4; c++) {
        xp[c] = 0;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int j = 0; j < 8; j++) xp[c] ^= (uint32_t)acc[i][j][c];
      }
      uint32_t x = (xp[0] ^ xp[1]) ^ (xp[2] ^ xp[3]);
      const uint32_t v = jp[ch & 15];
      jp[ch & 15] = ((v << 13) | (v >> 19)) ^ x;
      ch++;
    }
    st ^= 1;
  }
  uint32_t h[8];
  b3::compress(a_seed.w, jp, 0, 64, SINGLE_KEYED, h);
  bool le = true;
#pragma unroll
  for (int w = 7; w >= 0; w--) {
    if (h[w] != bnd.w[w]) { le = h[w] < bnd.w[w]; break; }
  }
  if (digest) {
#pragma unroll
    for (int w = 0; w < 8; w++) atomicXor(&digest[w], h[w]);
  }
  if (le) {
    unsigned i = atomicAdd(hit_count, 1u);
    if ((int)i < max_hits) {
      hits[i].row0 = bm * BM + wm + (lane >> 2);
      hits[i].col0 = bn * BN + wn + 2 * (lane & 3);
    }
  }
#endif
}

}  // namespace kern
