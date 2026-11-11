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

// dst[i][l] = src[i][l] + E[i][first[l]] - E[i][second[l]]; блок на строку
__global__ void k_apply(int8_t *dst, const int8_t *src, const int8_t *E, const uint8_t *first,
                        const uint8_t *second, int k, int r) {
  extern __shared__ int8_t e_row[];
  size_t row = blockIdx.x;
  for (int c = threadIdx.x; c < r; c += blockDim.x) e_row[c] = E[row * r + c];
  __syncthreads();
  const char4 *s4 = (const char4 *)(src + row * k);
  char4 *d4 = (char4 *)(dst + row * k);
  const uchar4 *f4 = (const uchar4 *)first, *g4 = (const uchar4 *)second;
  for (int q = threadIdx.x; q < k / 4; q += blockDim.x) {
    char4 a = s4[q];
    uchar4 f = f4[q], g = g4[q];
    char4 o;
    o.x = (int8_t)(a.x + e_row[f.x] - e_row[g.x]);
    o.y = (int8_t)(a.y + e_row[f.y] - e_row[g.y]);
    o.z = (int8_t)(a.z + e_row[f.z] - e_row[g.z]);
    o.w = (int8_t)(a.w + e_row[f.w] - e_row[g.w]);
    d4[q] = o;
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
  char4 *d4 = (char4 *)(dst + row * k);
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
    d4[q] = o;
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

__device__ __forceinline__ void cp16(void *s, const void *g) {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(s)), "l"(g));
#else
  *(uint4 *)s = *(const uint4 *)g;
#endif
}
__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}
template <int N> __device__ __forceinline__ void cp_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void ldsm4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
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

// загрузка тайла ROWS×64 байт: ROWS*4 сегментов по 16 байт, NTHR потоков
template <int ROWS, int NTHR>
__device__ __forceinline__ void load_tile(int8_t *s, const int8_t *g, int k, int t) {
#pragma unroll
  for (int i = 0; i < ROWS * 4 / NTHR; i++) {
    int idx = t + i * NTHR, row = idx >> 2, seg = idx & 3;
    cp16(s + sw(row, seg), g + (size_t)row * k + seg * 16);
  }
}

// Turing: следующая стадия грузится в регистры до вычислений и кладётся в smem после
template <int SA, int SB> struct RegStage { uint4 a[SA], b[SB]; };
template <int NTHR, int SA, int SB>
__device__ __forceinline__ void load_regs(RegStage<SA, SB> &rg, const int8_t *gA, const int8_t *gB, int k, int t) {
#pragma unroll
  for (int i = 0; i < SA; i++) {
    int idx = t + i * NTHR, row = idx >> 2, seg = idx & 3;
    rg.a[i] = *(const uint4 *)(gA + (size_t)row * k + seg * 16);
  }
#pragma unroll
  for (int i = 0; i < SB; i++) {
    int idx = t + i * NTHR, row = idx >> 2, seg = idx & 3;
    rg.b[i] = *(const uint4 *)(gB + (size_t)row * k + seg * 16);
  }
}
template <int NTHR, int SA, int SB>
__device__ __forceinline__ void store_regs(const RegStage<SA, SB> &rg, int8_t *sA, int8_t *sB, int t) {
#pragma unroll
  for (int i = 0; i < SA; i++) {
    int idx = t + i * NTHR, row = idx >> 2, seg = idx & 3;
    *(uint4 *)(sA + sw(row, seg)) = rg.a[i];
  }
#pragma unroll
  for (int i = 0; i < SB; i++) {
    int idx = t + i * NTHR, row = idx >> 2, seg = idx & 3;
    *(uint4 *)(sB + sw(row, seg)) = rg.b[i];
  }
}

// A' (m×k), B'^T (n×k): int8, k-непрерывные. k кратно BK, r кратно BK.
// Сетка одномерная, порядок блоков сгруппирован по GROUP_M строк блоков —
// одновременно работающие блоки делят панели A и B в L2.
// MODE — диагностика узких мест (0 = рабочий режим):
// 1 — без загрузок global→smem в цикле; 2 — ещё и без барьеров; 3 — ещё и без ldmatrix
// (фрагменты загружены один раз). Результат в режимах 1–3 бессмысленный, важна только скорость.
// WN — ширина куска warp: 64 (тайл jackpot 8×16, 128 аккумуляторов на поток) или
// 32 (тайл 8×8, 64 аккумулятора, вдвое больше warp на SM). Кусок warp — 64×WN.
// PIPE — конвейер фрагментов через границу стадий (v0.15); false — прежний цикл (v0.7–v0.13)
template <int BM, int WN = 64, int MODE = 0, bool PIPE = true>
__global__ void __launch_bounds__(32 * (BM / 64) * (BN / WN), BM == 128 ? 2 : 1)
k_gemm(const int8_t *__restrict__ A, const int8_t *__restrict__ Bt, int m, int n, int k, int r,
       Seed a_seed, Bound bnd, unsigned *hit_count, Hit *hits, int max_hits, uint32_t *digest) {
  constexpr int WARPS_N = BN / WN, NTHR = 32 * (BM / 64) * WARPS_N, GROUP_M = 1024 / BM, NJ = WN / 8;
  extern __shared__ __align__(128) int8_t smem[];  // STAGES × (A BM×BK | B BN×BK)
  int8_t(*sA)[BM * BK] = (int8_t(*)[BM * BK])smem;
  int8_t(*sB)[BN * BK] = (int8_t(*)[BN * BK])(smem + STAGES * BM * BK);
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int wm = (warp / WARPS_N) * 64, wn = (warp % WARPS_N) * WN;
  const int nbm = m / BM, nbn = n / BN, gsz = GROUP_M * nbn, pid = blockIdx.x;
  const int first = (pid / gsz) * GROUP_M, gs = min(nbm - first, GROUP_M), loc = pid % gsz;
  const int bm = first + loc % gs, bn = loc / gs;
  const int8_t *gA = A + (size_t)bm * BM * k;
  const int8_t *gB = Bt + (size_t)bn * BN * k;

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

  const int KT = k / BK, per_chunk = r / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KT) { load_tile<BM, NTHR>(sA[s], gA + s * BK, k, t); load_tile<BN, NTHR>(sB[s], gB + s * BK, k, t); }
    cp_commit();
  }

  // адреса ldmatrix: A — строка (lane%8)+8*((lane/8)%2), сегмент lane/16;
  //                  B — строка (lane%8)+8*(lane/16),    сегмент (lane/8)%2
  const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1), a_seg = lane >> 4;
  const int b_row = (lane & 7) + 8 * (lane >> 4), b_seg = (lane >> 3) & 1;

  if constexpr (PIPE) {
  // Конвейер по шагам k32 (2 шага на стадию BK=64), фрагменты в двух буферах:
  //   шаг 0 стадии: ldmatrix шага 1 → загрузка будущей стадии → mma шага 0;
  //   шаг 1 стадии: ожидание + барьер → ldmatrix шага 0 СЛЕДУЮЩЕЙ стадии → mma шага 1.
  // Так барьер и задержка ldmatrix перекрываются уже выданными mma, а не стоят перед ними.
  uint32_t af[2][4][4], bf[2][NJ][2];
  auto ldfr = [&](int b, int st, int ks) {
#pragma unroll
    for (int i = 0; i < 4; i++)
      ldsm4(af[b][i][0], af[b][i][1], af[b][i][2], af[b][i][3], &sA[st][sw(wm + 16 * i + a_row, 2 * ks + a_seg)]);
#pragma unroll
    for (int j = 0; j < NJ / 2; j++)
      ldsm4(bf[b][2 * j][0], bf[b][2 * j][1], bf[b][2 * j + 1][0], bf[b][2 * j + 1][1],
            &sB[st][sw(wn + 16 * j + b_row, 2 * ks + b_seg)]);
  };
  auto mmas = [&](int b) {
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int j = 0; j < NJ; j++) mma16832(acc[i][j], af[b][i], bf[b][j][0], bf[b][j][1]);
  };

  cp_wait<STAGES - 2>();
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
      if (nk < KT) { load_tile<BM, NTHR>(sA[ns], gA + nk * BK, k, t); load_tile<BN, NTHR>(sB[ns], gB + nk * BK, k, t); }
    cp_commit();
    mmas(0);
#else
    constexpr int SA = BM * 4 / NTHR, SB = BN * 4 / NTHR;
    [[maybe_unused]] RegStage<SA, SB> rg;
    if constexpr (MODE < 1)
      if (nk < KT) load_regs<NTHR>(rg, gA + nk * BK, gB + nk * BK, k, t);
    mmas(0);
    if constexpr (MODE < 1)
      if (nk < KT) store_regs<NTHR>(rg, sA[ns], sB[ns], t);
#endif
    if (kt + 1 < KT) {
      if constexpr (MODE < 2) {
        cp_wait<STAGES - 2>();
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
          ldsm4(af[ks][i][0], af[ks][i][1], af[ks][i][2], af[ks][i][3],
                &sA[st][sw(wm + 16 * i + a_row, 2 * ks + a_seg)]);
#pragma unroll
        for (int j = 0; j < NJ / 2; j++)
          ldsm4(bf[ks][2 * j][0], bf[ks][2 * j][1], bf[ks][2 * j + 1][0], bf[ks][2 * j + 1][1],
                &sB[st][sw(wn + 16 * j + b_row, 2 * ks + b_seg)]);
      }
  };
  if constexpr (MODE >= 3) { cp_wait<0>(); __syncthreads(); load_frags(0); }

  int st = 0;
  for (int kt = 0; kt < KT; kt++) {
    if constexpr (MODE < 2) {
      cp_wait<STAGES - 2>();
      __syncthreads();  // стадия kt готова; буфер стадии kt-1 больше никто не читает
    }
    const int nk = kt + STAGES - 1, ns = st == 0 ? STAGES - 1 : st - 1;
#if __CUDA_ARCH__ >= 800
    if constexpr (MODE < 1)
      if (nk < KT) { load_tile<BM, NTHR>(sA[ns], gA + nk * BK, k, t); load_tile<BN, NTHR>(sB[ns], gB + nk * BK, k, t); }
    cp_commit();
#else
    constexpr int SA = BM * 4 / NTHR, SB = BN * 4 / NTHR;
    [[maybe_unused]] RegStage<SA, SB> rg;
    if constexpr (MODE < 1)
      if (nk < KT) load_regs<NTHR>(rg, gA + nk * BK, gB + nk * BK, k, t);
#endif
    {  // два шага k32: сначала все ldmatrix обоих шагов, потом все mma
      if constexpr (MODE < 3) load_frags(st);
#pragma unroll
      for (int ks = 0; ks < 2; ks++)
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int j = 0; j < NJ; j++) mma16832(acc[i][j], af[ks][i], bf[ks][j][0], bf[ks][j][1]);
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
      if (nk < KT) store_regs<NTHR>(rg, sA[ns], sB[ns], t);
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

}  // namespace kern
