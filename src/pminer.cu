// pminer v0.2 — майнинг pearlhash (dense, сертификат V3) на NVIDIA RTX, без пула.
// Каждая GPU: самотест против эталона → цикл попыток на тестовом заголовке.
// Попытка = новый чанк 0 матрицы A → пересчёт корня A (дерево blake3), сиды,
// шум A на GPU, NoisyGEMM m×n×k с jackpot и проверкой границы в эпилоге.
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include "kernels.cuh"
#include "pool.h"
#include "nvml.h"
#include "pearl.h"
#include "selftest.h"

#ifndef PMINER_VERSION
#define PMINER_VERSION "dev"
#endif

static std::mutex g_log_mu;
#define LOG(...) do { std::lock_guard<std::mutex> _l(g_log_mu); printf(__VA_ARGS__); fflush(stdout); } while (0)

static const uint64_t SEED_A = 0xA11CE, SEED_B = 0xB0B, SEED_N = 0x5EED;

struct Gpu {
  int id = 0, bus = 0;
  char busid[16] = "00:00.0";
  std::string name;
  std::atomic<double> hs{0};
  std::atomic<unsigned> hits{0};
  std::atomic<int> state{0};
  std::atomic<double> ms{0};  // среднее время попытки  // 0 старт, 1 работает, -1 ошибка
  nvml::Power pw;  // v0.23: мощность из NVML (только поток статистики)
};

static bool ck(cudaError_t e, Gpu *g, const char *what) {
  if (e == cudaSuccess) return true;
  LOG("GPU%d CUDA error %s: %s\n", g->id, what, cudaGetErrorString(e));
  g->state = -1;
  return false;
}
#define CK(x) do { if (!ck((x), g, #x)) return false; } while (0)

static kern::Seed to_seed(const uint8_t b[32]) {
  kern::Seed s;
  b3::load_words(b, s.w, 8);
  return s;
}

// Буферы и состояние задачи на одной GPU. Две «полосы» (поток + свой A' + шум):
// попытки чередуются по полосам, GEMM одной перекрывается с шумом и хвостом другой.
struct Lane {
  cudaStream_t st = nullptr;
  int8_t *dAp = nullptr;
  uint8_t *dF = nullptr, *dS = nullptr;
};

struct Miner {
  Gpu *g = nullptr;
  int m = 0, n = 0, smem = 0, bm = 128, wn = 64, tw = 16, lanes = LANES;  // tw — ширина тайла = wn/4
  bool pin = false;  // v0.18: B' закреплена в L2 (persisting access policy), если GPU умеет
  size_t pinned = 0;  // сколько байт L2 отдано под закрепление
  int gm = 8;  // v0.20: строк блоков в группе обхода (общие панели A/B в L2)
  bool pipe = true;  // v0.15: конвейер фрагментов через границу стадий (только блок 128, тайл 8×16)
  const int k = pearl::K, r = pearl::R;
  static const int LANES = 2, MAX_HITS = 64, BATCH = 64;  // BATCH — максимум слотов пакета
  Lane ln[LANES];
  int8_t *dB = nullptr, *dBp = nullptr, *dE = nullptr;
  unsigned *dCount = nullptr;  // счётчик удач на слот пакета
  kern::Hit *dHits = nullptr;
  uint32_t *dDigest = nullptr;
  unsigned *hCount = nullptr;  // pinned
  cudaEvent_t done = nullptr;  // конец пакета; BlockingSync — поток CPU спит, а не крутится
  b3::Pow2Tree treeA;
  std::vector<int8_t> hA, hB;  // базовые A (m×k) и B^T (n×k) на хосте — для доказательств
  bool salted = true;          // сиды V3 (cert_version ≥ 3)
  bool dense = false;          // false: A и B почти нулевые — доказательство сжимается gzip в ~20+ раз
  uint8_t jk[32], bseed[32];
  kern::Bound bnd;
  uint64_t seedA = SEED_A, seedB = SEED_B, seedN = SEED_N;

  bool alloc(int m_, int n_, int lanes_, int bm_, int wn_ = 64, bool pipe_ = true, bool pin_ = false, int gm_ = 0) {
    gm = gm_ > 0 ? std::min(gm_, m_ / bm_) : std::max(1, std::min(1024 / bm_, m_ / bm_));
    m = m_; n = n_; lanes = lanes_; bm = bm_; wn = wn_; tw = wn / 4; pipe = pipe_ && bm == 128 && wn == 64;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, g->id));
    smem = kern::gemm_smem(bm, p.major);
    const auto attr = cudaFuncAttributeMaxDynamicSharedMemorySize;
    if (bm == 256) CK(cudaFuncSetAttribute(kern::k_gemm<256, 64, 0, false>, attr, smem));
    else if (wn == 32) CK(cudaFuncSetAttribute(kern::k_gemm<128, 32, 0, false>, attr, smem));
    else if (pipe) CK(cudaFuncSetAttribute(kern::k_gemm<128, 64, 0, true>, attr, smem));
    else CK(cudaFuncSetAttribute(kern::k_gemm<128, 64, 0, false>, attr, smem));
    for (int i = 0; i < lanes; i++) {
      CK(cudaStreamCreateWithFlags(&ln[i].st, cudaStreamNonBlocking));
      CK(cudaMalloc(&ln[i].dAp, (size_t)m * k));
      CK(cudaMalloc(&ln[i].dF, k));
      CK(cudaMalloc(&ln[i].dS, k));
    }
    CK(cudaMalloc(&dB, (size_t)n * k));
    CK(cudaMalloc(&dBp, (size_t)n * k));
    pin = false;
    pinned = 0;
    if (pin_ && p.persistingL2CacheMaxSize > 0 && p.accessPolicyMaxWindowSize > 0) {
      // B' неизменна всю задачу: закрепляем её в L2, чтобы поток A' её не вытеснял
      size_t bytes = std::min((size_t)n * k, (size_t)p.accessPolicyMaxWindowSize);
      pinned = std::min(bytes, (size_t)p.persistingL2CacheMaxSize);
      CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, pinned));
      cudaStreamAttrValue a = {};
      a.accessPolicyWindow.base_ptr = dBp;
      a.accessPolicyWindow.num_bytes = bytes;
      a.accessPolicyWindow.hitRatio = std::min(1.0f, (float)pinned / (float)bytes);
      a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
      a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      for (int i = 0; i < lanes; i++) CK(cudaStreamSetAttribute(ln[i].st, cudaStreamAttributeAccessPolicyWindow, &a));
      pin = true;
    }
    CK(cudaMalloc(&dE, (size_t)n * r));
    CK(cudaMalloc(&dCount, BATCH * sizeof(unsigned)));
    CK(cudaMalloc(&dHits, BATCH * MAX_HITS * sizeof(kern::Hit)));
    CK(cudaMallocHost(&hCount, BATCH * sizeof(unsigned)));
    CK(cudaMalloc(&dDigest, 32));
    CK(cudaEventCreateWithFlags(&done, cudaEventBlockingSync | cudaEventDisableTiming));
    return true;
  }
  void release() {
    if (pin) {  // вернуть L2 в общее пользование
      cudaCtxResetPersistingL2Cache();
      cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
      pin = false;
    }
    for (auto &l : ln) {
      if (l.dAp) cudaFree(l.dAp);
      if (l.dF) cudaFree(l.dF);
      if (l.dS) cudaFree(l.dS);
      if (l.st) cudaStreamDestroy(l.st);
      l = Lane();
    }
    hA.clear(); hB.clear();
    void *p[] = {dB, dBp, dE, dCount, dHits, dDigest};
    for (void *q : p) if (q) cudaFree(q);
    if (hCount) cudaFreeHost(hCount);
    if (done) cudaEventDestroy(done);
    done = nullptr;
    dB = dBp = dE = nullptr; dCount = hCount = nullptr; dHits = nullptr; dDigest = nullptr;
  }

  // тестовая задача из заголовка-структуры и nbits
  bool job(const pearl::Header &h) {
    uint8_t hb[76];
    pearl::header_bytes(h, hb);
    uint32_t bw[8];
    pearl::bound(h.nbits, bw, tw);
    return job_raw(hb, bw, true);
  }

  // задача: заголовок (76 Б), граница, вид сидов. Матрицы A и B^T генерируются один раз
  // на размер; на каждую задачу пересчитываются дерево A, хэш B (ключ — job_key) и шум B
  bool job_raw(const uint8_t hdr[76], const uint32_t bound[8], bool salted_) {
    salted = salted_;
    pearl::job_key_raw(hdr, jk, tw);
    memcpy(bnd.w, bound, 32);
    bool fresh = hA.size() != (size_t)m * k || hB.size() != (size_t)n * k;
    if (fresh) {
      hA.assign((size_t)m * k, 0);
      hB.assign((size_t)n * k, 0);
      if (dense) {  // иначе нули (assign выше)
        pearl::fill(hA, (uint32_t)seedA);
        pearl::fill(hB, (uint32_t)seedB);
      }
    }
    treeA.build((const uint8_t *)hA.data(), hA.size(), jk);
    uint8_t hb[32];  // n — любое кратное 128, поэтому общий хэш, а не дерево степени двойки
    b3::hash_large((const uint8_t *)hB.data(), hB.size(), jk, hb);
    pearl::seed_b(jk, hb, n, bseed, salted);
    if (fresh) CK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    kern::Seed s = to_seed(bseed);
    kern::k_uniform<<<(n * r / 32 + 255) / 256, 256>>>(dE, n, r, s, false);
    kern::k_perm<<<((k + 7) / 8 + 127) / 128, 128>>>(ln[0].dF, ln[0].dS, k, r, s, false);
    kern::k_apply<<<n, 256, r>>>(dBp, dB, dE, ln[0].dF, ln[0].dS, k, r);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    return true;
  }

  // шум A на полосе: перестановка (+ сброс счётчика) и слитое ядро A'
  void noise_a(Lane &l, const kern::Seed &s, uint64_t nonce, unsigned *reset) {
    kern::k_perm<<<((k + 7) / 8 + 127) / 128, 128, 0, l.st>>>(l.dF, l.dS, k, r, s, true, reset);
    kern::k_apply_fused<<<m, 256, r, l.st>>>(l.dAp, (uint32_t)seedA, seedN, nonce, s, l.dF, l.dS, k, r, dense);
  }
  void gemm(Lane &l, const kern::Seed &s, unsigned *cnt, kern::Hit *hits, int maxh, uint32_t *dig) {
    int blocks = (m / bm) * (n / kern::BN);
    int thr = kern::gemm_threads_rt(bm, wn);
#define PM_GEMM(...) kern::k_gemm<__VA_ARGS__><<<blocks, thr, smem, l.st>>>(l.dAp, dBp, m, n, gm, s, bnd, cnt, hits, maxh, dig)
    if (bm == 256) PM_GEMM(256, 64, 0, false);
    else if (wn == 32) PM_GEMM(128, 32, 0, false);
    else if (pipe) PM_GEMM(128, 64, 0, true);
    else PM_GEMM(128, 64, 0, false);
#undef PM_GEMM
  }

  // диагностика: время одного GEMM (мс) в режиме MODE на полосе 0, только блок 128
  template <int WN, int MODE, bool PIPE> float diag_gemm(const kern::Seed &s, int reps) {
    cudaFuncSetAttribute(kern::k_gemm<128, WN, MODE, PIPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    const int thr = kern::gemm_threads_rt(128, WN);
    Lane &l = ln[0];
    int blocks = (m / 128) * (n / kern::BN);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    kern::k_gemm<128, WN, MODE, PIPE><<<blocks, thr, smem, l.st>>>(l.dAp, dBp, m, n, gm, s, bnd, dCount, dHits, 0, nullptr);
    cudaEventRecord(e0, l.st);
    for (int i = 0; i < reps; i++)
      kern::k_gemm<128, WN, MODE, PIPE><<<blocks, thr, smem, l.st>>>(l.dAp, dBp, m, n, gm, s, bnd, dCount, dHits, 0, nullptr);
    cudaEventRecord(e1, l.st);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / reps;
  }

  // постановка попытки в поток полосы (без синхронизации); slot — место в пакете
  bool enqueue(uint64_t nonce, int slot, bool with_digest) {
    int8_t c0[1024];
    pearl::chunk0(seedN, nonce, c0, dense);  // на GPU тот же чанк считается в k_apply_fused
    uint8_t ha[32], aseed[32];
    treeA.root_with_chunk0((const uint8_t *)c0, ha);
    pearl::seed_a(bseed, ha, m, aseed, salted);
    Lane &l = ln[slot % lanes];
    kern::Seed s = to_seed(aseed);
    if (with_digest) CK(cudaMemsetAsync(dDigest, 0, 32, l.st));
    noise_a(l, s, nonce, dCount + slot);
    gemm(l, s, dCount + slot, dHits + slot * MAX_HITS, MAX_HITS, with_digest ? dDigest : nullptr);
    CK(cudaGetLastError());
    return true;
  }

  // пакет из cnt попыток; возвращает сумму удач. on_hit(nonce, row0, col0) — для каждой удачи
  template <class F>
  bool batch(uint64_t nonce0, uint64_t step, int cnt, unsigned &hits, F &&on_hit) {
    for (int s = 0; s < cnt; s++)
      if (!enqueue(nonce0 + s * step, s, false)) return false;
    for (int i = 1; i < lanes; i++) {  // событие в полосе 0 ждёт остальные полосы
      CK(cudaEventRecord(done, ln[i].st));
      CK(cudaStreamWaitEvent(ln[0].st, done, 0));
    }
    CK(cudaEventRecord(done, ln[0].st));
    CK(cudaEventSynchronize(done));
    CK(cudaMemcpy(hCount, dCount, cnt * sizeof(unsigned), cudaMemcpyDeviceToHost));
    hits = 0;
    for (int s = 0; s < cnt; s++) {
      hits += hCount[s];
      if (!hCount[s]) continue;
      int nh = std::min<int>(hCount[s], MAX_HITS);
      std::vector<kern::Hit> hv(nh);
      CK(cudaMemcpy(hv.data(), dHits + s * MAX_HITS, nh * sizeof(kern::Hit), cudaMemcpyDeviceToHost));
      for (auto &h : hv) on_hit(nonce0 + s * step, h.row0, h.col0);
    }
    return true;
  }
  bool batch(uint64_t nonce0, uint64_t step, int cnt, unsigned &hits) {
    return batch(nonce0, step, cnt, hits, [](uint64_t, uint32_t, uint32_t) {});
  }

  // шар в пул: полная A с чанком 0 этой попытки, строки/столбцы тайла
  void submit(const PmJob &job, uint64_t nonce, uint32_t row0, uint32_t col0) {
    std::vector<int8_t> A(hA);
    pearl::chunk0(seedN, nonce, A.data(), dense);
    uint32_t rows[pearl::TILE_H], cols[pearl::TILE_W];
    tile_indices(row0, col0, rows, cols);
    pm_pool_submit(&job, A.data(), m, hB.data(), n, k, r, rows, pearl::TILE_H, cols, tw, g->id);
  }
  // строки и столбцы тайла jackpot потока по его (row0, col0); столбцов tw
  void tile_indices(uint32_t row0, uint32_t col0, uint32_t *rows, uint32_t *cols) const {
    for (int i = 0; i < pearl::TILE_H; i++) rows[i] = row0 + 8 * i;
    for (int j = 0; j < tw / 2; j++) { cols[2 * j] = col0 + 8 * j; cols[2 * j + 1] = col0 + 8 * j + 1; }
  }
};

static pearl::Header test_header(uint32_t nbits, uint64_t salt) {
  pearl::Header h;
  for (int i = 0; i < 32; i++) {
    h.prev_block[i] = salt ? (uint8_t)pearl::splitmix(salt + i) : (uint8_t)i;
    h.merkle_root[i] = salt ? (uint8_t)pearl::splitmix(salt + 100 + i) : (uint8_t)(0x80 + i);
  }
  h.timestamp = 0x6a000000;
  h.nbits = nbits;
  return h;
}

static bool selftest(Gpu *g, int bm, int wn, bool pipe = false) {
  Miner mn;
  mn.g = g;
  mn.dense = true;  // самотест — на случайных матрицах эталона
  bool ok = mn.alloc(256, 128, 1, bm, wn, pipe) && mn.job(test_header(0x1e100000, 0));
  const int vi = wn == 32 ? 1 : 0;  // набор ожидаемых значений
  unsigned hits = 0;
  uint32_t dig[8] = {0};
  ok = ok && mn.enqueue(0, 0, true) && ck(cudaDeviceSynchronize(), g, "sync") && ck(cudaMemcpy(&hits, mn.dCount, 4, cudaMemcpyDeviceToHost), g, "hits") &&
       ck(cudaMemcpy(dig, mn.dDigest, 32, cudaMemcpyDeviceToHost), g, "digest");
  // координаты первой удачи с GPU → доказательство официальным кодом Pearl → проверка
  int proof_ok = -1;
  if (ok && hits > 0) {
    kern::Hit h0;
    ok = ck(cudaMemcpy(&h0, mn.dHits, sizeof h0, cudaMemcpyDeviceToHost), g, "hit");
    if (ok) {
      std::vector<int8_t> A(mn.hA);
      pearl::chunk0(mn.seedN, 0, A.data());
      uint32_t rows[pearl::TILE_H], cols[pearl::TILE_W];
      mn.tile_indices(h0.row0, h0.col0, rows, cols);
      pearl::Header th = test_header(0x1e100000, 0);
      uint8_t hb[76];
      pearl::header_bytes(th, hb);
      proof_ok = pm_verify_tile(hb, A.data(), mn.m, mn.hB.data(), mn.n, mn.k, mn.r, rows, pearl::TILE_H, cols,
                                mn.tw, 3, th.nbits) == 0;
    }
  }
  mn.release();
  if (!ok) return false;
  bool pass = hits == SELFTEST_HITS[vi] && !memcmp(dig, SELFTEST_DIGEST[vi], 32) && proof_ok == 1;
  LOG("GPU%d selftest блок %d тайл 8×%d%s: %s (hits %u/%u, digest %08x.. ожид. %08x.., доказательство %s)\n", g->id,
      bm, wn / 4, mn.pipe ? " конвейер" : "", pass ? "OK" : "FAIL", hits, SELFTEST_HITS[vi], dig[0], SELFTEST_DIGEST[vi][0],
      proof_ok == 1 ? "OK" : "FAIL");
  return pass;
}

// Микробенчмарк: TH/s чистых mma при 8/16/32 warp на SM, постоянные и меняющиеся данные.
// Отвечает, во что упирается ядро: в число warp (регистры) или в энергию на операцию.
static void microbench(Gpu *g) {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, g->id);
  int *d = nullptr;
  if (cudaMalloc(&d, 4) != cudaSuccess) return;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  char line[512];
  int pos = snprintf(line, sizeof line, "GPU%d микробенч mma, TH/s (постоянные / меняющиеся данные):", g->id);
  for (int bps : {2, 4, 8}) {  // блоков по 4 warp на SM → 8, 16, 32 warp на SM
    const int blocks = p.multiProcessorCount * bps;
    // ~50 мс на запуск при ~60 TH/s
    const int iters = (int)(3e12 / ((double)blocks * 4 * 8 * 4096));
    double ths[2];
    for (int rnd = 0; rnd < 2; rnd++) {
      auto launch = [&] {
        if (rnd) kern::k_bench<true><<<blocks, 128>>>(iters, d, 1);
        else kern::k_bench<false><<<blocks, 128>>>(iters, d, 1);
      };
      launch();
      cudaEventRecord(e0);
      for (int i = 0; i < 4; i++) launch();
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      ths[rnd] = (double)blocks * 4 * 8 * 4096.0 * iters * 4 / (ms * 1e-3) / 1e12;
    }
    pos += snprintf(line + pos, sizeof line - pos, " %d warp: %.1f / %.1f |", bps * 4, ths[0], ths[1]);
  }
  LOG("%s\n", line);
  {  // ILP при 8 warp на SM (как в нашем ядре), меняющиеся данные: 1, 4, 8 цепочек на warp
    const int blocks = p.multiProcessorCount * 2;
    const int iters = (int)(1.5e12 / ((double)blocks * 4 * 8 * 4096));
    pos = snprintf(line, sizeof line, "GPU%d микробенч ILP при 8 warp/SM, меняющиеся данные, TH/s:", g->id);
    for (int ch : {1, 4, 8}) {
      auto launch = [&] {
        if (ch == 1) kern::k_bench<true, 1><<<blocks, 128>>>(iters, d, 1);
        else if (ch == 4) kern::k_bench<true, 4><<<blocks, 128>>>(iters, d, 1);
        else kern::k_bench<true, 8><<<blocks, 128>>>(iters, d, 1);
      };
      launch();
      cudaEventRecord(e0);
      for (int i = 0; i < 4; i++) launch();
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      double ths = (double)blocks * 4 * 8 * 4096.0 * iters * 4 / (ms * 1e-3) / 1e12;
      pos += snprintf(line + pos, sizeof line - pos, " %d цеп.: %.1f |", ch, ths);
    }
    LOG("%s\n", line);
  }
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaFree(d);
}

struct Opts { int m = 0, n = 0, lanes = 0, bm = 0, wn = 0, pipe = 1, pin = 1, gm = 0;  // 0 — автонастройка
  uint32_t nbits = 0x1c020000; bool selftest_only = false;
  bool pool = false, dense = false; };

static void worker(Gpu *g, Opts o, int ngpu) {
  if (!ck(cudaSetDevice(g->id), g, "cudaSetDevice")) return;
  if (!selftest(g, 128, 64, true) || !selftest(g, 128, 64, false) || !selftest(g, 128, 32) || !selftest(g, 256, 64)) {
    g->state = -1;
    return;
  }
  if (o.selftest_only) { g->state = 2; return; }
  microbench(g);
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, g->id);
  const bool big_l2 = p.l2CacheSize >= (20 << 20);
  // кандидаты (bm, m, n, полосы). L2-карты: всё должно жить в L2 (A' на полосу + B')
  struct Cand { int bm, m, n, lanes, wn; bool pipe, pin; int gm = 0; };  // gm 0 — 1024/bm
  std::vector<Cand> cands;
  // («n под волну» в v0.9 не выиграл; в v0.22 вернулся — после устранения других тормозов)
  // {блок, m, n, полосы, ширина warp, конвейер}. Тайл 8×8 (v0.13) везде проигрывал — только --tile 8
  if (o.m && o.n && o.lanes && o.bm) cands.push_back({o.bm, o.m, o.n, o.lanes, o.wn ? o.wn : 64, o.pipe != 0, o.pin != 0, o.gm});
  else if (big_l2)
    cands = {{128, 512, 4096, 2, 64, true, true}, {128, 1024, 4096, 2, 64, true, true},
             {128, 512, 4096, 2, 64, false, true}, {128, 1024, 4096, 2, 64, false, true},
             {128, 512, 4096, 2, 64, true, false}, {128, 1024, 4096, 2, 64, true, false},
             {128, 512, 4096, 2, 64, false, false}, {128, 2048, 2048, 1, 64, true, true}};
  if (big_l2 && cands.size() > 1) {
    // v0.22: n «под волну» — число блоков кратно 2·SM (2 блока на SM), последняя волна полная.
    // Профиль ncu v0.21 (5060, m=512 n=4096: 128 блоков на 60 мест) — SM активны 86% времени запуска
    const int slots = 2 * p.multiProcessorCount;
    for (int mm : {512, 1024}) {
      const int rows = mm / 128, L = slots / std::gcd(slots, rows);  // столбцов блоков кратно L
      int cols = 32 / L * L;
      if (cols == 0) cols = L;
      if (cols * 128 == 4096 || cols * 128 > 8192) continue;
      for (bool pp : {true, false}) cands.push_back({128, mm, cols * 128, 2, 64, pp, true});
    }
  }
  if (!big_l2 && cands.empty())
    // v0.20: малый L2 (Turing) упирается в DRAM — перебор группы обхода gm (сколько строк блоков
    // идут рядом и делят панели B в L2)
    cands = {{128, 8192, 16384, 2, 64, false, false, 8}, {128, 8192, 16384, 2, 64, false, false, 2},
             {128, 8192, 16384, 2, 64, false, false, 4}, {128, 8192, 16384, 2, 64, false, false, 16},
             {128, 8192, 16384, 2, 64, false, false, 32}, {128, 8192, 16384, 2, 64, false, false, 64},
             {128, 4096, 8192, 2, 64, false, false, 8}};
  // блок 256 по замерам v0.6 всегда хуже — только вручную (--bm 256)
  Miner mn;
  mn.g = g;
  mn.dense = o.dense;
  mn.seedA = SEED_A + 0x1000000ull * g->id;
  mn.seedB = SEED_B + 0x1000000ull * g->id;
  const pearl::Header hdr = test_header(o.nbits, 0x1234 + g->id);
  uint64_t nonce = 1 + g->id;
  // автонастройка: ~1 с майнинга на каждом кандидате, берём лучший
  Cand best = cands[0];
  double best_hs = -1;
  int best_bs = 8;
  for (const Cand &c : cands) {
    if (!mn.alloc(c.m, c.n, c.lanes, c.bm, c.wn, c.pipe, c.pin, c.gm) || !mn.job(hdr)) { mn.release(); g->state = 0; continue; }
    unsigned h = 0;
    int bs = 8;
    auto tw = std::chrono::steady_clock::now();
    mn.batch(nonce, ngpu, bs, h);  // прогрев и оценка времени попытки
    nonce += (uint64_t)ngpu * bs;
    double ta = std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count() / bs;
    // пакет ~8 мс: пробуждение спящего потока CPU после пакета (~0.1 мс) не заметно
    bs = std::max(2, std::min(Miner::BATCH, (int)(0.008 / ta) + 1));
    int it = 0;
    auto t0 = std::chrono::steady_clock::now();
    double sec = 0;
    bool ok = true;
    while (sec < 1.5 && (ok = mn.batch(nonce, ngpu, bs, h))) {
      nonce += (uint64_t)ngpu * bs;
      it += bs;
      sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    double hs = ok ? (double)c.m * c.n * pearl::K * it / sec : 0;
    LOG("GPU%d тест: блок %d тайл 8×%d%s%s m=%d n=%d группа %d полос %d пакет %d → %.2f TH/s\n", g->id, c.bm, c.wn / 4,
        mn.pipe ? " конвейер" : "", mn.pin ? " L2-закреп" : "", c.m, c.n, mn.gm, c.lanes, bs, hs / 1e12);
    if (hs > best_hs) { best_hs = hs; best = c; best_bs = bs; }
    mn.release();
    g->state = 0;
  }
  if (best_hs <= 0 || !mn.alloc(best.m, best.n, best.lanes, best.bm, best.wn, best.pipe, best.pin, best.gm) || !mn.job(hdr)) {
    g->state = -1;
    return;
  }
  LOG("GPU%d выбрано: блок %d тайл 8×%d%s%s m=%d n=%d группа %d полос %d пакет %d (L2 %d МБ)\n", g->id, best.bm, best.wn / 4,
      mn.pipe ? " конвейер" : "", mn.pin ? " L2-закреп" : "", best.m, best.n, mn.gm, best.lanes,
      best_bs, p.l2CacheSize >> 20);
  o.m = best.m; o.n = best.n;
  if (best.bm == 128) {  // диагностика узких мест GEMM на выбранных размерах
    uint8_t sd[32] = {1};
    kern::Seed s = to_seed(sd);
    mn.noise_a(mn.ln[0], s, 1, nullptr);
    const int reps = (double)o.m * o.n > 2e7 ? 5 : 50;
    double mac1 = (double)o.m * o.n * pearl::K;
    float t[4];
    if (best.wn == 32) {
      t[0] = mn.diag_gemm<32, 0, false>(s, reps); t[1] = mn.diag_gemm<32, 1, false>(s, reps);
      t[2] = mn.diag_gemm<32, 2, false>(s, reps); t[3] = mn.diag_gemm<32, 3, false>(s, reps);
    } else if (mn.pipe) {
      t[0] = mn.diag_gemm<64, 0, true>(s, reps); t[1] = mn.diag_gemm<64, 1, true>(s, reps);
      t[2] = mn.diag_gemm<64, 2, true>(s, reps); t[3] = mn.diag_gemm<64, 3, true>(s, reps);
    } else {
      t[0] = mn.diag_gemm<64, 0, false>(s, reps); t[1] = mn.diag_gemm<64, 1, false>(s, reps);
      t[2] = mn.diag_gemm<64, 2, false>(s, reps); t[3] = mn.diag_gemm<64, 3, false>(s, reps);
    }
    LOG("GPU%d диагностика GEMM, TH/s: полный %.1f | без загрузок %.1f | +без барьеров %.1f | +без ldmatrix %.1f\n",
        g->id, mac1 / t[0] / 1e9, mac1 / t[1] / 1e9, mac1 / t[2] / 1e9, mac1 / t[3] / 1e9);
    if (!ck(cudaDeviceSynchronize(), g, "diag")) return;
  }
  const double mac = (double)o.m * o.n * pearl::K;
  PmJob job{};
  uint64_t job_serial = 0;
  if (!o.pool) g->state = 1;
  auto t0 = std::chrono::steady_clock::now();
  int it = 0;
  while (true) {
    if (o.pool) {  // новое задание пула → пересчёт задачи; без задания ждём
      PmJob nj;
      uint64_t s = pm_pool_job(&nj);
      if (!s) {
        g->state = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        continue;
      }
      if (s != job_serial) {
        uint32_t bw[8];
        pearl::bound_from_target(nj.target_le, bw, mn.tw);
        if (!mn.job_raw(nj.header, bw, nj.cert_version >= 3)) break;
        job = nj;
        job_serial = s;
        g->state = 1;
        it = 0;
        t0 = std::chrono::steady_clock::now();
      }
    }
    unsigned h = 0;
    bool ok = o.pool ? mn.batch(nonce, ngpu, best_bs, h,
                                [&](uint64_t nc, uint32_t r0, uint32_t c0) { mn.submit(job, nc, r0, c0); })
                     : mn.batch(nonce, ngpu, best_bs, h);
    if (!ok) break;
    nonce += (uint64_t)ngpu * best_bs;
    g->hits += h;
    it += best_bs;
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) {
      g->hs = mac * it / sec;
      g->ms = sec * 1000 / it;
      it = 0;
      t0 = std::chrono::steady_clock::now();
    }
  }
  g->state = -1;
  g->hs = 0;
  mn.release();
}

int main(int argc, char **argv) {
  const char *stats = "/run/pminer/stats.json";
  const char *pool_url = nullptr, *wallet = nullptr, *pass = "x";
  Opts o;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--stats") && i + 1 < argc) stats = argv[++i];
    else if (!strcmp(argv[i], "--m") && i + 1 < argc) o.m = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--n") && i + 1 < argc) o.n = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--nbits") && i + 1 < argc) o.nbits = strtoul(argv[++i], nullptr, 16);
    else if (!strcmp(argv[i], "--lanes") && i + 1 < argc) o.lanes = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--nopipe")) o.pipe = 0;
    else if (!strcmp(argv[i], "--nopin")) o.pin = 0;
    else if (!strcmp(argv[i], "--group") && i + 1 < argc) o.gm = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bm") && i + 1 < argc) o.bm = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--tile") && i + 1 < argc) o.wn = atoi(argv[++i]) * 4;  // 16 или 8
    else if (!strcmp(argv[i], "--selftest")) o.selftest_only = true;
    else if (!strcmp(argv[i], "--dense")) o.dense = true;
    else if (!strcmp(argv[i], "--pool") && i + 1 < argc) pool_url = argv[++i];
    else if (!strcmp(argv[i], "--wallet") && i + 1 < argc) wallet = argv[++i];
    else if (!strcmp(argv[i], "--pass") && i + 1 < argc) pass = argv[++i];
  }
  if (o.m % 256 || o.n % 128 || (o.bm && o.bm != 128 && o.bm != 256) || (o.wn && o.wn != 64 && o.wn != 32) ||
      (o.wn == 32 && o.bm == 256) || o.lanes < 0 || o.lanes > 2 || (o.m & (o.m - 1)) || o.m < 0 || o.n < 0) {
    fprintf(stderr, "pminer: --m (≥256) — степень двойки, --n — кратно 128; --bm 128|256; --tile 16|8 (8 — только с --bm 128); --lanes 1|2\n");
    return 1;
  }
  if (pool_url && *pool_url && !o.selftest_only) {
    if (!wallet || !*wallet) {
      fprintf(stderr, "pminer: для пула нужен --wallet КОШЕЛЁК.ВОРКЕР\n");
      return 1;
    }
    std::string agent = std::string("pminer/") + PMINER_VERSION;
    if (pm_pool_start(pool_url, wallet, pass, agent.c_str()) != 0) return 1;
    o.pool = true;
    printf("pminer %s: pearlhash, пул %s, кошелёк %s\n", PMINER_VERSION, pool_url, wallet);
  } else {
    printf("pminer %s: pearlhash dense V3, k=%d r=%d тайл %dx%d, nbits=%08x (тест, без пула)\n",
           PMINER_VERSION, pearl::K, pearl::R, pearl::TILE_H, pearl::TILE_W, o.nbits);
  }
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    fprintf(stderr, "pminer: CUDA GPU не найдены\n");
    return 1;
  }
  std::vector<Gpu> gpus(n);
  for (int i = 0; i < n; i++) {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, i);
    gpus[i].id = i;
    gpus[i].bus = p.pciBusID;
    gpus[i].name = p.name;
    snprintf(gpus[i].busid, sizeof gpus[i].busid, "%02x:%02x.0", p.pciBusID, p.pciDeviceID);
    char pci[32] = "";
    if (cudaDeviceGetPCIBusId(pci, sizeof pci, i) == cudaSuccess) gpus[i].pw.init(pci);
    printf("GPU%d %s sm_%d%d SM=%d L2=%d МБ (закрепить можно %d МБ, окно %d МБ) bus %s\n", i, p.name, p.major,
           p.minor, p.multiProcessorCount, p.l2CacheSize >> 20, (int)(p.persistingL2CacheMaxSize >> 20),
           p.accessPolicyMaxWindowSize >> 20, gpus[i].busid);
  }
  fflush(stdout);
  std::vector<std::thread> th;
  for (auto &g : gpus) th.emplace_back(worker, &g, o, n);
  if (o.selftest_only) {
    for (auto &t : th) t.join();
    bool all = true;
    for (auto &g : gpus) all &= g.state == 2;
    return all ? 0 : 2;
  }

  auto start = std::chrono::steady_clock::now();
  std::string tmp = std::string(stats) + ".tmp";
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    long up = (long)std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double total = 0;
    unsigned acc = 0;
    std::string hs = "[", bus = "[", busid = "[", pwr = "[";
    double wsum = 0;
    bool wall = true;
    for (int i = 0; i < n; i++) {
      double h = gpus[i].state == 1 ? gpus[i].hs.load() : 0;
      total += h;
      acc += gpus[i].hits;
      char b[64];
      snprintf(b, sizeof b, "%s%.0f", i ? "," : "", h); hs += b;
      snprintf(b, sizeof b, "%s%d", i ? "," : "", gpus[i].bus); bus += b;
      snprintf(b, sizeof b, "%s\"%s\"", i ? "," : "", gpus[i].busid); busid += b;
      const char *st = gpus[i].state == 1 ? "" : gpus[i].state < 0 ? " [ОШИБКА]" : " [старт]";
      const double w = gpus[i].pw.watts();
      char ws[64] = "";
      if (w > 0) {
        snprintf(ws, sizeof ws, ", %.0f Вт (%.2f TH/Вт)", w, h / 1e12 / w);
        wsum += w;
      } else wall = false;
      snprintf(b, sizeof b, "%s%.0f", i ? "," : "", w > 0 ? w : 0); pwr += b;
      LOG("GPU%d %s: %.2f TH/s%s, %s %u, попытка %.2f мс%s\n", i, gpus[i].name.c_str(), h / 1e12, ws,
          o.pool ? "шаров" : "удач", gpus[i].hits.load(), gpus[i].ms.load(), st);
    }
    uint64_t pacc = 0, prej = 0, pst = 0;
    int pcon = 0;
    if (wall && wsum > 0) LOG("Мощность карт: %.0f Вт, %.3f TH/Вт\n", wsum, total / 1e12 / wsum);
    if (o.pool) {  // в режиме пула «принято/отклонено» — ответы пула
      pm_pool_stats(&pacc, &prej, &pst, &pcon);
      LOG("Total: %.2f TH/s, найдено шаров %u, принято %llu, отклонено %llu, устарело %llu%s\n", total / 1e12, acc,
          (unsigned long long)pacc, (unsigned long long)prej, (unsigned long long)pst, pcon ? "" : " [нет связи с пулом]");
    } else {
      LOG("Total: %.2f TH/s, удач %u\n", total / 1e12, acc);
      pacc = acc;
    }
    FILE *f = fopen(tmp.c_str(), "w");
    if (f) {
      fprintf(f, "{\"ver\":\"%s\",\"uptime\":%ld,\"hs\":%s],\"bus\":%s],\"busid\":%s],\"power\":%s],\"acc\":%llu,\"rej\":%llu}\n",
              PMINER_VERSION, up, hs.c_str(), bus.c_str(), busid.c_str(), pwr.c_str(), (unsigned long long)pacc,
              (unsigned long long)(prej + pst));
      fclose(f);
      rename(tmp.c_str(), stats);
    }
  }
}
