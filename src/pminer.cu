// pminer v0.2 — майнинг pearlhash (dense, сертификат V3) на NVIDIA RTX, без пула.
// Каждая GPU: самотест против эталона → цикл попыток на тестовом заголовке.
// Попытка = новый чанк 0 матрицы A → пересчёт корня A (дерево blake3), сиды,
// шум A на GPU, NoisyGEMM m×n×k с jackpot и проверкой границы в эпилоге.
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "kernels.cuh"
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

// Буферы и состояние задачи на одной GPU
struct Miner {
  Gpu *g = nullptr;
  int m = 0, n = 0;
  const int k = pearl::K, r = pearl::R;
  int8_t *dA = nullptr, *dAp = nullptr, *dB = nullptr, *dBp = nullptr, *dE = nullptr;
  uint8_t *dF = nullptr, *dS = nullptr;
  unsigned *dCount = nullptr;  // счётчик удач на слот пакета
  kern::Hit *dHits = nullptr;
  uint32_t *dDigest = nullptr;
  static const int MAX_HITS = 64, BATCH = 8;
  int8_t *hC0 = nullptr;      // pinned: чанки 0 для слотов пакета
  unsigned *hCount = nullptr; // pinned: счётчики удач
  b3::Pow2Tree treeA;
  uint8_t jk[32], bseed[32];
  kern::Bound bnd;
  uint64_t seedA = SEED_A, seedB = SEED_B, seedN = SEED_N;

  bool alloc(int m_, int n_) {
    m = m_; n = n_;
    size_t big = (size_t)(m > n ? m : n);
    CK(cudaMalloc(&dA, (size_t)m * k));
    CK(cudaMalloc(&dAp, (size_t)m * k));
    CK(cudaMalloc(&dB, (size_t)n * k));
    CK(cudaMalloc(&dBp, (size_t)n * k));
    CK(cudaMalloc(&dE, big * r));
    CK(cudaMalloc(&dF, k));
    CK(cudaMalloc(&dS, k));
    CK(cudaMalloc(&dCount, BATCH * sizeof(unsigned)));
    CK(cudaMalloc(&dHits, BATCH * MAX_HITS * sizeof(kern::Hit)));
    CK(cudaMallocHost(&hC0, BATCH * 1024));
    CK(cudaMallocHost(&hCount, BATCH * sizeof(unsigned)));
    CK(cudaMalloc(&dDigest, 32));
    return true;
  }
  void release() {
    void *p[] = {dA, dAp, dB, dBp, dE, dF, dS, dCount, dHits, dDigest};
    for (void *q : p) if (q) cudaFree(q);
    if (hC0) cudaFreeHost(hC0);
    if (hCount) cudaFreeHost(hCount);
  }

  bool noise(int8_t *dst, const int8_t *src, int rows, const uint8_t seed[32], bool isA) {
    kern::Seed s = to_seed(seed);
    int nh = rows * r / 32;
    kern::k_uniform<<<(nh + 255) / 256, 256>>>(dE, rows, r, s, isA);
    kern::k_perm<<<((k + 7) / 8 + 127) / 128, 128>>>(dF, dS, k, r, s, isA);
    kern::k_apply<<<rows, 256, r>>>(dst, src, dE, dF, dS, k, r);
    CK(cudaGetLastError());
    return true;
  }

  // новая задача: заголовок, матрицы A и B^T, дерево A, шум B
  bool job(const pearl::Header &h) {
    pearl::job_key(h, jk);
    uint32_t bw[8];
    pearl::bound(h.nbits, bw);
    memcpy(bnd.w, bw, 32);
    std::vector<int8_t> A((size_t)m * k), B((size_t)n * k);
    pearl::fill(A, seedA);
    pearl::fill(B, seedB);
    treeA.build((const uint8_t *)A.data(), A.size(), jk);
    b3::Pow2Tree treeB;
    treeB.build((const uint8_t *)B.data(), B.size(), jk);
    uint8_t hb[32];
    treeB.root_with_chunk0((const uint8_t *)B.data(), hb);
    pearl::seed_b(jk, hb, n, bseed);
    CK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice));
    if (!noise(dBp, dB, n, bseed, false)) return false;
    CK(cudaDeviceSynchronize());
    return true;
  }

  // постановка попытки в поток (без синхронизации); slot — место в пакете
  bool enqueue(uint64_t nonce, int slot, bool with_digest) {
    int8_t *c0 = hC0 + slot * 1024;
    pearl::chunk0(seedN, nonce, c0);
    uint8_t ha[32], aseed[32];
    treeA.root_with_chunk0((const uint8_t *)c0, ha);
    pearl::seed_a(bseed, ha, m, aseed);
    CK(cudaMemcpyAsync(dA, c0, 1024, cudaMemcpyHostToDevice));
    if (!noise(dAp, dA, m, aseed, true)) return false;
    CK(cudaMemsetAsync(dCount + slot, 0, sizeof(unsigned)));
    if (with_digest) CK(cudaMemsetAsync(dDigest, 0, 32));
    int blocks = (m / kern::BM) * (n / kern::BN);
    kern::k_gemm<<<blocks, kern::NTHR>>>(dAp, dBp, m, n, k, r, to_seed(aseed), bnd, dCount + slot,
                                         dHits + slot * MAX_HITS, MAX_HITS, with_digest ? dDigest : nullptr);
    CK(cudaGetLastError());
    return true;
  }

  // пакет из cnt попыток; возвращает сумму удач
  bool batch(uint64_t nonce0, uint64_t step, int cnt, unsigned &hits) {
    for (int s = 0; s < cnt; s++)
      if (!enqueue(nonce0 + s * step, s, false)) return false;
    CK(cudaMemcpyAsync(hCount, dCount, cnt * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CK(cudaDeviceSynchronize());
    hits = 0;
    for (int s = 0; s < cnt; s++) hits += hCount[s];
    return true;
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

static bool selftest(Gpu *g) {
  Miner mn;
  mn.g = g;
  bool ok = mn.alloc(128, 128) && mn.job(test_header(0x1e100000, 0));
  unsigned hits = 0;
  uint32_t dig[8] = {0};
  ok = ok && mn.enqueue(0, 0, true) && ck(cudaMemcpy(&hits, mn.dCount, 4, cudaMemcpyDeviceToHost), g, "hits") &&
       ck(cudaMemcpy(dig, mn.dDigest, 32, cudaMemcpyDeviceToHost), g, "digest");
  mn.release();
  if (!ok) return false;
  bool pass = hits == SELFTEST_HITS && !memcmp(dig, SELFTEST_DIGEST, 32);
  LOG("GPU%d selftest %s (hits %u/%u, digest %08x.. ожид. %08x..)\n", g->id, pass ? "OK" : "FAIL", hits,
      SELFTEST_HITS, dig[0], SELFTEST_DIGEST[0]);
  return pass;
}

struct Opts { int m = 0, n = 0;  // 0 — авто по размеру L2
  uint32_t nbits = 0x1c020000; bool selftest_only = false; };

static void worker(Gpu *g, Opts o, int ngpu) {
  if (!ck(cudaSetDevice(g->id), g, "cudaSetDevice")) return;
  if (!selftest(g)) { g->state = -1; return; }
  if (o.selftest_only) { g->state = 2; return; }
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, g->id);
  // авто-размер: если A'+B' помещаются в L2 (≥ 20 МБ) — 2048×2048 (16 МБ, почти без DRAM),
  // иначе большие матрицы с групповым обходом блоков
  if (!o.m) o.m = p.l2CacheSize >= (20 << 20) ? 2048 : 8192;
  if (!o.n) o.n = p.l2CacheSize >= (20 << 20) ? 2048 : 16384;
  Miner mn;
  mn.g = g;
  mn.seedA = SEED_A + 0x1000000ull * g->id;
  mn.seedB = SEED_B + 0x1000000ull * g->id;
  auto t_job = std::chrono::steady_clock::now();
  if (!mn.alloc(o.m, o.n) || !mn.job(test_header(o.nbits, 0x1234 + g->id))) { g->state = -1; return; }
  LOG("GPU%d задача готова за %.1f с (m=%d n=%d k=%d, L2 %d МБ)\n", g->id,
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_job).count(), o.m, o.n, pearl::K,
      p.l2CacheSize >> 20);
  g->state = 1;
  const double mac = (double)o.m * o.n * pearl::K;
  uint64_t nonce = 1 + g->id;
  auto t0 = std::chrono::steady_clock::now();
  int it = 0;
  while (true) {
    unsigned h = 0;
    if (!mn.batch(nonce, ngpu, Miner::BATCH, h)) break;
    nonce += (uint64_t)ngpu * Miner::BATCH;
    g->hits += h;
    it += Miner::BATCH;
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
  Opts o;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--stats") && i + 1 < argc) stats = argv[++i];
    else if (!strcmp(argv[i], "--m") && i + 1 < argc) o.m = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--n") && i + 1 < argc) o.n = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--nbits") && i + 1 < argc) o.nbits = strtoul(argv[++i], nullptr, 16);
    else if (!strcmp(argv[i], "--selftest")) o.selftest_only = true;
  }
  if (o.m % 128 || o.n % 128 || (o.m & (o.m - 1)) || (o.n & (o.n - 1)) || o.m < 0 || o.n < 0) {
    fprintf(stderr, "pminer: --m/--n должны быть степенями двойки ≥ 128\n");
    return 1;
  }
  printf("pminer %s: pearlhash dense V3, k=%d r=%d тайл %dx%d, nbits=%08x (тест, без пула)\n",
         PMINER_VERSION, pearl::K, pearl::R, pearl::TILE_H, pearl::TILE_W, o.nbits);
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
    printf("GPU%d %s sm_%d%d SM=%d L2=%d МБ bus %s\n", i, p.name, p.major, p.minor, p.multiProcessorCount,
           p.l2CacheSize >> 20, gpus[i].busid);
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
    std::string hs = "[", bus = "[", busid = "[";
    for (int i = 0; i < n; i++) {
      double h = gpus[i].state == 1 ? gpus[i].hs.load() : 0;
      total += h;
      acc += gpus[i].hits;
      char b[64];
      snprintf(b, sizeof b, "%s%.0f", i ? "," : "", h); hs += b;
      snprintf(b, sizeof b, "%s%d", i ? "," : "", gpus[i].bus); bus += b;
      snprintf(b, sizeof b, "%s\"%s\"", i ? "," : "", gpus[i].busid); busid += b;
      const char *st = gpus[i].state == 1 ? "" : gpus[i].state < 0 ? " [ОШИБКА]" : " [старт]";
      LOG("GPU%d %s: %.2f TH/s, удач %u, попытка %.2f мс%s\n", i, gpus[i].name.c_str(), h / 1e12,
          gpus[i].hits.load(), gpus[i].ms.load(), st);
    }
    LOG("Total: %.2f TH/s, удач %u\n", total / 1e12, acc);
    FILE *f = fopen(tmp.c_str(), "w");
    if (f) {
      fprintf(f, "{\"ver\":\"%s\",\"uptime\":%ld,\"hs\":%s],\"bus\":%s],\"busid\":%s],\"acc\":%u}\n",
              PMINER_VERSION, up, hs.c_str(), bus.c_str(), busid.c_str(), acc);
      fclose(f);
      rename(tmp.c_str(), stats);
    }
  }
}
