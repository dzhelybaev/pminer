// pminer — этап 1: бенчмарк INT8 tensor cores на всех GPU (основа будущего pearlhash).
// Пишет статистику в JSON для HiveOS (h-stats.sh). Реального майнинга пока нет.
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifndef PMINER_VERSION
#define PMINER_VERSION "dev"
#endif

#define ITERS 4096
#define CHAINS 8  // независимые аккумуляторы прячут латентность mma

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s @%s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return; } } while (0)

__global__ void imma_kernel(int *out, int seed) {
  unsigned a0 = (threadIdx.x * 0x01010101u) ^ seed, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7;
  unsigned b0 = a0 ^ 0x5a5a5a5au, b1 = b0 * 9;
  int acc[CHAINS][4] = {};
#pragma unroll 1
  for (int i = 0; i < ITERS; i++) {
#pragma unroll
    for (int c = 0; c < CHAINS; c++) {
#if __CUDA_ARCH__ >= 800
      // m16n8k32 s8: 16*8*32 = 4096 MAC на warp
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                   : "+r"(acc[c][0]), "+r"(acc[c][1]), "+r"(acc[c][2]), "+r"(acc[c][3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
      // Turing sm_75: m8n8k16 s8 = 1024 MAC, 4 шт. = 4096 MAC
      asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};"
                   : "+r"(acc[c][0]), "+r"(acc[c][1]) : "r"(a0), "r"(b0));
      asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};"
                   : "+r"(acc[c][2]), "+r"(acc[c][3]) : "r"(a1), "r"(b1));
      asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};"
                   : "+r"(acc[c][0]), "+r"(acc[c][1]) : "r"(a2), "r"(b0));
      asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};"
                   : "+r"(acc[c][2]), "+r"(acc[c][3]) : "r"(a3), "r"(b1));
#endif
    }
  }
  int s = 0;
  for (int c = 0; c < CHAINS; c++) s += acc[c][0] ^ acc[c][1] ^ acc[c][2] ^ acc[c][3];
  if (s == 0x7fffffff) out[0] = s;  // не даём компилятору выкинуть работу
}

struct Gpu {
  int id = 0;
  char busid[16] = "00:00.0";
  int bus = 0;
  std::string name;
  std::atomic<double> hs{0};  // MAC/s ≈ H/s pearlhash
};

static std::atomic<bool> g_run{true};

static void worker(Gpu *g) {
  CK(cudaSetDevice(g->id));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, g->id));
  int *d;
  CK(cudaMalloc(&d, 4));
  const int blocks = p.multiProcessorCount * 8, threads = 256;
  const double mac_per_launch = (double)blocks * threads / 32 * ITERS * CHAINS * 4096.0;
  int seed = 1;
  while (g_run) {
    auto t0 = std::chrono::steady_clock::now();
    int n = 0;
    double sec = 0;
    do {
      imma_kernel<<<blocks, threads>>>(d, seed++);
      n++;
      if ((n & 15) == 0) {
        CK(cudaDeviceSynchronize());
        sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      }
    } while (sec < 2.0);
    CK(cudaGetLastError());
    g->hs = mac_per_launch * n / sec;
  }
  cudaFree(d);
}

int main(int argc, char **argv) {
  const char *stats = "/run/pminer/stats.json";
  for (int i = 1; i < argc; i++)
    if (!strcmp(argv[i], "--stats") && i + 1 < argc) stats = argv[++i];

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
    printf("GPU%d %s sm_%d%d SM=%d bus %s\n", i, p.name, p.major, p.minor, p.multiProcessorCount,
           gpus[i].busid);
  }
  std::vector<std::thread> th;
  for (auto &g : gpus) th.emplace_back(worker, &g);

  auto start = std::chrono::steady_clock::now();
  std::string tmp = std::string(stats) + ".tmp";
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    long up = (long)std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double total = 0;
    std::string hs = "[", bus = "[", busid = "[";
    for (int i = 0; i < n; i++) {
      double h = gpus[i].hs;
      total += h;
      char b[64];
      snprintf(b, sizeof b, "%s%.0f", i ? "," : "", h); hs += b;
      snprintf(b, sizeof b, "%s%d", i ? "," : "", gpus[i].bus); bus += b;
      snprintf(b, sizeof b, "%s\"%s\"", i ? "," : "", gpus[i].busid); busid += b;
      printf("GPU%d %s: %.2f TH/s\n", i, gpus[i].name.c_str(), h / 1e12);
    }
    printf("Total: %.2f TH/s\n", total / 1e12);
    fflush(stdout);
    FILE *f = fopen(tmp.c_str(), "w");
    if (f) {
      fprintf(f, "{\"ver\":\"%s\",\"uptime\":%ld,\"hs\":%s],\"bus\":%s],\"busid\":%s]}\n",
              PMINER_VERSION, up, hs.c_str(), bus.c_str(), busid.c_str());
      fclose(f);
      rename(tmp.c_str(), stats);
    }
  }
}
