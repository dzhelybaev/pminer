// pool.h — C ABI библиотеки pool/ (Rust): stratum-клиент и PlainProof официальным кодом Pearl
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
struct PmJob {
  uint64_t serial;         // растёт с каждым новым заданием
  uint8_t header[76];      // IncompleteBlockHeader как прислал пул
  uint8_t target_le[32];   // цель шара (U256, little-endian)
  uint64_t height;
  uint32_t cert_version;   // 3 — V3 salted; 1–2 — legacy сиды
  char job_id[64];
};
int pm_pool_start(const char *url, const char *user, const char *pass, const char *agent);
uint64_t pm_pool_job(PmJob *out);
void pm_pool_stats(uint64_t *acc, uint64_t *rej, uint64_t *stale, int *connected);
int pm_pool_submit(const PmJob *job, const int8_t *a, size_t m, const int8_t *bt, size_t n, size_t k, size_t rank,
                   const uint32_t *rows, size_t nrows, const uint32_t *cols, size_t ncols, int gpu);
int pm_verify_tile(const uint8_t *header76, const int8_t *a, size_t m, const int8_t *bt, size_t n, size_t k,
                   size_t rank, const uint32_t *rows, size_t nrows, const uint32_t *cols, size_t ncols,
                   uint32_t cert_version, uint32_t nbits);
}
