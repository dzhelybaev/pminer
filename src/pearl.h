// pearl.h — хостовая часть pearlhash: сериализация, сиды шума (V3 salted), граница, ГПСЧ.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include "blake3.h"

namespace pearl {

// Конфигурация ядра pminer: тайл 8×16 целиком в регистрах одного потока warp.
// rows [0,8,..,56] → PeriodicPattern bytes 07 07 00 00 00 00
// cols [0,1,8,9,..,56,57] → 00 01 03 07 00 00
static constexpr int K = 4096, R = 128, TILE_H = 8, TILE_W = 16;
static const uint8_t ROWS_PATTERN[6] = {7, 7, 0, 0, 0, 0};
static const uint8_t COLS_PATTERN[6] = {0, 1, 3, 7, 0, 0};

struct Header {
  uint32_t version = 1;
  uint8_t prev_block[32] = {}, merkle_root[32] = {};
  uint32_t timestamp = 0, nbits = 0;
};

inline void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }

inline void header_bytes(const Header &h, uint8_t out[76]) {
  put32(out, h.version);
  for (int i = 0; i < 32; i++) out[4 + i] = h.prev_block[31 - i];
  for (int i = 0; i < 32; i++) out[36 + i] = h.merkle_root[31 - i];
  put32(out + 68, h.timestamp);
  put32(out + 72, h.nbits);
}

inline void config_bytes(uint8_t out[52]) {
  memset(out, 0, 52);
  put32(out, K);
  out[4] = R & 0xff; out[5] = R >> 8;  // rank u16, mma_type u16 = 0
  memcpy(out + 8, ROWS_PATTERN, 6);
  memcpy(out + 14, COLS_PATTERN, 6);
}

inline void job_key(const Header &h, uint8_t out[32]) {
  uint8_t buf[128];
  header_bytes(h, buf);
  config_bytes(buf + 76);
  b3::hash_small(buf, 128, nullptr, out);
}

// V3: hash' = blake3_keyed(SALT, hash | dim_le32 | 0*28)
inline void salt(const uint8_t hash[32], uint32_t dim, const char *ctx, uint8_t out[32]) {
  uint8_t key[32], msg[64] = {0};
  b3::hash_small((const uint8_t *)ctx, strlen(ctx), nullptr, key);
  memcpy(msg, hash, 32);
  put32(msg + 32, dim);
  b3::hash_small(msg, 64, key, out);
}

inline void seed_b(const uint8_t jk[32], const uint8_t hash_b[32], uint32_t n, uint8_t out[32]) {
  uint8_t hb[32], msg[64];
  salt(hash_b, n, "pearl/cert-v3/noise-seed/B", hb);
  memcpy(msg, jk, 32); memcpy(msg + 32, hb, 32);
  b3::hash_small(msg, 64, nullptr, out);
}
inline void seed_a(const uint8_t bseed[32], const uint8_t hash_a[32], uint32_t m, uint8_t out[32]) {
  uint8_t ha[32], msg[64];
  salt(hash_a, m, "pearl/cert-v3/noise-seed/A", ha);
  memcpy(msg, bseed, 32); memcpy(msg + 32, ha, 32);
  b3::hash_small(msg, 64, nullptr, out);
}

// граница: target(nbits) * TILE_H*TILE_W*K, насыщение до 2^256-1; 8 слов LE
inline void bound(uint32_t nbits, uint32_t out[8]) {
  uint32_t e = nbits >> 24, mant = nbits & 0xffffff;
  memset(out, 0, 32);
  if (!mant || !e || (mant & 0x800000)) return;
  uint64_t factor = (uint64_t)TILE_H * TILE_W * K;
  // t = mant * 2^(8(e-3)); итог = mant*factor * 2^(8(e-3))
  int sh = 8 * ((int)e - 3);
  if (sh < 0) { mant >>= -sh; sh = 0; }  // как в Rust: сначала target, потом умножение
  unsigned __int128 v = (unsigned __int128)mant * factor;  // < 2^43
  // проверка переполнения 256 бит
  int bits = 0; for (unsigned __int128 t = v; t; t >>= 1) bits++;
  if (bits + sh > 256) { for (int i = 0; i < 8; i++) out[i] = 0xffffffffu; return; }
  for (int b = 0; b < 128; b++)
    if ((v >> b) & 1) { int p = b + sh; out[p / 32] |= 1u << (p % 32); }
}

// детерминированный ГПСЧ для матриц: значения [-64, 64]
__host__ __device__ inline uint64_t splitmix(uint64_t z) {
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
__host__ __device__ inline int8_t rnd_val(uint64_t seed, uint64_t i) { return (int8_t)((int)(splitmix(seed + i) % 129) - 64); }

// дешёвый генератор базовых матриц (host и GPU одинаково): murmur3 fmix32 + сужение до [-64,64]
__host__ __device__ __forceinline__ int8_t gen_val(uint32_t seed, uint32_t idx) {
  uint32_t x = idx * 0x9E3779B1u + seed;
  x ^= x >> 16; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
  return (int8_t)((int)(((x >> 8) * 129u) >> 24) - 64);
}
inline void fill(std::vector<int8_t> &v, uint32_t seed) {
  for (size_t i = 0; i < v.size(); i++) v[i] = gen_val(seed, (uint32_t)i);
}
// чанк 0 матрицы A для номера попытки nonce
inline void chunk0(uint64_t seed, uint64_t nonce, int8_t out[1024]) {
  for (int i = 0; i < 1024; i++) out[i] = rnd_val(seed, nonce * 1024 + i);
}

}  // namespace pearl
