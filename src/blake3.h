// blake3.h — BLAKE3 для pminer: компрессия (host+device), хэш до 1 чанка и
// дерево для степени двойки чанков (host). Совместим со стандартным BLAKE3.
#pragma once
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __forceinline__ inline
#endif

namespace b3 {

static constexpr uint32_t CHUNK_START = 1, CHUNK_END = 2, PARENT = 4, ROOT = 8, KEYED = 16;
static constexpr size_t CHUNK = 1024;

__host__ __device__ __forceinline__ uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

#define B3_G(a, b, c, d, x, y)                                   \
  s[a] = s[a] + s[b] + (x); s[d] = rotr(s[d] ^ s[a], 16);        \
  s[c] = s[c] + s[d];       s[b] = rotr(s[b] ^ s[c], 12);        \
  s[a] = s[a] + s[b] + (y); s[d] = rotr(s[d] ^ s[a], 8);         \
  s[c] = s[c] + s[d];       s[b] = rotr(s[b] ^ s[c], 7);

// out[8] = первые 8 слов выхода компрессии (chaining value)
__host__ __device__ __forceinline__ void compress(const uint32_t cv[8], const uint32_t m_in[16],
                                                  uint64_t counter, uint32_t len, uint32_t flags,
                                                  uint32_t out[8]) {
  uint32_t s[16] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
                    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                    (uint32_t)counter, (uint32_t)(counter >> 32), len, flags};
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = m_in[i];
#pragma unroll
  for (int r = 0; r < 7; r++) {
    B3_G(0, 4, 8, 12, m[0], m[1]);
    B3_G(1, 5, 9, 13, m[2], m[3]);
    B3_G(2, 6, 10, 14, m[4], m[5]);
    B3_G(3, 7, 11, 15, m[6], m[7]);
    B3_G(0, 5, 10, 15, m[8], m[9]);
    B3_G(1, 6, 11, 12, m[10], m[11]);
    B3_G(2, 7, 8, 13, m[12], m[13]);
    B3_G(3, 4, 9, 14, m[14], m[15]);
    if (r < 6) {  // перестановка сообщения 2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8
      uint32_t t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13],
                        m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
#pragma unroll
      for (int i = 0; i < 16; i++) m[i] = t[i];
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = s[i] ^ s[i + 8];
}
#undef B3_G

static const uint32_t IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                               0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

// ---------------- host ----------------
inline void load_words(const uint8_t *p, uint32_t *w, int n) {
  for (int i = 0; i < n; i++) w[i] = p[4 * i] | p[4 * i + 1] << 8 | p[4 * i + 2] << 16 | (uint32_t)p[4 * i + 3] << 24;
}
inline void store_words(const uint32_t *w, uint8_t *p, int n) {
  for (int i = 0; i < n; i++) for (int b = 0; b < 4; b++) p[4 * i + b] = (uint8_t)(w[i] >> (8 * b));
}

// CV одного чанка (до 1024 байт). root=true — это корень (вход ≤ 1 чанк).
inline void chunk_cv(const uint32_t key[8], uint32_t kflag, const uint8_t *data, size_t len,
                     uint64_t counter, bool root, uint32_t out[8]) {
  uint32_t cv[8];
  memcpy(cv, key, 32);
  size_t nblocks = len == 0 ? 1 : (len + 63) / 64;
  for (size_t b = 0; b < nblocks; b++) {
    uint8_t blk[64] = {0};
    size_t bl = len - b * 64 < 64 ? len - b * 64 : 64;
    if (len) memcpy(blk, data + b * 64, bl);
    uint32_t m[16];
    load_words(blk, m, 16);
    uint32_t f = kflag | (b == 0 ? CHUNK_START : 0) | (b == nblocks - 1 ? CHUNK_END : 0);
    if (root && b == nblocks - 1) f |= ROOT;
    compress(cv, m, counter, (uint32_t)bl, f, cv);
  }
  memcpy(out, cv, 32);
}

inline void parent_cv(const uint32_t key[8], uint32_t kflag, const uint32_t l[8], const uint32_t r[8],
                      bool root, uint32_t out[8]) {
  uint32_t m[16];
  memcpy(m, l, 32);
  memcpy(m + 8, r, 32);
  compress(key, m, 0, 64, kflag | PARENT | (root ? ROOT : 0), out);
}

// Хэш входа ≤ 1024 байт. key==nullptr — обычный хэш.
inline void hash_small(const uint8_t *data, size_t len, const uint8_t *key, uint8_t out[32]) {
  uint32_t k[8], o[8];
  if (key) load_words(key, k, 8); else memcpy(k, IV, 32);
  chunk_cv(k, key ? KEYED : 0, data, len, 0, true, o);
  store_words(o, out, 8);
}

// Дерево BLAKE3 для числа чанков — степени двойки (≥1). Хранит CV правых соседей
// левой ветви, чтобы быстро пересчитывать корень при изменении чанка 0.
struct Pow2Tree {
  uint32_t key[8], kflag = 0;
  std::vector<std::array<uint32_t, 8>> sib;  // sib[j] = CV поддерева [2^j, 2^{j+1})
  size_t nchunks = 0;

  void build(const uint8_t *data, size_t len, const uint8_t *k);
  // корень, если чанк 0 заменить на chunk0 (1024 байта)
  void root_with_chunk0(const uint8_t *chunk0, uint8_t out[32]) const {
    uint32_t cv[8];
    chunk_cv(key, kflag, chunk0, CHUNK, 0, nchunks == 1, cv);
    for (size_t j = 0; j < sib.size(); j++) parent_cv(key, kflag, cv, sib[j].data(), j + 1 == sib.size(), cv);
    store_words(cv, out, 8);
  }
};

}  // namespace b3

inline void b3::Pow2Tree::build(const uint8_t *data, size_t len, const uint8_t *k) {
  if (k) load_words(k, key, 8); else memcpy(key, IV, 32);
  kflag = k ? KEYED : 0;
  nchunks = len / CHUNK;
  sib.clear();
  std::vector<std::array<uint32_t, 8>> lvl(nchunks);
  for (size_t c = 0; c < nchunks; c++) chunk_cv(key, kflag, data + c * CHUNK, CHUNK, c, false, lvl[c].data());
  while (lvl.size() > 1) {
    sib.push_back(lvl[1]);
    std::vector<std::array<uint32_t, 8>> nx(lvl.size() / 2);
    for (size_t i = 0; i < nx.size(); i++) parent_cv(key, kflag, lvl[2 * i].data(), lvl[2 * i + 1].data(), false, nx[i].data());
    lvl.swap(nx);
  }
}
