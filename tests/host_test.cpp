// Хостовый тест blake3/pearl.h: печатает значения для сверки с Python (tests/host_test.py)
#include <cstdio>
#include "../src/pearl.h"
static void hex(const char *n, const uint8_t *p, int len) { printf("%s ", n); for (int i = 0; i < len; i++) printf("%02x", p[i]); printf("\n"); }
int main() {
  uint8_t out[32], key[32];
  for (int i = 0; i < 32; i++) key[i] = (uint8_t)(i * 7 + 1);
  std::vector<uint8_t> d(1 << 16);
  for (size_t i = 0; i < d.size(); i++) d[i] = (uint8_t)pearl::splitmix(i);
  for (int len : {0, 1, 63, 64, 65, 128, 1000, 1024}) {
    char n[32]; snprintf(n, 32, "small%d", len); b3::hash_small(d.data(), len, nullptr, out); hex(n, out, 32);
    snprintf(n, 32, "ksmall%d", len); b3::hash_small(d.data(), len, key, out); hex(n, out, 32);
  }
  for (size_t ch : {1, 2, 64}) {
    b3::Pow2Tree t; t.build(d.data(), ch * 1024, key);
    t.root_with_chunk0(d.data(), out); char n[32]; snprintf(n, 32, "tree%zu", ch); hex(n, out, 32);
    t.root_with_chunk0(d.data() + 50000, out); snprintf(n, 32, "treex%zu", ch); hex(n, out, 32);
  }
  pearl::Header h; for (int i = 0; i < 32; i++) { h.prev_block[i] = i; h.merkle_root[i] = 0x80 + i; }
  h.timestamp = 0x6a000000; h.nbits = 0x1e100000;
  uint8_t hb[76]; pearl::header_bytes(h, hb); hex("header", hb, 76);
  uint8_t cb[52]; pearl::config_bytes(cb); hex("config", cb, 52);
  uint8_t jk[32], bs[32], as[32]; pearl::job_key(h, jk); hex("job_key", jk, 32);
  pearl::seed_b(jk, key, 128, bs); hex("seed_b", bs, 32);
  pearl::seed_a(bs, d.data(), 128, as); hex("seed_a", as, 32);
  for (uint32_t nb : {0x1e100000u, 0x1c020000u, 0x2000013bu, 0x03123456u, 0x02123456u, 0x2100ffffu}) {
    uint32_t b[8]; pearl::bound(nb, b); uint8_t bb[32]; b3::store_words(b, bb, 8);
    char n[32]; snprintf(n, 32, "bound%08x", nb); hex(n, bb, 32);
  }
  int8_t c0[1024]; pearl::chunk0(5, 3, c0); hex("chunk0", (uint8_t *)c0, 16);
}
