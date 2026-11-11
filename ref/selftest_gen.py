"""Ожидаемые значения самотеста pminer (m=256, n=128, k=4096, r=128, тайл 8×16 в регистрах потока).
Должно совпадать с selftest() в src/pminer.cu. Вывод вставляется в src/selftest.h."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pearlhash_ref as R  # noqa: E402

M64 = (1 << 64) - 1
SEED_A, SEED_B, SEED_N = 0xA11CE, 0xB0B, 0x5EED
M, N = 256, 128
K, RANK = 4096, 128
NBITS = 0x1E100000


def splitmix_arr(start, count):
    z = (np.arange(count, dtype=np.uint64) + np.uint64(start)) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def rnd(seed, start, count):
    return ((splitmix_arr(seed + start, count) % np.uint64(129)).astype(np.int16) - 64).astype(np.int8)


def gen(seed, start, count):
    """murmur3 fmix32 от idx*0x9E3779B1+seed, сужение ((x>>8)*129)>>24 - 64 (как pearl::gen_val)."""
    M32 = np.uint32(0xFFFFFFFF)
    x = (np.arange(start, start + count, dtype=np.uint64) * np.uint64(0x9E3779B1) + np.uint64(seed)) & np.uint64(M32)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x85EBCA6B)) & np.uint64(M32)
    x ^= x >> np.uint64(13)
    x = (x * np.uint64(0xC2B2AE35)) & np.uint64(M32)
    x ^= x >> np.uint64(16)
    return ((((x >> np.uint64(8)) * np.uint64(129)) >> np.uint64(24)).astype(np.int16) - 64).astype(np.int8)


def matrices(nonce=0):
    a = gen(SEED_A, 0, M * K)
    a[:1024] = rnd(SEED_N, nonce * 1024, 1024)  # чанк 0 — splitmix от nonce
    bt = gen(SEED_B, 0, N * K)
    return a.reshape(M, K), bt.reshape(N, K)


def main():
    with np.errstate(over="ignore"):
        a, bt = matrices()
    rows = R.Pattern(list(range(0, 64, 8)))
    cols = R.Pattern([j + b for j in range(0, 64, 8) for b in (0, 1)])
    hdr = R.header_bytes(1, bytes(range(32)), bytes(0x80 + i for i in range(32)), 0x6A000000, NBITS)
    dig = np.zeros(8, np.uint32)
    hits = []
    for (r0, c0), h, hit in R.mine_once(hdr, K, RANK, rows, cols, a, bt):
        dig ^= np.frombuffer(h, "<u4")
        if hit:
            hits.append((r0, c0))
    print("// сгенерировано ref/selftest_gen.py")
    print(f"static const unsigned SELFTEST_HITS = {len(hits)};")
    print("static const uint32_t SELFTEST_DIGEST[8] = {" + ", ".join(f"0x{int(x):08x}u" for x in dig) + "};")
    print(f"// первые удачи (row0,col0): {sorted(hits)[:6]}")


if __name__ == "__main__":
    main()
