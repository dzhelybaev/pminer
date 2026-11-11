"""Сверка tests/host_test.cpp с эталоном: python3 tests/host_test.py <вывод host_test>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
import pearlhash_ref as R  # noqa: E402
from blake3 import blake3  # noqa: E402

M64 = (1 << 64) - 1


def splitmix(z):
    z = (z + 0x9e3779b97f4a7c15) & M64
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & M64
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & M64
    return z ^ (z >> 31)


got = dict(l.split() for l in open(sys.argv[1]))
key = bytes((i * 7 + 1) & 255 for i in range(32))
d = bytes(splitmix(i) & 255 for i in range(1 << 16))
exp = {}
for n in [0, 1, 63, 64, 65, 128, 1000, 1024]:
    exp[f"small{n}"] = blake3(d[:n]).hexdigest()
    exp[f"ksmall{n}"] = blake3(d[:n], key=key).hexdigest()
for ch in [1, 2, 64]:
    exp[f"tree{ch}"] = blake3(d[:ch * 1024], key=key).hexdigest()
    exp[f"treex{ch}"] = blake3(d[50000:51024] + d[1024:ch * 1024], key=key).hexdigest()
rows, cols = R.Pattern(list(range(0, 64, 8))), R.Pattern([j + b for j in range(0, 64, 8) for b in (0, 1)])
hdr = R.header_bytes(1, bytes(range(32)), bytes(0x80 + i for i in range(32)), 0x6a000000, 0x1e100000)
exp["header"] = hdr.hex()
cfg = R.config_bytes(4096, 128, rows, cols)
exp["config"] = cfg.hex()
jk = R.bk(hdr + cfg)
exp["job_key"] = jk.hex()
hb = R.bk(key + (128).to_bytes(4, "little") + bytes(28), R.SALT_B)
bs = R.bk(jk + hb)
exp["seed_b"] = bs.hex()
ha = R.bk(d[:32] + (128).to_bytes(4, "little") + bytes(28), R.SALT_A)
exp["seed_a"] = R.bk(bs + ha).hex()
for nb in [0x1e100000, 0x1c020000, 0x2000013b, 0x03123456, 0x02123456, 0x2100ffff]:
    exp[f"bound{nb:08x}"] = R.bound(nb, 8, 16, 4096, 128).to_bytes(32, "little").hex()
exp["chunk0"] = bytes(((splitmix(5 + 3 * 1024 + i) % 129) - 64) & 255 for i in range(16)).hex()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
import selftest_gen as S  # noqa: E402
exp["gen"] = S.gen(0xA11CE, 0, 64).tobytes().hex()
bad = [k for k in exp if got.get(k) != exp[k]]
for k in bad:
    print("FAIL", k, got.get(k), exp[k])
print(f"{len(exp) - len(bad)}/{len(exp)} OK")
sys.exit(1 if bad else 0)
