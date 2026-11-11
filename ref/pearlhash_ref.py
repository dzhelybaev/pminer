"""Эталон pearlhash (Pearl PoUW, плотный GEMM, сертификат V3 «salted») на numpy.

Повторяет zk-pow/src/ffi/mine.rs::try_mine_one из pearl-research-labs/pearl
(коммит 0dbdb2f). Нужен для побитовой сверки будущего CUDA-ядра.

Конвейер одной попытки:
  1. job_key = blake3(header(76) | config(52))
  2. hash_A = blake3_keyed(job_key, pad1024(A row-major)), hash_B — то же для B^T
     V3: hash_A' = blake3_keyed(SALT_A, hash_A | m_le32 | 0*28), hash_B' — с n
  3. b_seed = blake3(job_key | hash_B'), a_seed = blake3(b_seed | hash_A')
  4. шум низкого ранга r: NA = E_AL·E_AR, NB^T = E_BR·E_BL^T, где E_AL/E_BR —
     равномерные [-32,31], E_AR/E_BL — разреженные (+1/-1 в каждой строке)
  5. A' = A+NA, B' = B+NB (всё в int8: |A|<=64, |N|<=63)
  6. для каждого тайла (строки rows_pattern × столбцы cols_pattern):
     каждые r элементов k — XOR всех int32 аккумуляторов тайла,
     jackpot[t % 16] = rotl(jackpot[t % 16], 13) ^ xor
  7. hash = blake3_keyed(a_seed, jackpot LE), удача если LE-число <= bound,
     bound = target(nbits) * (h*w) * k'
"""
import numpy as np
from blake3 import blake3

JACKPOT_SIZE = 16
LROT = 13
SALT_A = blake3(b"pearl/cert-v3/noise-seed/A").digest()
SALT_B = blake3(b"pearl/cert-v3/noise-seed/B").digest()
LABEL_A = b"A_tensor".ljust(32, b"\0")
LABEL_B = b"B_tensor".ljust(32, b"\0")
U256_MAX = (1 << 256) - 1


def bk(data: bytes, key: bytes | None = None) -> bytes:
    return (blake3(data, key=key) if key else blake3(data)).digest()


# ---------- PeriodicPattern ----------
class Pattern:
    NUM_DIMS = 3

    def __init__(self, lst):
        p = list(lst)
        assert p[0] == 0 and all(a < b for a, b in zip(p, p[1:]))
        shape = []
        while len(p) > 1:
            for period in range(1, len(p)):
                if len(p) % period == 0:
                    s = p[period]
                    if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                        shape.append((s, len(p) // period))
                        p = p[:period]
                        break
            else:
                raise ValueError("pattern is not periodic")
        shape.reverse()
        per = shape[-1][0] * shape[-1][1] if shape else 1
        while len(shape) < self.NUM_DIMS:
            shape.append((per, 1))
        self.shape = shape
        self.list = list(lst)

    def to_bytes(self) -> bytes:
        out, mn = bytearray(), 1
        for stride, length in self.shape:
            out += bytes([stride // mn - 1, length - 1])
            mn = stride * length
        return bytes(out)

    @property
    def period(self):
        s, l = self.shape[-1]
        return s * l

    def offset_valid(self, off):
        for stride, length in reversed(self.shape):
            off %= stride * length
            if off >= stride:
                return False
        return True

    def partitions(self, total):
        assert total % self.period == 0
        base = np.array(self.list)
        return [base + o for o in range(total) if self.offset_valid(o)]


# ---------- сериализация ----------
def header_bytes(version, prev_block, merkle_root, timestamp, nbits) -> bytes:
    return (version.to_bytes(4, "little") + bytes(prev_block)[::-1] + bytes(merkle_root)[::-1]
            + timestamp.to_bytes(4, "little") + nbits.to_bytes(4, "little"))


def config_bytes(k, rank, rows: Pattern, cols: Pattern) -> bytes:
    return (k.to_bytes(4, "little") + rank.to_bytes(2, "little") + (0).to_bytes(2, "little")
            + rows.to_bytes() + cols.to_bytes() + bytes(32))


def pad1024(b: bytes) -> bytes:
    return b + bytes(-len(b) % 1024)


# ---------- сиды ----------
def noise_seeds(job_key, a: np.ndarray, bt: np.ndarray, salted=True):
    m, n = a.shape[0], bt.shape[0]
    ha = bk(pad1024(a.astype(np.int8).tobytes()), job_key)
    hb = bk(pad1024(bt.astype(np.int8).tobytes()), job_key)
    if salted:
        ha = bk(ha + m.to_bytes(4, "little") + bytes(28), SALT_A)
        hb = bk(hb + n.to_bytes(4, "little") + bytes(28), SALT_B)
    b_seed = bk(job_key + hb)
    a_seed = bk(b_seed + ha)
    return b_seed, a_seed


# ---------- шум ----------
def _rand_hash(i, label, key, slot):
    pre = bytearray(32)
    pre[slot * 4: slot * 4 + 4] = (1 + i).to_bytes(4, "little", signed=True)
    return bk(bytes(pre) + label, key)


def uniform(label, key, rows, r):
    """rows × r, значения (byte & 63) - 32. Хэш №b покрывает элементы [32b, 32b+32)."""
    nb = rows * r // 32
    raw = np.frombuffer(b"".join(_rand_hash(b, label, key, 0) for b in range(nb)), np.uint8)
    return ((raw & 63).astype(np.int16) - 32).reshape(rows, r)


def perm(label, key, k, r):
    """k пар (first, second): first = u & (r-1), second = first ^ (1 + mulhi(r-1, u))."""
    nh = -(-k // 8)
    u = np.frombuffer(b"".join(_rand_hash(i, label, key, 1) for i in range(nh)), "<u4")[:k].astype(np.uint64)
    first = u & (r - 1)
    second = first ^ (1 + (((r - 1) * u) >> 32))
    return first.astype(np.int64), second.astype(np.int64)


def noise(a_seed, b_seed, m, n, k, r):
    eal = uniform(LABEL_A, a_seed, m, r)
    f, s = perm(LABEL_A, a_seed, k, r)
    na = eal[:, f] - eal[:, s]
    ebr = uniform(LABEL_B, b_seed, n, r)
    f, s = perm(LABEL_B, b_seed, k, r)
    nbt = ebr[:, f] - ebr[:, s]
    return na.astype(np.int8), nbt.astype(np.int8)


# ---------- сложность ----------
def nbits_target(nbits):
    e, mant = nbits >> 24, nbits & 0xFFFFFF
    if mant == 0 or e == 0 or mant & 0x800000:
        return 0
    t = mant >> (8 * (3 - e)) if e <= 3 else mant << (8 * (e - 3))
    return min(t, U256_MAX)


def bound(nbits, h, w, k, r):
    t = nbits_target(nbits) * h * w * (k - k % r)
    return min(t, U256_MAX)


# ---------- jackpot ----------
def rotl(x, s):
    return ((x << s) | (x >> (32 - s))) & 0xFFFFFFFF


def tile_jackpot(ap: np.ndarray, btp: np.ndarray, r: int):
    """ap: h×k, btp: w×k (int, уже с шумом). Возвращает 16 u32."""
    h, k = ap.shape
    jp = [0] * JACKPOT_SIZE
    acc = np.zeros((h, btp.shape[0]), np.int64)
    for c in range(k // r):
        sl = slice(c * r, (c + 1) * r)
        acc += ap[:, sl].astype(np.int64) @ btp[:, sl].astype(np.int64).T
        x = int(np.bitwise_xor.reduce((acc & 0xFFFFFFFF).astype(np.uint32), axis=None))
        t = c % JACKPOT_SIZE
        jp[t] = rotl(jp[t], LROT) ^ x
    return jp


def jackpot_hash(jp, a_seed):
    return bk(b"".join(v.to_bytes(4, "little") for v in jp), a_seed)


def mine_once(header: bytes, k, r, rows: Pattern, cols: Pattern, a, bt, salted=True):
    """Одна попытка: проверка всех тайлов матриц A (m×k) и B^T (n×k). Генератор (tile, hash, hit)."""
    nbits = int.from_bytes(header[72:76], "little")
    job_key = bk(header + config_bytes(k, r, rows, cols))
    b_seed, a_seed = noise_seeds(job_key, a, bt, salted)
    na, nbt = noise(a_seed, b_seed, a.shape[0], bt.shape[0], k, r)
    ap = a.astype(np.int16) + na
    btp = bt.astype(np.int16) + nbt
    bnd = bound(nbits, len(rows.list), len(cols.list), k, r)
    for ri in rows.partitions(a.shape[0]):
        for ci in cols.partitions(bt.shape[0]):
            h = jackpot_hash(tile_jackpot(ap[ri], btp[ci], r), a_seed)
            yield (int(ri[0]), int(ci[0])), h, int.from_bytes(h, "little") <= bnd
