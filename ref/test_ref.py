"""Сверка эталона с векторами, сгенерированными Rust-кодом Pearl (vectors.json).
Запуск: python3 -I ref/test_ref.py ref/vectors.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pearlhash_ref as R  # noqa: E402


def check(name, got, exp):
    ok = got == exp
    print(f"  {'OK ' if ok else 'FAIL'} {name}")
    return ok


def main(path):
    allok = True
    for i, c in enumerate(json.load(open(path))):
        m, n, k, r = c["m"], c["n"], c["k"], c["rank"]
        print(f"case {i}: m={m} n={n} k={k} r={r} rows={c['rows_pattern']} cols={c['cols_pattern']}")
        rows, cols = R.Pattern(c["rows_pattern"]), R.Pattern(c["cols_pattern"])
        hdr = bytes.fromhex(c["header"])
        a = np.frombuffer(bytes.fromhex(c["a"]), np.int8).reshape(m, k)
        bt = np.frombuffer(bytes.fromhex(c["bt"]), np.int8).reshape(n, k)
        cfg = R.config_bytes(k, r, rows, cols)
        allok &= check("config bytes", cfg.hex(), c["config"])
        jk = R.bk(hdr + cfg)
        allok &= check("job_key", jk.hex(), c["job_key"])
        bs, as_ = R.noise_seeds(jk, a, bt)
        allok &= check("b_noise_seed", bs.hex(), c["b_noise_seed"])
        allok &= check("a_noise_seed", as_.hex(), c["a_noise_seed"])
        na, nbt = R.noise(as_, bs, m, n, k, r)
        allok &= check("noise A", na.tobytes().hex(), c["noise_a"])
        allok &= check("noise B^T", nbt.tobytes().hex(), c["noise_bt"])
        bnd = R.bound(c["nbits"], len(rows.list), len(cols.list), k, r)
        allok &= check("bound", bnd.to_bytes(32, "little").hex(), c["bound_le"])
        exp = {(t["row0"], t["col0"]): t for t in c["tiles"]}
        bad = hits = 0
        for (key, h, hit) in R.mine_once(hdr, k, r, rows, cols, a, bt):
            t = exp.pop(key)
            bad += t["hash"] != h.hex() or t["hit"] != hit
            hits += hit
        allok &= check(f"tiles hash/hit ({len(c['tiles'])} тайлов, удач {hits}, ошибок {bad})",
                       bad == 0 and not exp, True)
    print("ВСЁ СОВПАЛО" if allok else "ЕСТЬ РАСХОЖДЕНИЯ")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "vectors.json")))
