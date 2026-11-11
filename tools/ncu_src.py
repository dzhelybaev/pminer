#!/usr/bin/env python3
# Выжимка по инструкциям SASS из /root/ncu/gpuN.sass.csv (tools/ncu.sh):
# топ по лишним обращениям к smem и по выборкам простоев warp.
#   python3 ncu_src.py 0      — GPU0
import csv, sys

g = sys.argv[1] if len(sys.argv) > 1 else "0"
path = f"/root/ncu/gpu{g}.sass.csv"
rows = list(csv.reader(open(path, encoding="utf-8", errors="replace")))
# первая строка с «Source» — заголовок
hi = next(i for i, r in enumerate(rows) if any(c.strip() == "Source" for c in r))
head = [c.strip() for c in rows[hi]]
data = [r for r in rows[hi + 1:] if len(r) == len(head)]


def col(*keys):
    for i, h in enumerate(head):
        if all(k.lower() in h.lower() for k in keys):
            return i
    return None


def num(s):
    try:
        return float(s.replace(",", "").replace("%", ""))
    except ValueError:
        return 0.0


src, adr = col("Source"), col("Address")
print("колонки:", len(head), "строк:", len(data))
for title, keys in [("лишние wavefront smem", ("Shared", "Excessive")),
                    ("wavefront smem", ("Wavefronts", "Shared")),
                    ("выборки простоев (все)", ("Stall Sampling", "All")),
                    ("выборки простоев (не выданные)", ("Stall Sampling", "Not"))]:
    c = col(*keys)
    if c is None:
        print(f"== {title}: колонки нет")
        continue
    tot = sum(num(r[c]) for r in data)
    print(f"== {title} [{head[c]}], всего {tot:.0f}")
    for r in sorted(data, key=lambda r: -num(r[c]))[:12]:
        v = num(r[c])
        if v <= 0:
            break
        a = r[adr] if adr is not None else ""
        print(f"  {v:10.0f} {100 * v / tot:5.1f}%  {a:>6}  {r[src].strip()[:70]}")
# какие ещё колонки есть (для следующего запроса)
print("== колонки:", "; ".join(h for h in head if h)[:1500])
