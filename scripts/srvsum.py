import json, sys, statistics as st
from collections import defaultdict
d = defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l); d[(r["config"], r["prompt"])].append(r)
for (c, p), rs in sorted(d.items()):
    tg = [r["tg_ts"] for r in rs if r["tg_ts"]]; pp = [r["pp_ts"] for r in rs if r["pp_ts"]]
    acc = [r["draft_acc"] / r["draft_n"] for r in rs if r.get("draft_n")]
    print(f"{c:8} {p:8} n={len(rs)} tg={st.median(tg):.2f} ({min(tg):.1f}-{max(tg):.1f}) pp={st.median(pp):.1f} acc={st.median(acc) if acc else 0:.2f} maxT={max(r['max_temp'] for r in rs)}")
