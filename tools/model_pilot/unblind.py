"""Map a blind judge's letter scores back to models and summarise (research 0025).

    uv run python tools/model_pilot/unblind.py output/pilot0025 output/pilot0025/blind_judge_claude.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

DIMS = ("identity", "prompt_adherence", "artefacts", "motion", "overall")


def main():
    ap = argparse.ArgumentParser(description="Unblind judge scores")
    ap.add_argument("root")
    ap.add_argument("judged")
    ap.add_argument("--key", default="blind_key.json")
    a = ap.parse_args()
    key = json.load(open(os.path.join(a.root, a.key)))
    bench = {s["index"]: s for s in json.load(open(os.path.join(a.root, "benchmark.json")))["shots"]}
    judged = json.load(open(a.judged))
    per_model = defaultdict(lambda: defaultdict(list))
    per_cat = defaultdict(lambda: defaultdict(list))
    wins = defaultdict(int)
    rows = []
    for r in judged["results"]:
        letters = key[str(r["shot"])]
        cat = bench[r["shot"]]["category"].split(":")[0]
        row = {"shot": r["shot"], "category": bench[r["shot"]]["category"], "best": letters.get(r.get("best"))}
        for letter, sc in r["scores"].items():
            m = letters[letter]
            for d in DIMS:
                per_model[m][d].append(float(sc[d]))
            per_cat[cat][m].append(float(sc["overall"]))
            row[m] = {d: sc[d] for d in DIMS} | {"note": sc.get("note", "")}
        wins[row["best"]] += 1
        rows.append(row)
    summary = {m: {d: round(sum(v) / len(v), 2) for d, v in dims.items()} | {"wins": wins[m], "n": len(dims["overall"])}
               for m, dims in per_model.items()}
    cats = {c: {m: round(sum(v) / len(v), 2) for m, v in ms.items()} for c, ms in per_cat.items()}
    out = os.path.splitext(a.judged)[0] + "_unblinded.json"
    json.dump({"judge": judged.get("judge"), "summary": summary, "by_category_overall": cats, "rows": rows},
              open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(json.dumps(cats, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
