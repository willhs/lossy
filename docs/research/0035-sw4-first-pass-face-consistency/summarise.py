"""Research 0035's pre-registered numbers from score_window.py's score.json, one arm per dir.

    summarise.py SCORES_DIR [SCORES_DIR ...]   (each holds score.json; the first is the baseline)

K-mid: mean middle-frame cosine to the sheet over the 9 single-person keyframe shots.
Drift: mean (keyframe - last) and (keyframe - middle) cosine to the sheet over those shots
that have both faces labelled. M and pooled: across-shot cosine, face-weighted over the three
characters (0034's definitions).
"""
import json
import os
import sys

KEY_SHOTS = {"han_solo": [1285, 1287, 1307], "obi_wan_ben_kenobi": [1290, 1292, 1296],
             "princess_leia_organa": [1286, 1297, 1321]}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarise(path):
    lab = json.load(open(os.path.join(path, "score.json")))["labelled"]
    mid, d_last, d_mid, per = [], [], [], {}
    for name, shots in KEY_SHOTS.items():
        pos = {p: {int(k): v for k, v in lab[name].get(p, {}).get("per_shot_to_sheet", {}).items()}
               for p in ("keyframe", "middle", "last")}
        for s in shots:
            k, m, l_ = pos["keyframe"].get(s), pos["middle"].get(s), pos["last"].get(s)
            per[s] = (k, m, l_)
            if m is not None:
                mid.append(m)
            if k is not None and l_ is not None:
                d_last.append(k - l_)
            if k is not None and m is not None:
                d_mid.append(k - m)

    def weighted(pos):
        cells = [lab[n][pos] for n in KEY_SHOTS if pos in lab[n] and lab[n][pos]["faces"] > 1]
        return sum(c["faces"] * c["across"] for c in cells) / sum(c["faces"] for c in cells)

    return {"K_mid": mean(mid), "n_mid": len(mid), "drift_last": mean(d_last), "n_last": len(d_last),
            "drift_mid": mean(d_mid), "M": weighted("middle"), "pooled": weighted("pooled"), "per_shot": per}


def main():
    rows = [(os.path.basename(p.rstrip("/")), summarise(p)) for p in sys.argv[1:]]
    base = rows[0][1]
    print(f"{'arm':<12} {'K-mid':>6} {'dK':>6} {'drift k->l':>10} {'d':>6} {'drift k->m':>10} {'M':>6} {'dM':>6} {'pooled':>7} {'dP':>6}")
    for name, r in rows:
        print(f"{name:<12} {r['K_mid']:6.3f} {r['K_mid'] - base['K_mid']:+6.3f} {r['drift_last']:10.3f} "
              f"{r['drift_last'] - base['drift_last']:+6.3f} {r['drift_mid']:10.3f} {r['M']:6.3f} "
              f"{r['M'] - base['M']:+6.3f} {r['pooled']:7.3f} {r['pooled'] - base['pooled']:+6.3f}"
              f"   (n mid {r['n_mid']}, n k->l {r['n_last']})")
    print("\nper shot keyframe / middle / last to sheet:")
    for s in sorted(base["per_shot"]):
        cells = "   ".join(f"{n}: " + "/".join("  -  " if v is None else f"{v:5.2f}" for v in r["per_shot"][s])
                           for n, r in rows)
        print(f"  {s}  {cells}")


if __name__ == "__main__":
    main()
