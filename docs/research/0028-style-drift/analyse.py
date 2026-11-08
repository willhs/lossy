"""Correlate per-clip rendering style with the per-shot generation inputs (research 0028).

Usage (from the repo root):
    uv run python docs/research/0028-style-drift/analyse.py OUTPUT_DIR STYLES_JSONL MTIMES_TXT OUT_CSV

OUTPUT_DIR   the run's output dir (shots.json, characters.json, decode_progress_runpod-ltx25.json)
STYLES_JSONL classify_style.py output, one row per clip stem
MTIMES_TXT   "<epoch> <clip>.mp4" lines (stat of clips/runpod-ltx25/*), used to infer pod sessions
OUT_CSV      per-clip table written for the research note

Rebuilds each clip's prompt with the real RunPodLtx25Strategy.format_prompt (identity
blocks + 2-character cap + name stripping), so the analysed text is what the model saw.
"""
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.getcwd())
from decode import build_character_shot_map  # noqa: E402
from prompt_format import vary_prompt_for_part  # noqa: E402
from strategies_video import RunPodLtx25Strategy  # noqa: E402

CARTOON = {"3d_animation", "2d_animation"}
POD_GAP_S = 600  # a gap this long between clip writes means a new pod (boot ~20 min)


def strategy(characters: dict) -> RunPodLtx25Strategy:
    s = object.__new__(RunPodLtx25Strategy)  # skip __init__: it builds a RunPod session
    s._init_character_identity(build_character_shot_map(characters), characters)
    return s


def part_prompts(strat, entry: dict, n_parts: int) -> list[str]:
    """Same per-part prompt choice as RunPodWanStrategy.generate."""
    base = strat.format_prompt(entry)
    segs = entry.get("temporal_segments") if n_parts > 1 else None
    out = []
    for i in range(n_parts):
        if segs:
            j = round(i * (len(segs) - 1) / max(n_parts - 1, 1))
            out.append(strat.format_prompt(dict(entry, description=segs[j])))
        else:
            out.append(vary_prompt_for_part(base, i, n_parts))
    return out


def rate(rows, key) -> list[tuple]:
    """Cartoon rate per value of key: (value, n, cartoon, rate)."""
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r["cartoon"])
    return sorted(((k, len(v), sum(v), sum(v) / len(v)) for k, v in g.items()), key=lambda t: str(t[0]))


def word_log_odds(rows, min_n=8) -> list[tuple]:
    """Smoothed log-odds of each prompt word appearing in a cartoon vs photoreal clip."""
    docs = [(set(re.findall(r"[a-z][a-z-]+", r["prompt"].lower())), r["cartoon"]) for r in rows]
    nc = sum(c for _, c in docs)
    npr = len(docs) - nc
    cnt = Counter()
    cc = Counter()
    for words, c in docs:
        for w in words:
            cnt[w] += 1
            cc[w] += c
    out = []
    for w, n in cnt.items():
        if n < min_n or n > len(docs) - min_n:
            continue
        a, b = cc[w], n - cc[w]
        lo = math.log((a + 0.5) / (nc - a + 0.5)) - math.log((b + 0.5) / (npr - b + 0.5))
        out.append((w, n, a, a / n, lo))
    return sorted(out, key=lambda t: -t[4])


def main(out_dir, styles_path, mtimes_path, out_csv):
    shots = {s["index"]: s for s in json.load(open(f"{out_dir}/shots.json"))["shots"]}
    chars = json.load(open(f"{out_dir}/characters.json"))
    progress = json.load(open(f"{out_dir}/decode_progress_runpod-ltx25.json"))
    styles = {}
    for line in open(styles_path):
        if line.strip():
            r = json.loads(line)
            styles[r["clip"]] = r
    mtimes = {}
    for line in open(mtimes_path):
        t, name = line.split()
        mtimes[name[:-4]] = int(t)

    # Pod sessions from gaps in clip write times.
    order = sorted(mtimes, key=lambda c: mtimes[c])
    pod, last, pod_of = 0, None, {}
    for c in order:
        if last is not None and mtimes[c] - last > POD_GAP_S:
            pod += 1
        pod_of[c] = pod
        last = mtimes[c]

    strat = strategy(chars)
    smap = build_character_shot_map(chars)
    rows = []
    for idx_s, parts in progress["clips"].items():
        idx = int(idx_s)
        entry = shots[idx]
        prompts = part_prompts(strat, entry, len(parts))
        present = sorted(smap.get(idx, []))
        blocks = strat._cap_to_most_prominent(smap.get(idx, []))
        for p_i, part in enumerate(parts):
            stem = part["path"][:-4]
            st = styles[stem]
            rows.append({
                "clip": stem, "shot": idx, "part": p_i + 1, "n_parts": len(parts),
                "style": st["style"], "cartoon": int(st["style"] in CARTOON),
                "lookalike": st["lookalike"],
                "seed": idx + p_i, "chained": int(p_i > 0),
                "duration_s": entry["duration_s"], "n_chars": len(present),
                "chars": "+".join(c.split("_")[-1] for c in present) or "none",
                "identity_blocks": "+".join(c.split("_")[-1] for c in sorted(blocks)) or "none",
                "kramer_block": int("cosmo_kramer" in blocks),
                "has_dialogue": int(bool(entry.get("dialogue"))),
                "setting": entry["description"].get("setting", ""),
                "pod": pod_of.get(stem, -1), "mtime": mtimes.get(stem, 0),
                "prompt_words": len(prompts[p_i].split()),
                "prompt": prompts[p_i],
            })
    rows.sort(key=lambda r: (r["shot"], r["part"]))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    first = [r for r in rows if r["part"] == 1]  # pure T2V clips: the style is chosen here
    print(f"clips={len(rows)} shots={len(first)} cartoon clips={sum(r['cartoon'] for r in rows)} "
          f"cartoon first parts={sum(r['cartoon'] for r in first)}")
    print("\nstyle counts (all clips):", Counter(r["style"] for r in rows))

    # Chaining: does a later part keep its first part's style?
    by_shot = defaultdict(list)
    for r in rows:
        by_shot[r["shot"]].append(r["cartoon"])
    multi = {k: v for k, v in by_shot.items() if len(v) > 1}
    mixed = [k for k, v in multi.items() if len(set(v)) > 1]
    print(f"\nmulti-part shots={len(multi)}, style flips within a shot={len(mixed)} {mixed}")

    def table(title, key, data=first):
        print(f"\n{title} (first parts)")
        for k, n, c, rt in rate(data, key):
            print(f"  {str(k):28s} n={n:3d} cartoon={c:3d} rate={rt:.2f}")

    table("identity blocks prepended", "identity_blocks")
    table("characters in shot", "chars")
    table("Kramer block prepended", "kramer_block")
    table("character count", "n_chars")
    table("pod session", "pod")
    table("has dialogue", "has_dialogue")
    for r in first:
        r["dur_bin"] = "<2s" if r["duration_s"] < 2 else "2-4.84s" if r["duration_s"] <= 4.84 else ">4.84s"
        r["words_bin"] = "<80" if r["prompt_words"] < 80 else "80-110" if r["prompt_words"] <= 110 else ">110"
        r["seed_parity"] = r["seed"] % 2
        r["setting_club"] = int(bool(re.search(r"comedy|club|stage", r["setting"], re.I)))
        r["half"] = "shots<150" if r["shot"] < 150 else "shots>=150"
    table("duration", "dur_bin")
    table("prompt length (words)", "words_bin")
    table("seed parity", "seed_parity")
    table("comedy-club setting", "setting_club")
    table("episode half", "half")

    # Controlled cuts: hold the setting fixed (no comedy club), split the identity blocks.
    def cut(label, sel):
        n, c = len(sel), sum(r["cartoon"] for r in sel)
        print(f"  {label:44s} n={n:3d} cartoon={c:3d} rate={c / max(n, 1):.2f}")
    apt = [r for r in first if not r["setting_club"]]
    nk = [r for r in apt if not r["kramer_block"]]
    print("\ncontrolled cuts (first parts, non-club settings)")
    cut("Kramer block", [r for r in apt if r["kramer_block"]])
    cut("no Kramer block", nk)
    cut("no Kramer, 1 identity block", [r for r in nk if r["identity_blocks"] != "none" and "+" not in r["identity_blocks"]])
    cut("no Kramer, 2 identity blocks", [r for r in nk if "+" in r["identity_blocks"]])
    cut("Jerry block only", [r for r in nk if r["identity_blocks"] == "seinfeld"])
    cut("Kramer in shot, block dropped by 2-cap", [r for r in first if "kramer" in r["chars"] and not r["kramer_block"]])

    # Same-style runs in shot order vs. a shuffle baseline (is there temporal stickiness?)
    seq = [r["cartoon"] for r in first]
    switches = sum(a != b for a, b in zip(seq, seq[1:]))
    p = sum(seq) / len(seq)
    print(f"\nstyle switches between consecutive shots={switches} of {len(seq)-1} "
          f"(independent draws would give ~{2*p*(1-p)*(len(seq)-1):.0f})")

    print("\nwords most associated with cartoon (log-odds, first parts, n>=8):")
    lo = word_log_odds(first)
    for w, n, a, rt, l in lo[:30]:
        print(f"  {w:16s} n={n:3d} cartoon={a:3d} rate={rt:.2f} lo={l:+.2f}")
    print("\nwords most associated with photoreal:")
    for w, n, a, rt, l in lo[-20:]:
        print(f"  {w:16s} n={n:3d} cartoon={a:3d} rate={rt:.2f} lo={l:+.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
