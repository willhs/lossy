---
id: 0020-wan22-identity-i2v-verification
type: note
purpose: "Verify, end-to-end on a real render, that RunPodWan22Strategy's locked character identity prefix (CharacterIdentityMixin) and I2V start_image chaining for split shots both work as implemented, and document the known Leia/Obi-Wan misattribution regression."
scope: ["research", "decode", "wan22", "character-identity", "i2v-chaining"]
tags: ["research", "verification", "wan22", "character-consistency", "i2v"]
related: ["research/0011-character-continuity/research.md", "research/0012-wan-model-variants/research.md"]
---

## Question

Two quality features on the `runpod-wan22` decode path were implemented but never checked against a real render:

1. **Locked character identity** — `CharacterIdentityMixin` (strategies_video.py:60-97) prepends a canonical, byte-identical `"{display_name}: {description}"` string per character to every prompt. As of commit `98a3154`, the old `prompt_blend.py` LLM-rewrite path was deleted — the string is now pure verbatim concatenation, no rewording.
2. **I2V `start_image` chaining** — `RunPodWan22Strategy._supports_i2v = True` conditions each split-shot part after the first on the previous part's last frame, to remove visible seams on long shots.

Test clip: `sw_r2_leia` (28 shots, R2-D2 recurs in shots 8/19/26; shot 2 is the longest split shot at 8.97s / 3 parts).

## Method

- **Baseline (pre-98a3154)**: reused the already-decoded+stitched clip at `/Users/will/projects/lossy/output/sw_r2_leia/sw_r2_leia_reconstructed_runpod-wan22.mp4` (28 shots, cost $0.88 — "the overnight job"). Not re-decoded.
- **After (post-98a3154)**: copied the stage1-3-only bench `shots.json`/`characters.json` from `/Users/will/workspaces/give-gemini-describe-pass-previous-shot-b493dc/lossy/output/sw_r2_leia_bench/` into `output/sw_r2_leia_after/` in this worktree, then ran `python decode.py output/sw_r2_leia_after --strategy runpod-wan22` (generation) followed by `python decode.py output/sw_r2_leia_after --strategy runpod-wan22 --stitch` (stitching) via a RunPod A6000-tier pod.
- Both baseline and after use the same deterministic per-shot seed (`seed=idx` in decode.py), so "same seeds" holds automatically without extra flags.
- Wrote `tools/dump_format_prompt.py` (no GPU/pod needed) to call `RunPodWan22Strategy.format_prompt()` directly against the AFTER shots/characters data, to inspect the exact prompt text before spending any RunPod time.

## Finding 1: locked character identity — confirmed

`tools/dump_format_prompt.py output/sw_r2_leia_after 2 8 19 26` shows the R2-D2 identity block is **byte-identical** across shots 2, 8, 19, and 26:

```
R2-D2: R2-D2 is an astromech droid, appearing as a short, blue and white robot with a silver dome. He has no age or gender, and his "skin" is metallic. His "eyes" are an optical sensor that glows. He wears no costume, as his appearance is his functional design.
```

(261 characters, confirmed identical via direct string comparison against `characters.json`'s `r2_d2` entry — this is exactly what `CharacterIdentityMixin.format_prompt()` at strategies_video.py:81-97 is documented to guarantee: "Identical tokens hold continuity, not more detail — the same string is prepended for a character in every shot, never reworded.") The block is prepended verbatim ahead of the shot's base-formatted description, with no LLM rewriting step in between (confirmed: `prompt_blend.py` is gone as of 98a3154, and `format_prompt()`'s only transformation is string concatenation).

For multi-character shots (e.g. shot 2: Luke + R2-D2), each character's block is prepended in turn, space-joined, ahead of the base prompt — also confirmed directly from the dump.

## Finding 2: I2V start_image chaining — confirmed

The AFTER decode ran 28 shots for $0.56 total (1.3hrs pod time, well under the $3-4 budget cap). Shot 2 (8.97s) split into 3 parts as expected: `clips/runpod-wan22/0002-01.mp4` (97 frames, 4.04s), `-02` (97 frames, 4.04s), `-03` (49 frames, 2.04s).

**Stitch behavior confirmed**: comparing raw clips to `adjusted/runpod-wan22/`, parts 1 and 2 (non-final) keep their full frame count (97→97, only sped up via a frame-rate change: 24fps → 325/12fps for retiming), while only the final part 3 is trimmed (49→46 frames). This matches `_retime_shot_clips`'s documented behavior exactly — non-final parts are never trimmed because trimming would cut into the exact frame the next part's I2V generation was conditioned on.

**Visual continuity confirmed**: extracted the last frame of part N and first frame of part N+1 at both boundaries (part1→part2, part2→part3). Luke and R2-D2's poses, lighting, and composition are near-identical across each boundary — no visible jump-cut, confirming `_upload_start_frame`'s `start_image` conditioning is working as designed.

## Finding 3: Leia/Obi-Wan hologram misattribution — confirmed, pre-existing, not fixed here

The bench `characters.json` (post-98a3154 stage2/3 run) attributes the shot 6-7 hologram to `obiwan_kenobi` ("An elderly human male, appearing as a blue, translucent hologram... hooded robe"), where the baseline `characters.json` correctly attributes it to `princess_leia_organa`. This is not just a mislabeled character-registry entry — it goes deeper: the *shot's own base description* (written by the Gemini describe pass, independent of the character registry) also says "the Obi-Wan hologram" and "a holographic figure of Obi-Wan Kenobi," confirmed via the `dump_format_prompt.py` misattribution check:

```
Obi-Wan "Ben" Kenobi: An elderly human male, appearing as a blue, translucent hologram...
A blue, translucent holographic figure of Obi-Wan Kenobi, wearing a hooded robe, stands in the center of the frame...
```

So the AFTER render's shots 6-7 visually depict an elderly hooded male hologram, not Leia — confirmed directly in the render (frame grab at t=28s in the comparison video: BEFORE shows a robed female figure, AFTER shows a helmeted/hooded male figure). This is a real, visible regression, attributed by the 98a3154 commit message to Gemini world-knowledge bias on the "help me, Obi-Wan" line (the previous-shot-context change is unrelated to this; it's a stage2/3 describe-pass artifact, not something introduced by the identity-locking or I2V-chaining work). This verification pass does not fix it — flagging only, as scoped.

By contrast, R2-D2 and C-3PO (shots 8/19/26) render as consistently recognizable, correctly-identified characters in both BEFORE and AFTER — the misattribution is isolated to the one shot pair where Gemini's stage2/3 pass got confused, not a general regression in character identification.

## Comparison video

`output/comparisons/sw_r2_leia_before_after_vstack.mp4` (22MB, 95s, before on top / after on bottom, labeled) — not committed to git (output/ is gitignored); this is the upload target for the Reddit reply.

## Summary for the Reddit reply

Locked character-identity prefixes are confirmed byte-identical across recurring characters (R2-D2 in shots 2/8/19/26) in a real render — no more per-clip appearance drift from prompt rewriting, and R2-D2/C-3PO look consistent and correctly identified in both before and after. I2V start_image chaining on a 3-part split shot (shot 2, 8.97s) is confirmed working — frame-level continuity checks at both part boundaries show no visible seam, and stitch.py correctly avoids trimming non-final parts so the chained conditioning frame is never cut. One known regression, unrelated to either fix: the Gemini describe pass misattributed the Leia hologram scene (shots 6-7) to Obi-Wan Kenobi in this test clip — a world-knowledge bias bug in the stage2/3 describe pass, visibly wrong in the render, flagged for follow-up rather than fixed here.
