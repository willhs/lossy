---
id: 0034-sw4-mixed-shot-keyframes
type: note
purpose: "Why SW4 characters were not consistent in the 2-minute window test, whether keyframing people who share a shot with troopers/droids fixes it, and a go/no-go for the 30 Nov run."
scope: ["research", "decode", "keyframes", "character-continuity", "star-wars"]
tags: ["research", "star-wars", "ltx-2.5", "keyframe", "klein", "face-consistency", "go-no-go"]
related:
  - "research/0030-sw4-full-run-preflight/research.md"
  - "research/0029-character-reference-continuity/research.md"
---

# SW4 character consistency: mixed-shot keyframes (23 Nov 2026)

Vision alignment: Have an online presence (goal: complete a movie or TV show reconstruction, Star Wars IV)

Steady task `82e4bf88`. Will watched research 0030's 2-minute window (shots 1284-1321) and "didn't
see the character consistency I was hoping for". The full run (task `0725e309`) is on hold until
this is answered.

## Pre-registration (written and committed before the new window was rendered or labelled)

**What was already known when this was written.** The flag below is built and unit-tested. The
old window (0030, low res + native audio, `output/sw4-window-test-20261123/`) and the source
film were run through the new candidate extractor, but neither was labelled or scored. One
thing was found on the way, and it is why the measure changes: 0030's `window_faces.py` embeds
"the first face YuNet finds". In this window that is often a stormtrooper helmet (YuNet scores
helmets up to 0.88), Chewbacca (0.82) or the other person in the shot (Luke in Han's 1285/1287),
and the shot map tags 1297 as Leia where the source shows Han from behind. Run unchanged on the
**real film**, 0030's measure gives Han 0.18, Obi-Wan 0.32, Leia 0.03 across shots (middle frame):
as low as the reconstruction. So 0030's 0.17-0.26 is mostly measuring the wrong faces.

**Measure** ([`score_window.py`](score_window.py)). Shots: the 15 whose shot map names exactly one
person (0030's set; Han 6, Obi-Wan 6, Leia 3). Frames: first, middle, last (0.06 s in from each
cut), scaled to 1280x704. Every YuNet face >= 40 px is a candidate, embedded with SFace. Whose face
it is gets decided by eye from the crop sheet under a fixed rule: the character's own human face,
never a helmet, droid or Wookiee, never the other named person (Luke blond young man, Han
dark-haired man, Obi-Wan old bearded man, Leia young woman); two that fit equally, the larger;
under 60 px or turned away, dropped. The automatic "largest face" number is reported beside it.

- **Primary (M):** mean pairwise SFace cosine across shots of each character's labelled middle-frame
  face, averaged over Han, Obi-Wan and Leia weighted by labelled faces. Low res + native audio arm
  (Will's chosen setup), old window vs new window. Ceiling: the source film, same labelling.
- **Secondary:** cosine to the character's sheet (both windows against the new run's sheets: the
  old run's sheets were not kept; Luke/Han/Obi-Wan have the same prompt and seed, Leia's block
  changed in `18e2855`), and the drift view on the new run's keyframe shots: drawn keyframe,
  first, middle and last frame, each to the sheet.

**Decision rules.**

1. Mixed-shot routing is worth turning on if **M rises by >= 0.10** over the old window, with
   troopers/droids still right on the contact sheet and no shot drawn with an extra or merged
   person that the text path did not have.
2. Where the loss is: keyframe-to-sheet **< 0.45** means klein/sheet (reference strength or sheet
   quality); keyframe-to-last-frame drop **>= 0.15** means LTX drifting from the first frame.
3. Face-swap post-pass (task step 4) is tried if the new **M < 0.40** (SFace's same-person line is
   about 0.36; Seinfeld 0029 had 0.55) **and** the source ceiling is clearly higher, so there is
   something to gain, and only if it can run inside the $6 cap with no new project dependency.
4. Go for 30 Nov with the flag on if rule 1 passes. If it fails, the 30 Nov recommendation is the
   0030 setup without the flag, and the consistency problem is written up as what it is.

Spend cap for the whole task: **$6 RunPod**, via `tools/capped_run.py` with this run's ownership
journal; 0 pods at the end.
