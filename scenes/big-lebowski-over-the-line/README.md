# Scene: The Big Lebowski — "Over the line!"

Standalone authored-manifest scene for the `generate-scene` skill (Wan 2.2 5B on fal.ai).
Not part of the Star Wars IV reconstruction, not the source-encoded `output/big_lebowski`
pipeline run — this is a hand-written shot list per
`docs/research/0024-authored-manifest-verification`.

## Source beat (for reference, not reproduced verbatim in prompts)

Real scene runs ~60–90s. Dialogue beats, timestamped from `output/big_lebowski/subtitles.srt`:

- `00:17:05` — "Over the line!"
- `00:17:07` — "Sorry, you were over the line. That's a foul."
- `00:17:11`–`00:17:24` — foul-call argument (Smokey, Walter, Dude)
- `00:17:37`–`00:18:07` — Walter draws the gun, "you're entering a world of pain" / "mark it zero"

The manifest reconstructs the *visual* beats only (foul → draw → standoff → reactions).
Wan 2.2 5B has no lip-sync/dialogue audio, so no character names or literal lines are used
in prompts — each character is described by physical appearance only, per the skill's
"no trademarked names" guidance and the identity-bleed notes from research 0023.

## Shot list (`scene.json`)

12 shots, ~5s each (two chained to ~10s each) → 14 clips, ~65-70s total. Shots 1-6 are the
original confrontation; shots 7-12 were added later to roughly double the scene's length:

1. Wide — the foul: bowler's slide crosses the line.
2. Medium (chained ×2) — reaction at the scorers' table, rises, draws the gun.
3. Close-up — gun raised, shouting.
4. Two-shot — calming/defusing gesture.
5. Close-up — bewildered bystander reaction.
6. Wide — standoff resolution, gun still raised, room frozen.
7. Two-shot — the Dude and Donny react to a mention of the cops.
8. Close-up — Walter still aiming, insisting.
9. Medium (chained ×2) — Smokey relents, marks the scoresheet.
10. Medium — Walter lowers the gun, tucks it away.
11. Two-shot — the Dude exasperated, Donny still lost.
12. Wide — closing shot, alley life resuming, mirrors the opening framing.

Character identity carried by fixed physical descriptions repeated in every prompt
(mustache/build/wardrobe per character), since prompt-only identity bleeds badly with
four characters in rotation (research 0023 finding).

## Budget

Original 6 shots: 7 clips (6 shots + 1 chained continuation) × ~$0.05–0.20/clip ≈
**$0.35–$1.40**. Extension (shots 7-12): 7 more clips (6 shots + 1 chained continuation),
same per-clip range.

## Status

- [x] Shot list / authored manifest written (`scene.json`, this README).
- [x] Budget confirmed by Will (in conversation).
- [x] Generated and reviewed. Output: `over-the-line.mp4` (70.7s, 14 clips, ~$1.20 total
      actual spend at fal.ai's per-clip rate across both passes).
- [x] Extended to double length (shots 7-12) after review of the original 6-shot cut.

## Review notes (retry log)

Verification frames extracted per clip are in `frames/` (`ffmpeg -ss 2 -frames:v 1`).

- Shots 1, 2 (+chained draw), 5, 6 were solid on the first pass — good wardrobe/identity
  consistency, correct action.
- Shot 3 (Walter close-up, gun raised) **failed** on the first pass: rendered an unrelated
  shirtless man, no gun, no wardrobe. Root-caused to body/sweat-focused phrasing pulling the
  model off-identity — rewrote to lead with wardrobe + weapon before any anatomical
  description. Fixed on retry (new seed).
- Shot 4 (Dude calming gesture) **failed** on the first pass: rendered the Dude grabbing at a
  rifle instead of a placating gesture, likely triggered by "the gun" reference and a
  two-character composition. Rewrote as a single-character medium shot with no weapon
  reference. Fixed on retry (new seed).
- Both retries only regenerated the two weak clips (`generate.py` skips existing clips in
  `work/`) — cheap to iterate on individual shots without re-rendering the whole scene.

Net: 2 of 7 clips needed a retry, both on close/medium shots where wardrobe or weapon
phrasing was ambiguous — consistent with research 0023's identity-bleed warning, though here
it was prompt phrasing rather than multi-character crowding that caused the miss.

The extension (shots 7-12, added later) rendered cleanly on the first pass — all 7 new clips
usable with no retries needed.

## Regenerating

```bash
cd lossy
.venv/bin/python3 .gent/skills/generate-scene/generate.py \
  scenes/big-lebowski-over-the-line/scene.json \
  --env .env --workdir scenes/big-lebowski-over-the-line/work
```
`work/` (raw per-clip mp4s) is gitignored — only the stitched `over-the-line.mp4` and
verification `frames/` are committed. Delete a specific `work/cNN.mp4` and rerun to retry
just that shot.

## LTX-2 comparison

Ran the same 6-shot prompts through `RunPodLtx2Strategy` (self-hosted LTX-2.3 22B distilled
FP8 via RunPod ComfyUI, 1280x704) for a side-by-side quality comparison against Wan 2.2 5B.
This bypasses the lightweight `generate-scene` skill entirely — LTX-2 isn't wired into it, so
`generate_ltx2.py` in this directory drives the strategy class directly with a minimal
`{"description": {"action": prompt}}` entry (no character-identity map needed; each prompt is
already self-contained).

**Result: `over-the-line-ltx2.mp4` — all 12 shots, 14 clips, 63.3s.** Quality across the board is
clearly better than Wan/fal: sharper, denser background crowds, more filmic — the wide standoff
(shot 6), the two-shot at the scorers' table (shot 2), and Smokey marking the scoresheet
(shot 9) in particular read as close to production-quality coverage. Several shots have an
unwanted decorative border/vignette artifact, and one has a small logo-like watermark in a
corner (visible in `frames-ltx2/`) — not requested in any prompt, likely needs a
negative-prompt addition aimed at LTX-2 specifically (untried; cosmetic only, doesn't affect
the action/identity).

**One shot from the extension is a known miss**: shot 8 (Walter's second close-up, "insisting")
rendered a stray police/military-lineup look — wrong wardrobe (olive drab with what read as
rank insignia) and wrong setting (a wood-paneled room with an onlooker crowd in suits, not the
bowling alley), though Walter's face/mustache/aviators/gun stayed correct. Root cause looks like
stochastic sampling variance rather than a prompt problem — the same prompt style worked fine
for the near-identical shot 3 earlier in the scene. A retry with a new seed was attempted but
blocked by a RunPod capacity outage (see below); shipped as-is rather than drop the beat or keep
waiting. Retry later by deleting `work-ltx2/s07_p00.mp4` and rerunning `generate_ltx2.py` (after
bumping `scene.json`'s `seed` if the same outcome recurs).

Getting the last 2 shots (the foul, and Walter's rise-and-draw) took persistence: RunPod had
zero capacity across all 5 fallback GPU types on two separate attempts, each spanning several
minutes with backoff, before a third attempt found an L40 free and completed both remaining
shots cleanly (including the chained rise-and-draw continuation). Ran via `/loop` overnight-
style polling — check for capacity, and if the run itself finds a free GPU it just generates
directly rather than needing a separate "is it free yet" check.

**Cost and reliability, for the record:**
- First (successful, partial) run: 4 clips on an RTX 6000 Ada Generation pod ($0.74/hr),
  ~$0.39 total, clean pod termination.
- Second run: capacity contention forced 3 retries before landing an L40 pod ($0.69/hr); the
  first two clip attempts on that pod failed with a JSON-parse / "404 no body" error — looks
  like a readiness race where the ComfyUI health check passes just before the server can
  actually accept a workflow submission. **The pod's own cleanup then failed** ("pod not found
  to terminate") while a differently-IDed pod was still running and billing — caught this by
  checking `runpod.get_pods()` directly and terminated it manually (twice; a second orphaned
  pod turned up when checking again). Total idle exposure was on the order of minutes, not the
  10+ hours the 0023 rehearsal flagged as its worst case, but it's the same class of failure.
- Added to `generate_ltx2.py` (not to the shared `strategies_video.py`/`runpod_pod.py`, to
  keep this a scoped comparison rather than a pipeline change): a one-retry-with-delay wrapper
  around the first clip generation (works around the readiness race), and a `sweep_leftover_pods()`
  safety net that lists and terminates any pod named `lossy-comfyui` on exit, regardless of
  whether the session's own tracked pod ID resolves. Confirmed 0 pods running after every run
  in this session.
- Third and fourth follow-up attempts: two more hit "no capacity on any of 5 GPU types" (genuine
  outage, nothing billed); the fifth attempt found an L40 pod, and its own cleanup hit the same
  "pod not found to terminate" bug as before — confirmed via `runpod.get_pods()` that the pod
  was in fact gone (0 running) despite the warning, so no actual leak that time, but the
  warning itself is not a reliable signal either way and always needs a direct API check.

**Takeaway:** LTX-2 output quality is a real step up over Wan 2.2 5B for this kind of shot, but
the self-hosted RunPod path is materially less reliable end-to-end (capacity contention, a
readiness race on fresh pods, and a pod-cleanup bug that needs a human or agent watching for
"pod not found to terminate" and checking the RunPod dashboard directly — that warning fired
twice across this session, once for a genuinely leaked pod and once for a pod that had actually
already terminated). Anyone rerunning `generate_ltx2.py` should watch for it and always verify
against `runpod.get_pods()` rather than trusting the log line either way. The underlying
`runpod_pod.py` cleanup bug (pod ID mismatch after `with_setup_retry` re-provisions) is worth
fixing at the pipeline level separately rather than working around per-script indefinitely.

To regenerate from scratch or after deleting a clip:
```bash
cd lossy
.venv/bin/python3 scenes/big-lebowski-over-the-line/generate_ltx2.py
```
It resumes from whatever's already in `work-ltx2/` (also gitignored) — delete a specific
`s{NN}_p{NN}.mp4` and rerun to retry just that shot/part.

## Audio pass

Both silent renders got an ambience + dialogue pass via `add_audio.py`:

- **Ambience**: one 10s MMAudio bed (`fal-ai/mmaudio-v2/text-to-audio`, prompt: distant alley
  noise, pins, crowd murmur, pinsetter hum) looped at low volume under the whole scene — same
  approach the `generate-scene` skill's own `audio` field uses.
- **Dialogue**: 8 short lines via ElevenLabs Turbo v2.5 TTS (`fal-ai/elevenlabs/tts/turbo-v2.5`),
  cast from `voice_casting.py`'s curated voice bank — Walter: Callum (intense/gritty), the Dude:
  George (calm/gravelly), Smokey: Roger (default/measured). Donny stays silent, matching his
  "bewildered" characterization rather than putting words to it. Lines are muxed in at each
  clip's start offset (+0.3s), computed per video since Wan/fal (~5.05s/clip) and LTX-2
  (~4.52s/clip) run different lengths — **not lip-synced**, since neither video was generated
  with mouth movement matched to these specific words. This is a soundtrack pass, the same
  tradeoff the skill documents for dialogue scenes.

Outputs: `over-the-line-with-audio.mp4` (Wan/fal) and `over-the-line-ltx2-with-audio.mp4`
(LTX-2). Regenerate with:
```bash
cd lossy
.venv/bin/python3 scenes/big-lebowski-over-the-line/add_audio.py
```
Raw audio assets live in `audio/` (committed — tiny, and skipped on rerun if already present).
Edit `DIALOGUE`/`AMBIENCE_PROMPT` in `add_audio.py` and delete the relevant file in `audio/` to
regenerate just that line or the ambience bed.
