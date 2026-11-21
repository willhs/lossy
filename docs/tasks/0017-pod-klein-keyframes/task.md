---
status: in-progress
steady_task: 4cd29e30-5bea-400c-b68e-df111d1683dc
created: 2026-11-22
---

# 0017: self-hosted keyframes (FLUX.2 klein on the LTX RunPod pod)

Vision alignment: Have an online presence (goal: complete the Star Wars IV reconstruction and publish it).

## Why

Keyframe-first (research 0029, "SW4 paid check, finish") passed the face rule on Star Wars (+0.15 face
cosine). On fal it adds about $15-19 for the ~795 keyframe shots, which pushes the 30 Nov SW4 full run over
the $50-60 ceiling. klein 4B on the A6000 we already rent should cost under $1 for the whole film.

## What

1. A ComfyUI API graph for FLUX.2 klein 4B (text-to-image for sheets; reference edit for keyframes), with
   node input names checked against ComfyUI v0.39.2 (the version the pod upgrades to).
2. `PodKleinGenerator` in `keyframes.py`, a second `KeyframeGenerator` beside `FalKleinGenerator`, with the
   klein weights pinned by sha256.
3. `RunPodLtx25KeyframeStrategy` makes sheets and keyframes after pod boot and before the first video, then
   unloads the models before LTX loads. `--keyframe-generator fal|pod` picks the editor; fal stays available.
   `pod` becomes the default only if the paid validation passes.
4. Paid validation on a pod (Will approved ~$0.30, hard cap $0.50 total) on SW4 window 684-710.
5. Results into research 0029; SW4 total into research 0030's cost table.

## Success criteria

- [ ] `uv run python -m pytest` passes; new tests cover the graph, the generator and the strategy's pod-first
      keyframe pass with the pod mocked.
- [ ] `decode.py --strategy runpod-ltx25-keyframe --keyframe-generator pod` makes keyframes on the pod
      before the first clip, frees models, then renders a clip (proved on a real pod).
- [ ] Pod keyframe face cosine to the sheet is within 0.05 of fal klein's on the same metric, no
      wardrobe/sheet leak by eye; per-image time recorded; contact sheet of fal vs pod keyframes.
- [ ] Spend <= $0.50; no pod left running.

## Constraints / won't do

- klein only, not Qwen. No change to job 08db840b's schedule. Secrets via the proj-lossy vault.
