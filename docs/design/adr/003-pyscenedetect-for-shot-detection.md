---
id: adr-003-pyscenedetect
type: decision
purpose: "Record the decision to use PySceneDetect with AdaptiveDetector for shot boundary detection in the encode stage."
scope: ["design", "encoder"]
non_goals: []
tags: ["adr", "encoder", "video-analysis"]
related: ["design/architecture.md", "design/adr/001-scene-manifest-as-intermediate-format.md"]
---

# Context

The encoder needs to split a source film into individual shots (not narrative scenes — individual camera cuts). We evaluated several approaches: raw FFmpeg scene filter, PySceneDetect (Python library wrapping OpenCV), and TransNetV2 (deep learning model).

We tested PySceneDetect against Star Wars Episode IV (632MB, ~2h, 179k frames) to validate accuracy and performance.

# Decision

Use **PySceneDetect** (v0.6.7) with `AdaptiveDetector` for shot boundary detection.

**Test results on Star Wars IV:**
- 2,070 shots detected in 50s (~4,400 frames/sec)
- Average shot duration: 3.62s (published film-studies figure is ~4.2s ASL)
- 94% of shots under 10s, median 2.3s
- Known false positives: full-screen laser flashes, opening credit roll text (crawl detected as multiple shots). Acceptable for v1.

**Why AdaptiveDetector over ContentDetector:** AdaptiveDetector uses a rolling average of HSL frame differences, making it more robust to gradual lighting changes and camera motion — both common in narrative film. ContentDetector triggers more false positives on action sequences.

# Consequences (Positive/Negative)

**Positive:**
- Fast — processes a full feature film in under a minute on a laptop
- Zero-config defaults work well for narrative film
- Built-in keyframe export (`save_images`) and video splitting (`split_video_ffmpeg`) integrate directly with the pipeline
- Pure Python with OpenCV backend — no GPU or model weights required
- Structured output (timecodes, frame numbers) maps cleanly to the scene manifest format

**Negative:**
- False positives on visual effects (flashes, explosions) and text overlays — some shots will be spurious
- Does not detect gradual transitions (dissolves, wipes) as reliably as TransNetV2
- Single-frame keyframe extraction misses motion context in longer shots (16% of shots are 5-15s)
- Shot count skews slightly high vs. film-studies ground truth (~2,070 vs ~2,228 reported, but the discrepancy likely reflects different cuts/versions)

# Alternatives Considered

- **FFmpeg `select='gt(scene,X)'`**: Simpler but less tuneable — single global threshold, no adaptive windowing, no structured output. Would need custom scripting for everything PySceneDetect provides out of the box.
- **TransNetV2**: Higher accuracy on transitions (F1 93-96% on benchmarks) but requires PyTorch/TensorFlow, model weights, and GPU for reasonable speed. Overkill for v1 given PySceneDetect's results are good enough.
- **PyCinemetrics**: Academic GUI tool bundling TransNetV2 + color/object analysis. Too heavyweight and not designed for pipeline integration.

# Links

- [PySceneDetect docs](https://www.scenedetect.com/docs/latest/)
- [PySceneDetect GitHub](https://github.com/Breakthrough/PySceneDetect)
- [TransNetV2](https://github.com/soCzech/TransNetV2) — potential upgrade path if detection quality becomes a bottleneck
- [Architecture overview](../architecture.md)
