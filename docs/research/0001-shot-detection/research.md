---
id: 0001-shot-detection
type: note
purpose: "Catalogue tools, libraries, datasets, and techniques for extracting shot-level information from video files."
scope: ["research", "encoder"]
non_goals: []
tags: ["research", "video-analysis", "shot-detection"]
related: ["design/adr/003-pyscenedetect-for-shot-detection.md"]
---

## Shot Boundary Detection Libraries

### PySceneDetect

- **Repo**: https://github.com/Breakthrough/PySceneDetect
- **License**: BSD-3-Clause
- **Version tested**: 0.6.7 (August 2025)
- **Install**: `pip install scenedetect[opencv]`
- **Backends**: OpenCV (default), PyAV, MoviePy

Five detection algorithms:

| Detector | Method | Best for |
|---|---|---|
| `AdaptiveDetector` | Rolling average of HSL differences | Narrative film with variable lighting/pace |
| `ContentDetector` | Weighted HSV pixel changes | Fast hard cuts |
| `ThresholdDetector` | Average pixel intensity (RGB) | Fade in/out transitions |
| `HistogramDetector` | Y-channel histogram differences (YUV) | Fast cuts (alternative to Content) |
| `HashDetector` | Perceptual hashing similarity | General purpose |

Built-in pipeline features: keyframe image export (`save_images`), CSV scene list (`write_scene_list`), video splitting via FFmpeg (`split_video_ffmpeg`), per-frame stats caching (`StatsManager`) for threshold tuning.

### TransNetV2

- **Repo**: https://github.com/soCzech/TransNetV2
- **Paper**: https://arxiv.org/pdf/2008.04838
- **PyPI**: `transnetv2-pytorch`
- **Architecture**: 3D separable convolutions (spatial 1x3x3 + temporal 3x1x1)

Benchmark F1 scores:

| Dataset | F1 |
|---|---|
| ClipShots | 77.9 |
| BBC Planet Earth | 96.2 |
| RAI | 93.9 |

State-of-the-art on all three benchmark datasets. Both TensorFlow and PyTorch inference available. Pre-trained weights included. Struggles with flickering, fast motion, and occlusions. Requires GPU for reasonable throughput on feature-length films.

### PyCinemetrics

- **Repo**: https://github.com/CBD-Lab/pyCinemetrics (v1), https://github.com/CBD-Lab/pyCinemetricsV2 (v2, 2025)
- Academic GUI tool bundling: TransNetV2 shot detection, K-Means color extraction, VGG19 object recognition, OpenPose shot scale judgment (close-up/medium/wide), EasyOCR subtitle extraction
- V2 adds interactive correction for detection errors and newer transformer-based models
- Not designed for pipeline integration — useful as a reference for what a full film analysis suite looks like

### FFmpeg Scene Filter

Raw FFmpeg can detect scene changes without any Python:

```bash
# Extract one frame per detected scene change (threshold 0-1, 0.3-0.5 typical)
ffmpeg -i movie.mp4 -vf "select='gt(scene,0.4)'" -vsync vfr scene_%04d.jpg

# I-frame (codec keyframe) extraction — less useful, placed by encoder not by content
ffmpeg -i movie.mp4 -vf "select='eq(pict_type,I)'" -vsync vfr keyframe_%04d.jpg
```

Single global threshold, no adaptive windowing, no structured output. Useful as a quick sanity check but not as a primary detection tool.

## Frame Description Pipeline

The standard approach for generating text descriptions of video shots:

1. **Detect shots** — PySceneDetect / TransNetV2
2. **Extract keyframes** — `save_images` (1-3 frames per shot: start/middle/end)
3. **Encode as base64** — for API consumption
4. **Send to vision model** — prompt for cinematic details

Vision model options:

| Model | Cost/image (approx) | Notes |
|---|---|---|
| Claude Haiku | ~$0.001 | Cheapest, may miss subtle details |
| Claude Sonnet | ~$0.003 | Good balance of quality and cost |
| GPT-4o-mini | ~$0.002 | 1M token context, can batch many frames |
| GPT-4o | ~$0.005 | Higher quality descriptions |
| Local (LLaVA, InternVL) | Free | Lower quality, useful for bulk/dev |

For ~2,000 shots at 1 frame each, total description cost is roughly $2-10 depending on model.

Prompt should ask for: shot type (wide/medium/close-up/extreme close-up), camera angle and movement, subjects and action, lighting quality and direction, color palette, composition, mood. These details make the descriptions usable as video generation prompts in the decode stage.

## Star Wars Specific Resources

### Cinemetrics Database

- **URL**: https://cinemetrics.uchicago.edu/
- University of Chicago film studies project
- Star Wars Episode IV: average shot length (ASL) of 4.2 seconds
- Contains shot-by-shot timing data submitted by film scholars

### Original Trilogy Shot List Spreadsheet

- **URL**: https://originaltrilogy.com/topic/Shot-List-Spreadsheet-v0-6-05-6-films-Multiple-SW-Audio-Mix-Changes-Added-Recently/id/13403
- Fan-maintained spreadsheet (v0.6.05) covering all 6 original/prequel films
- Tracks individual shots and documents changes between versions (theatrical, Special Edition, etc.)
- Requires forum login to access
- Reports ~2,228 shots for Episode IV

### FILM-SHOT-COUNTER Database

- **URL**: https://www.academia.edu/41152064/FILM_SHOT_COUNTER
- Academic database of shot counts and average lengths for quantitative cinema analysis
- Updated February 2025

## Our Test Results (Star Wars Episode IV)

- **Source**: 632MB MP4, 1080p, ~2h runtime, 179,437 frames
- **Detector**: PySceneDetect `AdaptiveDetector` (default threshold 3.0)
- **Processing time**: 50.2 seconds (~4,400 frames/sec on Apple Silicon laptop)
- **Shots detected**: 2,070

Duration distribution:

| Range | Count | % |
|---|---|---|
| 0-1s | 194 | 9.4% |
| 1-2s | 658 | 31.8% |
| 2-3s | 438 | 21.2% |
| 3-5s | 408 | 19.7% |
| 5-8s | 200 | 9.7% |
| 8-15s | 132 | 6.4% |
| 15-30s | 36 | 1.7% |
| 30s+ | 4 | 0.2% |

Key percentiles: p50 = 2.34s, p75 = 4.00s, p90 = 7.17s, p95 = 10.01s

Known false positives: full-screen laser/explosion flashes detected as shot boundaries, opening crawl text segmented into multiple shots. Both are acceptable for v1 — the vision model descriptions will make these obvious in the manifest for manual cleanup if needed.

The 4 shots over 30s are: end credits (233s), and likely the opening crawl / long establishing shots. These outliers would need special handling or exclusion in the decode stage.
