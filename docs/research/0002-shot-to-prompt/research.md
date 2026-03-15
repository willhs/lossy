---
id: 0002-shot-to-prompt
type: note
purpose: "Research options for extracting descriptive text prompts from film shots, covering vision APIs, local models, frame sampling, metadata sources, and motion description."
scope: ["research", "encoder"]
non_goals: []
tags: ["research", "vision-models", "video-captioning", "prompts"]
related: ["research/0001-shot-detection/research.md", "design/adr/003-pyscenedetect-for-shot-detection.md"]
---

## Problem

Given ~2,000-3,000 shots detected from a film, generate a text prompt for each shot that accurately describes: visual content, characters, action/motion, camera movement, lighting, composition, and mood. These prompts will be used in the decode stage to regenerate shots via video generation models.

A single keyframe is insufficient -- motion, camera work, and temporal dynamics are core to what makes a shot meaningful.

## Vision APIs (Commercial)

### Google Gemini (Recommended)

Gemini is the only major commercial API that accepts **native video input** via the File API. Video is tokenized at ~263 tokens/second at 1 FPS. This means you can upload a shot clip directly instead of extracting frames.

| Model | Input $/MTok | Output $/MTok | Batch 50% off? |
|---|---|---|---|
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | Yes |
| Gemini 2.5 Flash | $0.30 | $2.50 | Yes |
| Gemini 2.5 Pro (<=200k) | $1.25 | $10.00 | Yes |

Image token counts (2.5 models): 64 tokens (low res), 256 tokens (medium), ~2,048 tokens (high res with Pan & Scan).

**Free tier**: Flash-Lite gets 1,000 RPD, Flash gets 250 RPD, Pro gets 100 RPD. All 3,000 shots could be processed for free in 3-12 days depending on model.

### OpenAI

No native video input. Must extract frames as images.

| Model | Input $/MTok | Output $/MTok | Tokens per Image (low detail) |
|---|---|---|---|
| gpt-4.1-nano | $0.05 | $0.20 | ~209 (85 * 2.46x) |
| gpt-4.1-mini | $0.20 | $0.80 | ~138 (85 * 1.62x) |
| GPT-4o-mini | $0.15 | $0.60 | 2,833 (inflated, known quirk) |
| GPT-4o | $2.50 | $10.00 | 85 |
| o4-mini | $1.10 | $4.40 | 75 |

High detail: 85 base + 170 per 512x512 tile (1024x1024 = 765 tokens for GPT-4o/4.1).

Batch API available at 50% discount, runs async over 24 hours.

### Anthropic Claude

No native video input. Image tokens = `(width * height) / 750`.

| Model | Input $/MTok | Output $/MTok | Batch? |
|---|---|---|---|
| Claude Haiku 4.5 | $1.00 | $5.00 | Yes, 50% off |
| Claude Sonnet 4.6 | $3.00 | $15.00 | Yes, 50% off |

Strength: best at following detailed structured prompts, so output formatting may be most consistent. Weakness: most expensive for bulk vision work.

### Cost Comparison (3,000 shots, ~200 output tokens each)

| Provider | Model | Estimated Total |
|---|---|---|
| Google | Gemini 2.5 Flash-Lite (free tier) | **$0.00** |
| Google | Gemini 2.5 Flash-Lite (batch) | **~$0.30** |
| Google | Gemini 2.5 Flash-Lite (paid) | **~$0.60** |
| OpenAI | gpt-4.1-nano (low detail) | **~$0.33** |
| OpenAI | gpt-4.1-mini (low detail) | **~$1.28** |
| Google | Gemini 2.5 Flash | **~$3.75** |
| OpenAI | GPT-4o-mini (low detail) | **~$2.18** |
| Anthropic | Claude Haiku 4.5 (batch) | **~$5.25** |
| Anthropic | Claude Sonnet 4.6 (batch) | **~$31.50** |

Output token cost dominates for most providers. Gemini Flash-Lite is the clear winner on price.

## Vision Models (Local / Open-Source)

### Top Contenders

**Qwen2.5-VL** (Alibaba) -- Strongest general-purpose local option.
- Sizes: 3B, 7B, 32B, 72B
- Native video support (dynamic FPS sampling, absolute time encoding, >1 hour video comprehension)
- Available in **ollama** (all sizes). 7B download: 6.0 GB. Requires ollama 0.7.0+.
- Ollama limitation: accepts images only (not raw video files). Video-as-native-input works via Python (vLLM, Transformers).
- VRAM: 7B Q4 = ~6-7 GB. 3B Q4 = ~3 GB. 32B Q4 = ~22 GB.
- Apple Silicon: 7B runs on 16GB+ Macs. 3B runs on anything.

**Tarsier2-7B** (ByteDance) -- Best open model specifically for video description.
- Outperforms GPT-4o by +2.8% F1 on DREAM-1K benchmark
- Human evaluations: +8.6% over GPT-4o, +24.9% over Gemini-1.5-Pro for detailed video descriptions
- Architecture: CLIP-ViT + LLM (7B). Min 16GB RAM, NVIDIA 3090 ideal.
- Available on HuggingFace. Not in ollama. Requires Transformers setup.

**MiniCPM-V 2.6 / 4.5** (OpenBMB) -- Best token efficiency for video.
- 8B params, ~5.5 GB download. Native video support.
- MiniCPM-V 4.5: 96x video token compression (6 frames at 448x448 compress to just 64 tokens)
- Available in **ollama** and **llama.cpp** (GGUF). Also vLLM.
- Outperforms GPT-4V and Claude 3.5 Sonnet on Video-MME benchmark.

**Gemma 3** (Google) -- Good balance of size and capability.
- 4B, 12B, 27B with vision. Gemma 3n supports video/audio natively.
- Available in **ollama** with QAT quantization.
- 12B Q4 = ~6.6 GB. Sweet spot for 16GB Macs.
- 128K context window, can process ~500 images or ~8 min video at 1fps.

**LLaVA-OneVision** -- The LLaVA family's latest.
- 0.5B, 7B, 72B. Accepts up to 32 frames per video. Each video frame pooled to 196 tokens.
- Not in ollama natively. Deploy via HuggingFace Transformers or vLLM.

**NVIDIA Describe Anything Model (DAM-3B-Video)** -- Localized captioning.
- Specify a region (point, box, mask) and get descriptions of just that region, including changes over time
- Uses SAM 2.1 to propagate masks through video from a single frame annotation
- NVIDIA Noncommercial License only

### Ollama Availability Summary

| Model | In Ollama | Download | Video via Ollama? |
|---|---|---|---|
| Qwen2.5-VL 3B | Yes | 3.2 GB | Images only |
| Qwen2.5-VL 7B | Yes | 6.0 GB | Images only |
| Gemma 3 12B | Yes | ~8.1 GB | Images only |
| MiniCPM-V 2.6 | Yes | ~5.5 GB | Images only |
| Llama 3.2 Vision 11B | Yes | 7.9 GB | Images only |

No ollama model currently accepts raw video files. You always extract frames and send as images. This is fine since we already have shots decomposed.

### VRAM / Apple Silicon Requirements

| Model | Q4 Quantized | Apple Silicon Viable? |
|---|---|---|
| Qwen2.5-VL 3B | ~3 GB | Yes, any Mac |
| Gemma 3 4B | ~2.5 GB | Yes, any Mac |
| MiniCPM-V 2.6 | ~5 GB | Yes, 16GB+ |
| Qwen2.5-VL 7B | ~6-7 GB | Yes, 16GB+ |
| Gemma 3 12B | ~6.6 GB | Yes, 16GB+ |
| LLaVA-OV 7B | ~6 GB | Yes, 16GB+ |
| Gemma 3 27B | ~17 GB | Yes, 32GB+ |
| Qwen2.5-VL 32B | ~22 GB | Yes, 48GB+ |

### Throughput and Time Estimates

On Apple Silicon (single request, no concurrency): ~5-15 tokens/sec for a 7B Q4 model. For a 150-token output, that's ~10-30 seconds per shot, or ~2-6 shots/minute.

**3,000 shots locally on a Mac: ~8-25 hours** (varies with model size and Mac specs).

On NVIDIA GPU (RTX 3090/4090, with concurrency): ~50-100 images/minute. **3,000 shots: ~2-4 hours.**

Electricity cost in either case: under $1.

### Quality Gap

Commercial models still lead on video understanding benchmarks. On Video-MME: Gemini 2.5 Pro ~84.8%, GPT-4o ~71.9%. Best open-source models trail by 15-25 points on benchmarks, but for descriptive captioning (vs QA), the gap is smaller. In human evaluations of video descriptions specifically, Tarsier2-7B beats GPT-4o.

## Frame Sampling Strategy

### How Many Frames Per Shot?

Research converges on these findings:

- **1 frame**: Captures static content only. Loses all temporal information. Only viable for establishing shots.
- **4 frames**: Good minimum. Captures start, two midpoints, end. Sufficient for short shots (<3s).
- **8 frames**: Meaningfully better for shots with action or camera movement. Research shows consistent gains from 4 to 8.
- **16 frames**: Diminishing returns for short shots (2-5s), but valuable for longer shots (8-10s+).
- **Beyond 16**: Accuracy peaks around 256 frames on VideoMME for long videos, but irrelevant for individual shots of 1-5 seconds. More frames can hurt -- noise degrades performance.

**Recommendation**: Scale by duration. 4 frames for shots under 3s, 8 frames for 3-8s, scale linearly for longer.

### Sampling Methods

1. **Uniform** (simplest, best performer on VideoMME): Take N evenly-spaced frames. Best default.
2. **First + middle + last**: 3 frames capturing start/end state and midpoint.
3. **Keyframe-based**: Extract frames where visual content changes significantly. PySceneDetect can help.
4. **Native video input** (Gemini, Qwen-VL via vLLM): Let the model handle sampling internally. Gemini processes at 1 FPS by default.

### Resolution

- **512px on longest side**: Practical sweet spot for scene description. OpenAI recommends ~512-720px.
- **256px**: Works for gross scene understanding but loses text, facial expressions, small objects.
- **768px+**: Only needed for reading on-screen text or identifying fine visual details.
- Lower resolution = fewer tokens = cheaper. An OpenAI "low detail" image (85 tokens) vs high detail (765+ tokens) is a 9x cost difference.
- Film frames at 720p or even 480p should provide enough visual information for shot description.

### Key Research

- Moments Lab benchmark (2025): uniform FPS sampling works best overall; accuracy peaks around 256 frames but with diminishing returns
- GenS (ACL 2025): adaptive sampling improves accuracy by 13.4 points vs uniform when using <=40 frames on long-form video
- F-16 (2025): 16 fps achieves SOTA on Video-MME among 7B models, competitive with GPT-4o
- Claude docs: images under 200px on any edge "may degrade performance"

## Supplementary Metadata Sources

### Subtitles (Highest Value, Easiest)

Extract embedded subtitles with ffmpeg:
```bash
ffprobe -v error -select_streams s -show_entries stream=index,stream_tags:s=language -of csv=p=0 input.mkv
ffmpeg -i input.mkv -map 0:s:0 output.srt
```

Alignment with shots: SRT timestamps overlap check against shot `[start_s, end_s]` intervals. Trivial, no fuzzy matching needed.

Format comparison for speaker identification:

| Format | Speaker Support |
|---|---|
| SRT | Convention only (`CHARACTER:` prefix) |
| ASS/SSA | Formal `Name` field per dialogue line + per-character styles |
| VTT | Voice tags: `<v Sarah>Hello</v>` |

SDH (Subtitles for Deaf/Hard of Hearing) also include action annotations like `[music playing]`, `[door slams]`, `[tense music]` which can enrich mood and description fields.

**Fallback**: OpenSubtitles REST API for films without embedded subs. Search by IMDB/TMDB ID or file hash.

### Film Scripts

**IMSDb** (imsdb.com): Free screenplay database. Star Wars scripts available. HTML format -- use [imsdb_parse](https://github.com/alex-raw/imsdb_parse) to classify into heading/character/dialogue/action.

**Fountain format**: Standard for machine-parseable screenplays. [screenplay-tools](https://github.com/wildwinter/screenplay-tools) (Python, JS, C++, C#) parses Fountain and Final Draft into structured objects.

Screenplay element types useful for prompts:
- **Scene headings**: `INT. DEATH STAR - CONFERENCE ROOM` -- location + time
- **Action lines**: Scene descriptions ("The room is dark and oppressive") -- richest source for visual description
- **Transitions**: `CUT TO:`, `FADE IN:` -- editorial intent

**Camera directions** in scripts: Rare in modern spec scripts (considered bad form). More common in shooting scripts and writer-director scripts. Always CAPITALIZED: `CLOSE ON`, `WIDE SHOT`, `PAN TO`, `ANGLE ON`, `TRACKING SHOT`.

**Alignment challenge**: Scripts describe scenes (multiple shots), not individual shots. Best approach: use dialogue text as anchors (match screenplay dialogue to subtitle timestamps, then interpolate surrounding action descriptions).

### Audio Analysis

**WhisperX** -- transcription + word-level timestamps + speaker diarization:
- Combines Whisper with wav2vec2 alignment and pyannote.audio diarization
- ~70x realtime with batched GPU inference. 90-minute film: ~1-2 minutes.
- Output: segments with `start`, `end`, `text`, `speaker` (anonymous labels: SPEAKER_00)
- Use as fallback when subtitles are missing, or for speaker diarization that subtitles lack.

**YAMNet** -- audio classification into 521 categories (AudioSet ontology):
- Classifies per-frame (~0.48s): speech, music, silence, gunshot, explosion, laughter, rain, etc.
- Runs on CPU. TensorFlow Hub model.
- Useful for `mood` field: detecting tense music, silence, ambient sound, action sounds.

### Movie Databases (TMDB, OMDB)

Film-level metadata only (plot, genre, cast/crew, posters). No scene-level information. Useful for providing overall context in the vision model prompt ("This is Star Wars, a 1977 sci-fi film") but not shot-level enrichment.

## Motion Description

### Camera Motion via Optical Flow

Optical flow analysis can classify camera motion without any neural networks:

| Motion Type | Flow Signature |
|---|---|
| Pan (horizontal) | Uniform horizontal vectors across frame |
| Tilt (vertical) | Uniform vertical vectors across frame |
| Zoom in | Vectors diverging from center |
| Zoom out | Vectors converging toward center |
| Dolly/tracking | Pan-like but with parallax (near objects move faster) |
| Static | Near-zero magnitude |
| Handheld/shaky | High-frequency random vectors |

**Tool**: [antiboredom/camera-motion-detector](https://github.com/antiboredom/camera-motion-detector) -- uses OpenCV Farneback optical flow, produces per-frame CSV with motion angle, magnitude, and zoom factor. GPU-accelerated version available.

**Performance**: Farneback dense optical flow: ~8ms/frame on CPU. For 3,000 shots sampling every 5th frame: ~3.5 minutes on CPU.

**Limitation**: Cannot reliably distinguish dolly from pan (requires depth estimation, typically a neural network).

**Recommendation**: Run optical flow as a preprocessing step to label each shot with camera motion type. Include this label in the vision model prompt so it can focus on content rather than inferring camera movement.

### Character/Object Motion

Vision models handle this better than camera motion -- it's more visually obvious across frames. Providing 3+ frames with clear temporal ordering helps. Include "describe any movement or action occurring in this shot" in the prompt.

### Vision Model Camera Motion Capabilities

Commercial models (Gemini, GPT-4o) can describe obvious camera movements ("the camera pans left") when given multiple frames, but often miss subtleties or use imprecise terminology. Qwen3-VL is explicitly trained with temporal dynamics comprehension. General weakness: most VLMs are trained primarily on static image understanding.

## Related Work and Benchmarks

### UltraVideo Dataset

Most directly relevant reference. Generates structured captions with 10 semantic tag categories: Brief Description, Detailed Description, Background, Theme, Style, **Shot Type**, **Camera Movement**, **Lighting**, **Video Atmosphere**, and more. Average caption length: 824 words per video. Their pipeline is essentially what we want to build.

### Video-MME (CVPR 2025)

First comprehensive benchmark for multi-modal LLMs in video analysis. Gemini 2.5 Pro leads (~84.8%), with open-source models trailing.

### MotionBench (CVPR 2025)

Fine-grained video motion understanding benchmark. Specifically evaluates how well VLMs understand motion. Relevant for model selection.

### MPII Movie Description Corpus

94 Hollywood movies, 68K clips with paired sentence descriptions. Originally sourced from audio descriptions for the visually impaired. Demonstrates that movie shots can be described in ~1 sentence each.

### Dense Video Captioning

- **ActivityNet Dense Captioning**: Stanford benchmark for temporally-grounded descriptions
- **TA-Prompting** (Jan 2026): Enhances Video LLMs for dense captioning via temporal anchors
- **CM2 DVC** (CVPR 2024): Cross-modal memory retrieval for context-aware dense captioning

## Recommended Approaches

### Cheapest Pipeline (~$0-2)

1. Extract 3-6 frames per shot at 512-720px (ffmpeg, uniform sampling)
2. Extract subtitles (ffmpeg), align to shots by timestamp overlap
3. Run optical flow camera motion detection (antiboredom tool or OpenCV Farneback)
4. Pull screenplay from IMSDb if available, roughly align via dialogue anchors
5. Send to **Gemini 2.5 Flash-Lite** (batch API or free tier)
6. Prompt includes: frames, camera motion label, aligned dialogue, scene context
7. Request structured output: shot type, camera movement, subjects, action, lighting, color palette, mood

Cost: $0 (free tier, 3 days) to ~$0.60 (paid). For ~500 output tokens per shot instead of 200: ~$1.20.

### Best Quality-to-Cost Pipeline (~$2-5)

Same preprocessing, but use **Gemini 2.5 Flash** with native video upload (send the actual shot clip) or **gpt-4.1-mini** with batch API. Optional second pass with Claude Haiku for structured formatting if output is inconsistent.

### Fully Local Pipeline ($0, time cost)

Same preprocessing, then run **Qwen2.5-VL 7B** via ollama (feed extracted frames as images). Or Tarsier2-7B via Transformers for highest description quality. 8-25 hours on a Mac, 2-4 hours on an NVIDIA GPU.

### Hybrid: Local Preprocessing + Cheap API

1. Subtitles, optical flow, color palette extraction -- all local, all cheap
2. Assemble metadata context per shot
3. Send frames + metadata to Gemini Flash-Lite
4. Post-process with local LLM for formatting consistency

This gives the best ratio of description quality to cost.

## Open Questions for Next Research Phase

- What structured output format should prompts use? (JSON fields matching UltraVideo categories? Free text? Intermediate format?)
- Should we fine-tune a local model on UltraVideo or MPII movie descriptions for better film-specific output?
- How to handle the ~4 shots over 30s (credits, crawl) -- exclude, truncate, or special-case?
- Is there value in two-pass description (coarse local pass for filtering, then API pass for final prompts)?
- How much does including screenplay context actually improve description quality vs frames alone?

## Sources

### Vision APIs
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini Token Counting](https://ai.google.dev/gemini-api/docs/tokens)
- [Gemini Media Resolution](https://ai.google.dev/gemini-api/docs/media-resolution)
- [OpenAI Images and Vision](https://developers.openai.com/api/docs/guides/images-vision/)
- [OpenAI Pricing](https://openai.com/api/pricing/)
- [Claude Vision Docs](https://platform.claude.com/docs/en/build-with-claude/vision)
- [Claude Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [OpenAI Cookbook: Video Understanding](https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding)

### Local Models
- [Qwen2.5-VL on Ollama](https://ollama.com/library/qwen2.5vl)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Tarsier2 GitHub](https://github.com/bytedance/tarsier)
- [Tarsier2-Recap-7B on HuggingFace](https://huggingface.co/omni-research/Tarsier2-Recap-7b)
- [MiniCPM-V GitHub](https://github.com/OpenBMB/MiniCPM-o)
- [LLaVA-OneVision GitHub](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5)
- [NVIDIA DAM GitHub](https://github.com/NVlabs/describe-anything)
- [Gemma 3 Overview](https://ai.google.dev/gemma/docs/core)
- [Clarifai VLM Benchmark](https://www.clarifai.com/blog/benchmarking-best-open-source-vision-language-models)

### Frame Sampling
- [Frame Sampling Impact on Small VLMs (Moments Lab)](https://research.momentslab.com/blog-posts/frame-sampling-vlm)
- [GenS: Generative Frame Sampler (ACL 2025)](https://arxiv.org/html/2503.09146v1)
- [Adaptive Keyframe Sampling (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.pdf)
- [Less Is More: Picking Informative Frames](https://arxiv.org/abs/1803.01457)

### Metadata and Motion
- [antiboredom/camera-motion-detector](https://github.com/antiboredom/camera-motion-detector)
- [Camera Motion Estimation via Optical Flow](https://medium.com/@ikunyankin/camera-motion-estimation-using-optical-flow-ce441d7ffec)
- [WhisperX](https://github.com/m-bain/whisperX)
- [YAMNet (TensorFlow Hub)](https://www.tensorflow.org/hub/tutorials/yamnet)
- [ffsubsync](https://github.com/smacke/ffsubsync)
- [screenplay-tools](https://github.com/wildwinter/screenplay-tools)
- [imsdb_parse](https://github.com/alex-raw/imsdb_parse)
- [Fountain Syntax](https://fountain.io/syntax/)
- [OpenSubtitles API](https://opensubtitles.stoplight.io/docs/opensubtitles-api/e3750fd63a100-getting-started)

### Datasets and Benchmarks
- [UltraVideo Dataset](https://huggingface.co/datasets/APRIL-AIGC/UltraVideo)
- [Video-MME (CVPR 2025)](https://arxiv.org/abs/2405.21075)
- [MotionBench (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Hong_MotionBench_Benchmarking_and_Improving_Fine-grained_Video_Motion_Understanding_for_Vision_CVPR_2025_paper.pdf)
- [Dense Video Captioning Survey](https://www.mdpi.com/2076-3417/15/9/4990)
- [MPII Movie Description Corpus](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/mpii-movie-description-dataset)
