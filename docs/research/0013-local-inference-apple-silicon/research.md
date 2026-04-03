---
id: 0013-local-inference-apple-silicon
type: note
purpose: "Evaluate whether Apple Silicon Macs (24-48 GB unified memory) can run local LLM and video/audio generation models relevant to the lossy pipeline."
scope: ["research", "infrastructure", "decode", "video-generation", "audio-generation"]
tags: ["research", "apple-silicon", "local-inference", "gemma", "mlx", "hardware"]
related: ["research/0012-wan-model-variants/research.md", "research/0009-cost-breakdown/research.md"]
---

## Question

Can a 24 GB MacBook M5 Pro replace or supplement the current RunPod 24 GB GPU for local inference -- either for video/audio generation or for LLM tasks?

## Context

The pipeline currently runs video generation (Wan 2.1 T2V-1.3B) and audio generation (MMAudio) on RunPod with 24 GB NVIDIA GPUs (RTX 3090/4090 class). A 24 GB MacBook M5 Pro is now available as a potential local inference target.

---

## Video / Audio Generation: Not Viable

Apple Silicon cannot practically replace NVIDIA GPUs for the pipeline's generation workloads.

### Why

1. **Shared vs dedicated memory.** The M5 Pro's 24 GB is unified memory split between CPU, GPU, OS, and apps. A 24 GB NVIDIA card dedicates all VRAM to model inference.

2. **Compute throughput.** NVIDIA GPUs have massively more CUDA cores optimised for tensor math. Expect 10-30x slower inference for diffusion models on Apple Silicon vs equivalent NVIDIA hardware.

3. **Software ecosystem.** Wan 2.1/2.2 and MMAudio are built on PyTorch + CUDA. No optimised Apple Silicon ports exist. MPS backend support is incomplete for these architectures.

4. **Memory bandwidth.** M5 Pro ~273 GB/s vs RTX 4090 ~1 TB/s. For diffusion models where compute is the bottleneck, this compounds the throughput gap.

### Verdict

Stick with RunPod for video and audio generation. The Mac is well-suited as the dev/orchestration machine (encoding, FFmpeg stitching, pipeline control).

---

## LLM Inference: Strong Candidate

Apple Silicon is competitive for LLM inference thanks to unified memory (large models fit without VRAM limits) and good memory bandwidth (token generation is bandwidth-bound).

### Gemma 4 (Released 2026-04-02)

Google released the Gemma 4 family under Apache 2.0. Four variants:

| Model | Total Params | Active Params | Architecture | Context | Modalities |
|---|---|---|---|---|---|
| E2B | 5.1B | 2.3B | Dense | 128K | Text, Image, Audio |
| E4B | 8B | 4.5B | Dense | 128K | Text, Image, Audio |
| **26B A4B** | 25.2B | 3.8B | MoE (128 experts) | 256K | Text, Image |
| 31B | 30.7B | 30.7B | Dense | 256K | Text, Image |

The 26B A4B uses Mixture of Experts: 26B total parameters spread across 128 expert sub-networks, but only ~4B activate per token. This gives big-model quality at small-model speed and memory cost.

### What Fits on 24 GB

| Model | Quantisation | Weight Size | Headroom for KV Cache | Viable? |
|---|---|---|---|---|
| E2B | BF16 | ~10 GB | ~14 GB | Easily |
| E4B | BF16 | ~16 GB | ~8 GB | Yes |
| E4B | Q4_K_M | ~5 GB | ~19 GB | Easily |
| **26B A4B** | **Q4_K_M** | **~17 GB** | **~7 GB** | **Yes -- sweet spot** |
| 26B A4B | IQ4_XS | ~13 GB | ~11 GB | Yes, more context headroom |
| 31B | Q4_K_M | ~18 GB | ~6 GB | Tight, short context only |
| 31B | Q3_K_M | ~15 GB | ~9 GB | Workable with quality loss |

48 GB would comfortably fit the 26B A4B at Q8 (~28 GB) or the 31B at Q4_K_M with generous context.

### Estimated Speed on M5 Pro (24 GB)

Based on comparable MoE models (Qwen3 30B-A3B) on M4 Pro hardware:

| Model | Quant | Backend | Est. tok/s |
|---|---|---|---|
| 26B A4B | Q4 | MLX | 70-100 |
| 26B A4B | Q4 | Ollama (llama.cpp) | 35-50 |
| 31B | Q4 | MLX | 15-25 |
| 31B | Q3 | Ollama | 8-15 |

MLX runs ~2-3x faster than llama.cpp on Apple Silicon for MoE models. TurboQuant (KV cache quantisation in mlx-vlm) further extends usable context length.

### Software Support

- **MLX**: Day-one support via `mlx-vlm`. Pre-converted weights on `mlx-community` HuggingFace org.
- **Ollama**: `ollama run gemma4:26b` works out of the box.
- **LM Studio**: Immediate support.

---

## Gemma 4 Quality Benchmarks

### vs Open-Source Models

| Benchmark | Gemma 4 31B | Gemma 4 26B A4B | Llama 4 Scout (109B/17B active) | Qwen 3.5 27B | Phi-4 14B |
|---|---|---|---|---|---|
| MMLU Pro | **85.2%** | 82.6% | 74.3% | ~85.5% | 71.4% |
| GPQA Diamond | **84.3%** | 82.3% | 57.2% | 85.8% | 57.5% |
| AIME 2026 | **89.2%** | 88.3% | -- | -- | -- |
| LiveCodeBench v6 | **80.0%** | 77.1% | -- | -- | -- |
| Codeforces ELO | **2150** | 1718 | -- | -- | -- |

The 26B A4B with only 3.8B active params beats Llama 4 Scout (17B active, 109B total) on both MMLU Pro and GPQA Diamond.

### vs Closed-Source Models (GPQA Diamond)

| Model | GPQA Diamond | Type |
|---|---|---|
| Gemini 3.1 Pro | 94.3% | Closed |
| GPT-5.2 Pro | 93.2% | Closed |
| Claude Opus 4.6 | 91.3% | Closed |
| Gemini 3 Flash | 90.4% | Closed |
| Claude Sonnet 4.6 | 89.9% | Closed |
| Qwen 3.5 397B | 88.4% | Open |
| **Gemma 4 31B** | **84.3%** | **Open** |
| ChatGPT-4o Latest | 84.0% | Closed |
| **Gemma 4 26B A4B** | **82.3%** | **Open** |
| Gemini 2.5 Pro | 83.0% | Closed |
| Claude Sonnet 4 | 75.4% | Closed |

Gemma 4 31B is roughly GPT-4o tier. It trails current frontier reasoning models by 7-10 points.

### Software Engineering

Google did not report SWE-bench, HumanEval, or aider scores. Independent evals have not yet landed (model is 1 day old as of writing).

**What Google reports:**
- LiveCodeBench v6: 80.0% (31B) / 77.1% (26B A4B)
- Codeforces ELO: 2150 (31B) / 1718 (26B A4B)

**Frontier closed-source SWE benchmarks (for reference):**

| Model | SWE-bench Verified | Aider Polyglot |
|---|---|---|
| GPT-5 | -- | 88.0% |
| Claude Opus 4.6 | 80.8% | -- |
| Gemini 3.1 Pro | 80.6% | -- |
| Claude Sonnet 4.6 | 79.6% | -- |
| Gemini 2.5 Pro | -- | 83.1% |
| o3 | -- | 81.3% |
| Claude Opus 4 | -- | 72.0% |

Gemma 4 is likely competitive with older closed-source models (GPT-4o era) for coding but a clear step below current frontier for real-world SWE tasks.

**Caveat:** Google reports on AIME 2026 and LiveCodeBench v6 while competitors often use older benchmark versions. Numbers are not perfectly comparable across versions.

---

## Recommendation

| Use Case | Recommendation |
|---|---|
| Video generation (Wan T2V) | RunPod -- Apple Silicon not viable |
| Audio generation (MMAudio) | RunPod -- Apple Silicon not viable |
| Local LLM (general) | **Gemma 4 26B A4B Q4 via MLX** -- best quality/speed on 24 GB |
| Local LLM (coding) | Gemma 4 26B A4B is GPT-4o tier; use API models for frontier SWE work |
| Encoding (Gemini API) | Mac works fine -- API calls only |
| Stitching (FFmpeg) | Mac works fine -- CPU-bound |

The M5 Pro 24 GB is an excellent local LLM machine but cannot replace cloud GPUs for the pipeline's diffusion-based generation workloads.
