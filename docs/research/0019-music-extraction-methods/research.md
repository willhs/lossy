---
id: "0019"
type: research
purpose: "Survey methods for extracting and reconstructing a film's music/score in the lossy round-trip. Documents what could be done — not a pipeline change."
scope: ["audio", "music", "research", "investigation"]
tags: [audio, music, source-separation, music-generation, musicgen, demucs, gemini, research]
related:
  - "research/0006-mmaudio-duration/research.md"
  - "research/0008-audio-captioning/research.md"
  - "research/0014-eval-audio-mmaudio-20260327/research.md"
  - "research/0015-eval-audio-runpod-mmaudio-20260328/research.md"
---

# Music Extraction Methods for the Lossy Round-Trip

## Context and Gap

The pipeline already detects music: YAMNet assigns shots to a `music` bucket (classes 24–32 + 132–276 in `encode.py:329-353`), and Gemini writes a single `sound` field that blends music, SFX, and ambience. At decode time, MMAudio (`strategies_audio.py`) treats that blended field as an SFX/ambience prompt — it is not a music generator. The current cost table in the article correctly states: *Music / score — not generated — $0*.

Three gaps need to be closed before a score can appear in the round-trip:

1. **Representation gap** — music is lumped into `sound` with SFX and ambience; there is no dedicated `music` field.
2. **Generation gap** — no music generator in the decode stage.
3. **Copyright constraint** — the original score (e.g. John Williams) cannot be reproduced. Any generated audio is "in the style of", which the article's "the score is simply gone" framing correctly acknowledges. This constraint is permanent regardless of method choice.

This document surveys four method families, estimates costs at **film scale (~7,480s, Star Wars IV)**, and recommends a path.

All costs and quality assessments below are **desk estimates** unless marked *measured*.

---

## Film-Scale Reference Numbers

| Quantity | Value | Notes |
|----------|-------|-------|
| Total film audio | ~7,480s | Star Wars IV, 125 min |
| Total shots (estimated) | ~1,750 | From encode runs |
| Music-bucket shots (estimated) | ~700–900 | ~45–50% of shots; desk estimate |
| Music audio at film scale | ~3,500–4,500s | Music-bucket shots × avg ~5s/shot |

---

## Method Family 1 — Source Separation (Extract a Music Stem)

This is the enabling step. Getting a clean music stem removes SFX/dialogue bleed before captioning and produces a faithful representation of what the original score actually sounded like.

### 1a. Demucs v4 `htdemucs`

**What it is:** Facebook Research's hybrid Transformer+U-Net separator. Four stems: `vocals`, `drums`, `bass`, `other`. The `other` stem captures everything that is not voice, drums, or bass — for a film score this means orchestral strings, brass, woodwinds, synths, and most SFX that survived the other stems. Imperfect for film (some SFX will bleed into `other`), but usable as a first pass.

A 6-stem variant (`htdemucs_6s`) adds `guitar` and `piano` — marginal benefit for orchestral, but frees piano from `other`.

**License:** MIT. Model weights downloadable from HuggingFace.

**Self-host on RunPod:**
- GPU: RTX 4090 ($0.34/hr). Demucs htdemucs processes ~4–6× real-time on a 4090.
- 7,480s audio ÷ 5 (estimate) = ~1,500s processing = ~25 min GPU time.
- Cost: ~$0.14 for full-film separation. *(Desk estimate — no benchmark run.)*
- Alternatively: runs on the same pod as Wan/MMAudio at near-zero marginal cost.

**Self-host on CPU:** ~20–30× slower than real-time. ~40–60 hours for full film. Not viable for batch.

**API:** No hosted Demucs API with known reliable SLA. Would need a custom RunPod endpoint.

**Quality:** Adequate for a music *description* input, not clean enough to be a hi-fi music stem in isolation. "other" picks up SFX bleed, room noise, and compression artifacts from the original soundtrack. Acceptable for the captioning step below.

### 1b. Film-Specific DME Separators (Bandit / DnR family)

**What it is:** A second generation of separators trained specifically on the Divide and Remaster (DnR) dataset — a corpus of professional dialogue/music/effects (DME) mixes. Models in this family produce three clean stems: `dialogue`, `music`, `effects`. The `music` stem is far cleaner than Demucs's `other` because the model was trained to discriminate between music and cinematic effects.

Key models:
- **Bandit** (ICASSP 2024): multi-mask, multi-band separator. ~+4 dB SDR over Demucs on DnR test set. Paper: arXiv 2309.02878. Weights on HuggingFace (`kwatcharapon/bandit`). MIT license.
- **Banquet** (follow-up work): improves on Bandit on the DnR benchmark. Available via the same HuggingFace ecosystem.
- **audio-separator** PyPI package: wraps Demucs, UVR, MDX-Net, and others behind a common CLI/API. Easy to swap models without code changes.

**License:** MIT (Bandit). DnR dataset is CC BY 4.0.

**Self-host on RunPod:**
- Similar compute envelope to Demucs (Transformer architecture, ~130M params for Bandit).
- Cost estimate: ~$0.10–$0.20 for full-film. *(Desk estimate.)*
- Bandit is the preferred model over htdemucs for this use case — the DME split is exactly the structure the pipeline needs.

**Quality:** Bandit consistently outperforms Demucs on film audio in published benchmarks. The `music` stem is clean enough for captioning; SFX/dialogue bleed is measurably lower. Orchestral transients and cymbals still leak slightly (a known weakness of all current separators).

**Recommended over Demucs for this pipeline** — the film-specific training is directly aligned with the task.

---

## Method Family 2 — Music Captioning (Describe the Stem)

Once a clean music stem exists, we need a natural-language description: mood, tempo, instrumentation, key/tonality, intensity. This feeds the `music` field in the encoded representation.

### 2a. LP-MusicCaps

**What it is:** LAION's music captioning model built on MusicCaps dataset. Generates free-text descriptions of music clips. ~100M params, runs on CPU or GPU.

**Limitations:**
- Trained on 10s clips; performance degrades on shorter/longer inputs.
- MusicCaps training set skews toward Western pop/rock/classical. Film orchestral is underrepresented.
- Quality ceiling is lower than Gemini on complex cinematic music.
- MIT license; weights on HuggingFace.

**Cost:** $0 (local). ~1–2s per clip on CPU, <0.5s on GPU.

**Assessment:** Usable as a cheap fallback. Would produce descriptions like "orchestral piece with strings and brass, tense mood" rather than the more specific "strings rise to a climax over a minor-key brass motif at ~120bpm" that MusicGen benefits from.

### 2b. Qwen2-Audio

**What it is:** Alibaba's multimodal audio-language model (7B params). Handles speech, music, and general audio understanding in a single model. Can be prompted to describe music in detail — instrumentation, mood, tempo, genre.

**License:** Apache 2.0 (commercial use allowed).

**Self-host:** Requires ~16GB VRAM (FP16). Fits on an A100 80GB pod alongside other work. Roughly 2–5s per clip for music description.

**Cost:** Marginal if sharing GPU pod, otherwise similar to Demucs cost envelope (~$0.20 desk estimate for all music-bucket shots).

**Assessment:** Better music vocabulary than LP-MusicCaps. Useful if Gemini API cost is a concern, but see 2c.

### 2c. Gemini Native Audio Input ← Preferred

**What it is:** Gemini 1.5/2.x natively accepts audio files as input. The current pipeline already calls Gemini for shot descriptions (`encode.py:535`) — it's the same API call but feeding it the separated music stem with a music-specific prompt: *"Describe this music in cinematic terms: mood, tempo (bpm estimate), key/tonality, instrumentation, and how it would function in a scene."*

This directly fixes the weakness identified in research 0008: YAMNet+Gemini hallucinates sound from visual context. With Gemini receiving the actual audio (not just YAMNet labels), the description is grounded in what the score actually sounds like.

**Cost (Flash):**
- Audio input: Gemini 2.0 Flash charges ~$0.01 per minute of audio (desk estimate from pricing page — *not measured at film scale*).
- For ~700 music-bucket shots × avg 5s = ~3,500s = ~58 min audio → ~$0.58.
- This replaces the existing YAMNet→Gemini chain for music shots, so the incremental cost vs. the current pipeline is small.

**Integration fit:** Very high — already calling Gemini, already managing audio in the encode step. Separating into a clean stem first (Family 1) maximises grounding quality; feeding mixed audio still works but YAMNet's weakness persists.

**Assessment:** The strongest music captioning option for this pipeline. No new model infrastructure, no new latency bottleneck, richest output vocabulary, directly addresses 0008's hallucination problem for music shots.

---

## Method Family 3 — Music Generation (Decode Side)

At decode time, the new `music` field needs a generator. These are distinct from MMAudio (SFX/ambience) and need to co-exist with it.

### 3a. MusicGen (Meta)

**What it is:** Text+melody-conditioned music generator. Apache 2.0 license. Available in Small (300M), Medium (1.5B), and Large (3.3B) variants. `musicgen-melody` (1.5B) accepts a reference audio clip to condition style — highly relevant for leitmotif consistency (Family 5).

**Self-host on RunPod:**
- Large model: ~4.5GB VRAM. Runs alongside MMAudio on A100 or A6000.
- Generation speed: ~1–2× real-time for MusicGen Large on A100 (desk estimate).
- For ~700 music shots × 6s avg = 4,200s to generate → ~1.2–2.4 hours on A100.
- A100 on RunPod: $1.64/hr → ~$2–$4 desk estimate for full film.
- If running on the same pod as Wan (at marginal cost): essentially free additional GPU time on an already-rented pod.

**Quality:**
- Better than MMAudio at producing coherent, sustained musical pieces.
- Tends toward pop/electronic genres; cinematic orchestral is under-represented in training.
- 30s max generation length; film shots are typically <12s, so this is not a constraint.
- Does not reproduce copyright-protected melodies (model generates novel compositions conditioned on description).

**License:** Apache 2.0 — commercial use OK.

### 3b. Stable Audio Open

**What it is:** Stability AI's open music/sound generator (1.3B params, 44kHz stereo). Text-conditioned. Can generate up to ~190s. Apache 2.0 license.

**Self-host on RunPod:**
- ~8GB VRAM. Fits on A40/A100.
- Generation speed similar to MusicGen.
- Cost: same envelope as MusicGen, ~$2–$4 at film scale.

**Quality:**
- Strong at ambient/textural music and genre-specific prompts.
- Less precise on orchestral than MusicGen Large; better on electronic/cinematic textures.
- Useful as a fallback or A/B test.

**License:** Apache 2.0.

### 3c. Suno / Udio APIs

**What they are:** Commercial music generation APIs producing high-quality, full-production music from text prompts. Both support cinematic/orchestral styles.

**Pricing (desk estimate — subject to change):**
- Suno: approximately $0.01–$0.05 per generation via API (credit-based, opaque pricing).
- Udio: similar range.
- For ~700 clips: **$7–$35**. Much more expensive than self-hosted options.

**Quality:** Subjectively the best of any available option. Suno v4 in particular handles orchestral and dramatic genres well.

**Copyright/commercial concerns:** Both services prohibit using generated audio in commercial products without a commercial tier subscription. Generated audio may also incorporate elements of copyrighted training data in ways that are not yet settled legally. The article's thesis already acknowledges copyright — this compounds it.

**Assessment:** Best subjective quality, worst cost-vs-quality at scale, commercial licensing risk. Viable for one-off demos, not for pipeline production.

### 3d. ElevenLabs Music

**What it is:** ElevenLabs' sound effects endpoint (`fal-ai/elevenlabs/sound-effects/v2`) is already in the pipeline at $0.001/s (`strategies_audio.py`). ElevenLabs also has a music generation capability, though it is less documented than their SFX endpoint.

**Cost:** If using the same $0.001/s rate: 4,200s of music → $4.20.

**Assessment:** The SFX endpoint is confirmed in the pipeline (measured cost). Music generation quality from ElevenLabs is unknown for orchestral content. Not a primary recommendation.

### 3e. Google Lyria via Vertex AI

**What it is:** Google DeepMind's music foundation model, available via Vertex AI. Designed specifically for high-quality music generation including orchestral and cinematic.

**Pricing:** Not publicly listed as of research cutoff. Requires Vertex AI billing. Likely in the $0.01–$0.05/clip range.

**Quality:** Google's internal evals show Lyria competitive with Suno on musicality. Orchestral coverage claimed.

**License/commercial:** Google Workspace terms apply. Safer than Suno/Udio for commercial use, but still "in the style of" rather than Williams' actual score.

**Assessment:** Potentially the best API option for cinematic quality, but pricing opacity and Vertex AI setup overhead make it harder to evaluate quickly.

---

## Method Family 4 — Symbolic/MIDI Transcription

This family attempts to represent the score symbolically (MIDI notes, tempo, key) rather than as audio. The idea is a compact, lossless-ish representation that can be re-synthesised.

### 4a. basic-pitch (Spotify)

**What it is:** Lightweight neural pitch tracker. Converts monophonic or simple polyphonic audio to MIDI. Very fast; ~5–10× real-time on CPU.

**License:** Apache 2.0.

**Quality for orchestral:** Poor. Basic-pitch is designed for guitar/bass/vocal melody extraction. An orchestral score with 30+ simultaneous voices, wide dynamic range, and complex timbre produces chaotic MIDI output. The model was not trained on orchestral transcription.

### 4b. MT3 / MR-MT3 (Google Magenta)

**What it is:** Multi-track music transcription model. Transforms audio into multi-instrument MIDI. MR-MT3 is the improved follow-up. Trained on a broad mix of genres.

**License:** Apache 2.0.

**Quality for orchestral:** Better than basic-pitch on polyphonic content, but still limited. Published results on orchestral music show SDR improvements over basic-pitch but remain far from useful for complex film scores — particularly strings, which MT3's training data underrepresents relative to pop instruments.

**The deeper problem:** Even perfect MIDI transcription loses timbre entirely. A MIDI representation of a John Williams cue re-synthesised with FluidSynth produces a "music box" version, not a cinematic score. For the encode to be useful, the MIDI would need to feed a high-quality orchestral synthesiser — which does not exist in open-source form at the quality level needed.

**Verdict: Not recommended.** Include for completeness, but the timbre loss and orchestral accuracy ceiling make this a dead end for film scoring quality. If the goal were to encode melodic structure as a compact token (e.g. for leitmotif identity), a simpler audio embedding approach (CLAP, see Family 5) is more practical.

---

## Method Family 5 (Optional) — Leitmotif Memory

**What it is:** Detect recurring musical themes across the film, store one description per theme, and reference the same theme ID in the encode. At decode time, regenerate the same "leitmotif" consistently across all its appearances, rather than generating a fresh unrelated cue for each shot.

This mirrors the existing character-continuity work (`docs/research/0011-character-continuity`), which clusters character appearances and stores a consistent visual description.

**How it would work:**
1. Run source separation (Family 1) across all shots.
2. Embed each music-bucket shot's audio using CLAP (Contrastive Language-Audio Pretraining) or a music fingerprinting approach (Dejavu, Chromaprint).
3. Cluster embeddings to find recurring themes.
4. Caption each unique theme once (Family 2).
5. Store `theme_id` in the encode; decode maps theme IDs to descriptions and feeds the same description (with a fixed seed) to MusicGen.

**Cost overhead:** CLAP inference is fast (~0.1s/clip, free); clustering is negligible. Adds ~$0 to the pipeline.

**Expected quality gain:** High narrative coherence — the Force theme sounds like the Force theme throughout, rather than five unrelated orchestral cues. The biggest qualitative lift of any option here, and the one most aligned with the article's thesis about lossy compression of narrative information.

**Caveat:** Requires a robust clustering threshold — over-merging distinct themes is worse than no clustering. A pilot on a single film's score would validate the threshold.

---

## Copyright Note

None of the methods above reproduce the original score. Demucs extraction isolates Williams' music as it appears in the film — that stems directly from the copyrighted soundtrack and cannot be redistributed or used as a component in a generated work. Captioning and regeneration produce *novel* music "in the style of" the description, which is legally distinct from the original. The article's "the score is simply gone" narrative remains accurate: what any of these methods produce is a thematic echo, not a reconstruction. The copyright note does not change with any implementation choice.

---

## Recommended Path: Separate → Caption → Regenerate

| Step | Tool | Where | Cost at film scale |
|------|------|--------|--------------------|
| 1. Separate | Bandit (DnR separator) | RunPod (same pod) | ~$0.15 *(desk)* |
| 2. Caption stem | Gemini native audio + music prompt | API (existing) | ~$0.60 *(desk)* |
| 3. Store | New `music` field in encode JSON | n/a | $0 |
| 4. Generate | MusicGen Large (self-hosted) | RunPod (same pod) | ~$2–$3 *(desk)* |
| **Total** | | | **~$3–$4** *(desk)* |

**Why this path:**
- Bandit gives the cleanest music stem for film audio (directly trained on DME data vs. Demucs's generic 4-stem).
- Gemini native audio is the strongest captioning option and integrates trivially — no new model infrastructure.
- MusicGen is free at marginal cost on an already-running RunPod pod and has the right license for production use.
- End-to-end cost is a rounding error compared to the existing $8–10 per film for video generation.

**What this does not fix:** The generated music is stylistically inspired by the description, not a reproduction of Williams' cues. Leitmotif consistency requires Family 5 on top.

---

## Cheapest/Laziest Option: Gemini Music Prompt (No Separation)

Skip source separation entirely. Modify the existing Gemini shot-description call to add a second field `music`, asking Gemini to describe the music specifically from YAMNet's `music`-bucket labels:

```
"music": a description of the musical score in this shot — mood, tempo, instrumentation, and how it functions in the scene. If there is no discernible music, say so.
```

**Cost:** $0 incremental — folds into the existing Gemini call.

**Integration effort:** ~10 lines of code in `encode.py`.

**Quality:** Moderate. YAMNet labels give Gemini some grounding (e.g. "Orchestra, Theme music, Piano"), but without the actual audio waveform, Gemini will hallucinate musical details from visual context (the weakness documented in 0008). Better than the current merged `sound` field, but weaker than Gemini native audio on a clean stem.

**When to use:** As the entry point — ship this first to validate that a dedicated `music` field improves MusicGen output at all, before spending time on source separation.

---

## Method Comparison Table

| Method family | Local / API | Cost (film scale) | Quality ceiling | Notes |
|---------------|-------------|-------------------|-----------------|-------|
| Demucs v4 htdemucs | Local (RunPod) | ~$0.15 | Moderate (SFX bleed) | Generic 4-stem; `other` ≈ music+SFX |
| Bandit (DnR) | Local (RunPod) | ~$0.15 | Good (film-specific) | Best separation for this use case |
| LP-MusicCaps | Local | $0 | Low–Moderate | Limited orchestral coverage |
| Qwen2-Audio | Local | ~$0.20 | Moderate–Good | Needs 16GB VRAM |
| Gemini native audio | API | ~$0.60 | Good | Best fit; already integrated |
| MusicGen Large | Local (RunPod) | ~$2–$3 | Moderate | Apache 2.0, marginal on existing pod |
| Stable Audio Open | Local (RunPod) | ~$2–$3 | Moderate | Better on textural/ambient |
| Suno / Udio | API | $7–$35 | High | Commercial restrictions |
| ElevenLabs Music | API | ~$4.20 | Unknown (orchestral) | Pipeline-integrated SFX endpoint |
| Google Lyria | API | Unknown | High (claimed) | Pricing opacity |
| basic-pitch | Local | $0 | Poor (orchestral) | Not recommended |
| MT3 / MR-MT3 | Local | $0 | Poor (orchestral) | Timbre lost entirely |
| Gemini prompt only (lazy) | API | $0 | Low–Moderate | Zero integration cost |

---

## Flagged Follow-Up Experiment

**Run Bandit + MusicGen Large on one Star Wars IV scene (opening crawl, shots 1–15, ~3 min audio):**

1. Extract audio for shots 1–15 from the original file.
2. Run Bandit separation → `music` stem.
3. Run Gemini native audio on the music stem with the music-description prompt.
4. Run MusicGen Large with the resulting description.
5. Compare the output against the current MMAudio output for the same shots.

**Cost:** ~$0.10 of RunPod GPU time + ~$0.05 Gemini API = ~$0.15 total.

**What it proves:** Whether the separation → caption → generation chain produces music that is subjectively better than MMAudio's ambience-based output on music-heavy shots. The opening crawl is the highest-music-density portion of the film and gives a quick read on quality ceiling.

**This is the single most valuable thing to do next** — it turns all desk estimates above into measured numbers.
