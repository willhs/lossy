---
id: plan-0013
type: spec
purpose: "Implementation plan for evaluating lightweight audio captioning models as a replacement for the YAMNet + Gemini pipeline."
tags: ["plan", "audio", "encoder", "eval", "research"]
related: ["./task.md"]
---

# Audio Captioning Eval — Implementation Plan

## Overview

Build a standalone eval script (`tools/eval_audio_captioning.py`) that extracts audio clips for ~20 representative shots from the star_wars_iv_v2 encode, runs CLAPCap (and optionally Whisper Audio Captioning) on each clip, and produces a side-by-side comparison against the existing Gemini sound descriptions from `prompts.json`. Write findings to `docs/research/0008-audio-captioning/`.

**Primary Goal**: Determine whether CLAPCap produces sound descriptions that are grounded, speech-free, descriptive enough for MMAudio, and fast enough on M-series Mac.

**Approach**: Research-first eval. No changes to `encode.py`. CLAPCap is the primary candidate because it's the only model that (a) installs cleanly via pip, (b) works on Python 3.11+, and (c) has an MIT license.

## Current State Analysis

The current pipeline (encode stage2) works in two steps:
1. **YAMNet** (`encode.py:430-482`) classifies audio into 521 AudioSet classes, aggregated into 6 buckets per shot (`encode.py:355-403`), cached to `audio_labels.json`
2. **Gemini** (`encode.py:538-744`) receives YAMNet labels + keyframe images and writes a `sound` field in `prompts.json`

MMAudio strategies consume `description.sound` as a text prompt (`strategies_audio.py:296`), after speech filtering (`strategies_audio.py:77`).

### Key Discoveries
- CoNeTTE requires Python `>=3.10, <3.11` — project uses 3.11, so **CoNeTTE is out**
- CLAPCap (`msclap` on PyPI, MIT license) works on Python 3.8-3.13, simple API: `CLAP(version="clapcap", use_cuda=False).generate_caption(file_paths=[...])`
- Whisper Audio Captioning (CC BY-NC, `trust_remote_code=True` required) is a "might do" — license and install friction are concerns
- star_wars_iv_v2 has 1640 shots across 106 minutes; audio buckets: speech 654, music 492, effects 367, other 121, silence 6
- YAMNet expects mono 16kHz WAV (`encode.py:406-427`); CLAPCap resamples internally via librosa

## Desired End State

- `tools/eval_audio_captioning.py` extracts clips and runs models, producing a markdown comparison table and JSON results
- `docs/research/0008-audio-captioning/research.md` contains findings and a clear recommendation
- No changes to `encode.py`, `decode.py`, or any pipeline code

## What We're NOT Doing

- Modifying `encode.py` to integrate a new model (separate follow-up task)
- Removing YAMNet or TensorFlow from the pipeline
- Building a production-ready captioning stage
- Evaluating paid/API-based services
- Testing CoNeTTE (Python version incompatible)

---

## Selected Test Shots

19 shots chosen for bucket coverage, duration range (0.8s–11.5s), film spread (1m–99m), and content variety:

| Index | Time   | Dur   | Bucket  | DLG | Content                              |
|-------|--------|-------|---------|-----|--------------------------------------|
| 5     | 1.0m   | 10.0s | music   |     | Opening crawl, full orchestral score |
| 63    | 4.0m   | 0.8s  | effects |     | Blaster fire, explosions             |
| 133   | 7.0m   | 3.3s  | music   | Y   | Tense orchestral + mechanical hum    |
| 188   | 11.9m  | 1.5s  | silence |     | Faint bowed strings, canyon          |
| 208   | 12.7m  | 3.6s  | other   |     | Desert wind, footsteps               |
| 210   | 12.8m  | 5.5s  | speech  |     | Cave acoustics, metallic clinking    |
| 373   | 23.8m  | 2.2s  | effects | Y   | Small room, dishes clinking          |
| 432   | 28.4m  | 1.1s  | other   |     | Desert wind, mechanical beeping      |
| 484   | 30.5m  | 3.8s  | speech  | Y   | Desert wind, melancholic underscore  |
| 753   | 49.6m  | 6.8s  | music   |     | Cantina-like, glass clinking         |
| 787   | 52.8m  | 1.6s  | speech  | Y   | Spaceship interior hum               |
| 813   | 55.6m  | 3.5s  | effects | Y   | Spaceship engines, mechanical        |
| 1039  | 71.1m  | 3.2s  | speech  | Y   | Spaceship engine hum, whirring       |
| 1098  | 76.0m  | 2.2s  | music   |     | Tense orchestral score               |
| 1138  | 78.2m  | 1.0s  | effects |     | Laser blast, mechanical              |
| 1178  | 79.4m  | 1.5s  | other   | Y   | Trash compactor, grinding            |
| 1411  | 91.8m  | 1.3s  | music   | Y   | Lightsaber clash                     |
| 1485  | 94.4m  | 2.6s  | effects |     | Space battle explosions              |
| 1585  | 99.3m  | 11.5s | speech  | Y   | Station hum, electronic bleeps       |

---

## Phase 1: Install CLAPCap

### Overview
Install `msclap` into the project venv and verify it loads.

### Tasks

- [x] Install msclap: `pip install msclap`
- [x] Verify import and model download work:

```bash
python -c "from msclap import CLAP; m = CLAP(version='clapcap', use_cuda=False); print('OK')"
```

This will download the CLAPCap checkpoint on first run (~800MB). Confirm it completes without errors.

### Success Criteria
- [x] Run: `python -c "from msclap import CLAP; CLAP(version='clapcap', use_cuda=False)"` — no errors

---

## Phase 2: Build Eval Script

### Overview
Create `tools/eval_audio_captioning.py` that extracts audio clips for the selected shots, runs CLAPCap, loads Gemini baselines, and produces a comparison report.

### Tasks

#### 1. Create `tools/eval_audio_captioning.py`

- [x] Create `tools/eval_audio_captioning.py` with the structure below.

The script should:

1. **Load data**: Read `prompts.json` and `audio_labels.json` from the output dir
2. **Extract clips**: For each selected shot, use ffmpeg to extract the audio segment from `audio.wav` into a temp directory as individual WAV files (mono, 16kHz to match YAMNet convention, though CLAPCap resamples internally)
3. **Run CLAPCap**: Load model once, caption each clip, measure per-clip inference time
4. **Build comparison**: For each shot, collect YAMNet labels, Gemini `sound` description, and CLAPCap caption
5. **Assess quality dimensions**: For each CLAPCap caption, flag:
   - `speech_leak`: does the caption mention speech/voice/dialogue/talking?
   - `descriptive`: is the caption more than just a category label? (length > 20 chars as rough proxy)
6. **Output results**: Write JSON results + markdown comparison table

```
Usage:
  python tools/eval_audio_captioning.py output/star_wars_iv_v2/
  python tools/eval_audio_captioning.py output/star_wars_iv_v2/ --shots 5,63,133
  python tools/eval_audio_captioning.py output/star_wars_iv_v2/ --model clapcap
  python tools/eval_audio_captioning.py output/star_wars_iv_v2/ --model whisper  (if installed)
```

Key implementation details:

- **CLI**: Use argparse. Positional arg for output dir. `--shots` to override default shot list (comma-separated indices). `--model` to select model (default: `clapcap`). `--output` for results dir (default: `eval_audio_captioning/` inside the output dir).
- **Default shot indices**: Hardcode the 19 indices from the table above.
- **Clip extraction**: `ffmpeg -i audio.wav -ss {start_s} -t {duration_s} -ac 1 -ar 16000 clip_{index}.wav` into a temp dir. Use `subprocess.run`.
- **CLAPCap inference**: `from msclap import CLAP; model = CLAP(version="clapcap", use_cuda=False); captions = model.generate_caption(file_paths=[path])`. Time each call with `time.perf_counter()`.
- **Speech detection**: Reuse the keyword list from `strategies_audio.py` speech filter — check if caption contains any of: voice, speaking, dialogue, conversation, talking, shout, whisper, scream, yell, murmur, narrate, vocal, speech.
- **Output JSON** (`eval_audio_captioning/results.json`): List of dicts with `index`, `bucket`, `duration_s`, `has_dialogue`, `yamnet_labels`, `gemini_sound`, `model_caption`, `inference_time_s`, `speech_leak`, `caption_length`.
- **Output markdown** (`eval_audio_captioning/report.md`): Summary stats (avg inference time, speech leak rate, avg caption length) + per-shot comparison table.

#### 2. Add Whisper Audio Captioning support (optional)

- [x] If proceeding with Whisper: add a `--model whisper` path that loads `MU-NLPC/whisper-small-audio-captioning` with `trust_remote_code=True`, uses the `clotho > caption:` style prefix, and follows the same extract→caption→compare flow. Guard the import so the script works without it installed.

### Success Criteria

- [x] Run: `python tools/eval_audio_captioning.py output/star_wars_iv_v2/` — completes without errors, produces `results.json` and `report.md`
- [x] Manual: Review `report.md` — confirm it has a comparison table with all 19 shots showing YAMNet labels, Gemini sound, and CLAPCap caption side by side
- [x] Manual: Check inference times are reasonable (expect 1-5s per clip on M-series Mac CPU)

---

## Phase 3: Run Eval and Write Research Doc

### Overview
Run the eval, analyze results, and write the research document.

### Tasks

#### 1. Run CLAPCap eval

- [ ] Run: `python tools/eval_audio_captioning.py output/star_wars_iv_v2/`
- [ ] Manual: Review `output/star_wars_iv_v2/eval_audio_captioning/report.md` and assess:
  - Are CLAPCap captions grounded in actual audio? (Compare against what the shot should sound like)
  - How many captions leak speech references?
  - Are captions descriptive enough for MMAudio prompts? (Compare richness to Gemini descriptions)
  - What's the per-clip and projected full-film throughput?

#### 2. Run Whisper eval (if installed)

- [ ] Run: `python tools/eval_audio_captioning.py output/star_wars_iv_v2/ --model whisper`
- [ ] Manual: Compare Whisper results against CLAPCap

#### 3. Write research doc

- [ ] Create `docs/research/0008-audio-captioning/research.md` following the project's research doc format (see `docs/research/0007-speech-filter/research.md` for structure):
  - **Problem**: YAMNet+Gemini pipeline hallucinates, leaks speech, heavy dependency
  - **Approach**: Evaluated CLAPCap (and optionally Whisper) on 19 representative shots
  - **Results**: Per-model table with metrics (speech leak rate, avg caption length, avg inference time, qualitative grounding assessment)
  - **Comparison**: Side-by-side examples for a few interesting shots (one where CLAPCap beats Gemini, one where it doesn't, one edge case)
  - **Recommendation**: Clear verdict — adopt, investigate further, or keep current pipeline
  - **Integration notes**: If recommending adoption, sketch what `encode.py` changes would look like (which functions to modify, where CLAPCap would slot in) — but no code changes

### Success Criteria

- [ ] `docs/research/0008-audio-captioning/research.md` exists with complete findings
- [ ] Research doc answers all four success criteria from the task spec: (1) audio grounding, (2) speech-freedom, (3) MMAudio descriptiveness, (4) throughput
- [ ] Clear recommendation with rationale

---

## Final Checklist

- [ ] All phases complete
- [ ] Eval script runs cleanly: `python tools/eval_audio_captioning.py output/star_wars_iv_v2/`
- [ ] Research doc written with recommendation
- [ ] No changes to pipeline code (`encode.py`, `decode.py`, `stitch.py`, `strategies_*.py`)

## Documentation Updates

- [ ] Create `docs/research/0008-audio-captioning/research.md` with findings and recommendation
- [ ] If recommending CLAPCap adoption: note a follow-up task is needed for `encode.py` integration

## References

- Task: `docs/tasks/0013-audio-captioning-eval/task.md`
- Current audio pipeline: `encode.py:322-516` (YAMNet), `encode.py:538-744` (Gemini prompts)
- Speech filter: `strategies_audio.py:21-138`
- Prior research: `docs/research/0007-speech-filter/research.md`
- CLAPCap: `msclap` PyPI package, MIT license, [microsoft/CLAP](https://github.com/microsoft/CLAP)
