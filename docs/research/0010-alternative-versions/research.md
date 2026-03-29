---
id: 0010-alternative-versions
type: note
purpose: "Explore ideas for generating alternative versions of source media by transforming encoded prompts before decode."
scope: ["research", "prompts", "decode", "experimentation"]
tags: ["research", "prompts", "creative", "alternative-versions"]
related: ["research/0002-shot-to-prompt/research.md", "research/0003-prompt-to-video/research.md"]
---

## Question

What modifications to the lossy pipeline would generate alternative versions of a film — aesthetically or structurally different from the original — that are achievable today and interesting as experiments?

## Background: `prompts.json` as an Editable Intermediate

The lossy pipeline encodes a film into a structured per-shot representation stored in `prompts.json`. Each entry has:

```json
{
  "index": 0,
  "start_s": 0.0,
  "end_s": 28.779,
  "duration_s": 28.779,
  "camera_motion_detected": "static",
  "audio_detected": { "bucket": "music", "labels": [...] },
  "dialogue": null,
  "description": {
    "shot_type": "wide",
    "camera_movement": "static",
    "subjects": "...",
    "action": "...",
    "lighting": "...",
    "color_palette": "...",
    "mood": "...",
    "setting": "...",
    "sound": "..."
  }
}
```

Every field in `description` (and `dialogue`) is editable text. Modifying these before decode produces a different video of the same film structure. The pipeline already supports re-decoding from existing prompts:

```bash
python pipeline.py media/film.mp4 -o output/film_alt --strategy runpod-wan --skip encode1 encode2
```

This skips encoding entirely and decodes from whatever `prompts.json` exists in the output directory. The implication is that any transform script that edits `prompts.json` can produce an alternative version at zero encoding cost.

## Ideas

### 1. Aesthetic / Visual Transformations

These are pure field replacements — no LLM required. A short Python script can apply them across an entire `prompts.json` in seconds.

#### Film Noir Preset
Replace all `lighting`, `color_palette`, and `mood` fields with a cohesive noir palette:
- `lighting`: "harsh directional light, deep chiaroscuro shadows, pools of light in darkness"
- `color_palette`: "black, white, and charcoal grey with occasional stark contrast highlights"
- `mood`: `<original mood>` prepended with "tense, cynical, shadowy"

Cost: nothing (field replacement). Regeneration cost: full decode run.

#### Time-of-Day Shift
Replace all `lighting` fields with a consistent time-of-day. E.g.:
- "golden hour": "warm low-angle sunlight casting long golden shadows"
- "night": "moonlit, deep blue ambient light with scattered artificial sources"
- "overcast": "diffuse, flat grey daylight with no directional shadows"

Works best on exterior scenes; interior shots will look strange but interestingly so.

#### Weather Overlay
Append a weather condition to every `setting` field:
- `"in the Tatooine desert"` → `"in the Tatooine desert, during a heavy downpour"`
- `"aboard the Death Star"` → `"aboard the Death Star, visibility reduced by thick fog"`

Rain and snow have an outsized effect on video generation outputs. A film generated entirely in rain looks radically different despite no plot changes.

#### Color Palette Swap
Replace all `color_palette` values with a single theme. Some options:
- Sepia: `"warm amber, sienna, and faded tan"` — aged film aesthetic
- Neon cyberpunk: `"electric blue, hot pink, and acid green against black"`
- Muted Scandinavian: `"pale grey, off-white, and desaturated olive green"`

#### Camera Style Homogenization
Replace all `camera_movement` values with a single style to impose a consistent cinematographic voice:
- `"handheld, slightly unsteady"` — documentary / guerrilla filmmaking feel
- `"slow dolly"` — prestige drama
- `"static"` — Wes Anderson / Kubrick tableau style

The prompt formatters (especially `_format_prompt_wan`) remap camera terms to professional vocabulary, so this also changes how the backend interprets movement.

---

### 2. Character and Subject Alterations

Video models don't know character names. `"Luke Skywalker"` has no special meaning to Wan 2.1 or Seedance — only the descriptive context around it matters. Subject fields are therefore safe to rewrite.

#### Subject Substitution
Replace named subjects with new descriptors:
- `"Luke Skywalker, a young man with blond hair"` → `"a young woman with close-cropped dark hair"`
- Useful for exploring alternate casting, gender swaps, or de-named versions.

#### Costume / Era Swap
Append costume context to subject descriptions:
- `"...dressed in Victorian-era clothing"` → shifts the whole film's wardrobe
- `"...wearing tactical military gear"` → militarises the aesthetics
- `"...in futuristic chrome armour"` → sci-fi up a fantasy film

This pairs well with a matching `setting` transformation to make it coherent.

---

### 3. Structural / Narrative Changes

These require an LLM pass over `prompts.json` — Claude or Gemini processes each entry and rewrites the specified fields while preserving schema structure.

#### Alternate Dialogue
The most tractable narrative change. The `dialogue` field contains the subtitle text that feeds into TTS speech generation. Rewriting dialogue leaves video clips unchanged but changes what every character says.

Possibilities:
- **Comedy dub**: rewrite dialogue to be absurdist/comedic
- **Alternate language**: translate dialogue to another language (voice stays the same ElevenLabs voice — interesting artifact)
- **Plot deviation**: rewrite dialogue to suggest an alternate story (e.g., Vader is Luke's uncle, not father)
- **Explanation mode**: have all characters explain what's happening fourth-wall style

Implementation: batch all `dialogue` fields to an LLM with a system prompt describing the desired rewrite style.

#### Genre Shift
Rewrite `mood` and `action` fields via LLM to shift the film's genre while preserving its visual structure:
- Action → Horror: `"tense standoff"` → `"terrifying, claustrophobic dread"`, `"rushes toward enemy"` → `"flees in panic from something unseen"`
- Drama → Comedy: `"argues fiercely"` → `"bickers absurdly over something trivial"`
- Fantasy → Procedural: `"prepares to cast a spell"` → `"carefully files paperwork before engaging the procedure"`

The result keeps the same shot structure but the generated clips reflect the new genre feel.

#### LLM Plot Rewrite
The most ambitious option. Send the full `prompts.json` (or batches of scenes) to an LLM with a prompt like:

> "This is a shot-by-shot description of a film. Rewrite the `action` field of each shot to tell an alternate version of the story where [X]. Preserve all other fields exactly."

E.g., "the Empire actually wins", "Han Solo is the main villain", "it's a love story between Vader and Leia". The video model still generates from the same settings, subjects, and camera movements — only the events change.

This requires careful prompting to ensure:
- All shots get an `action` entry (no nulls)
- Actions remain visually generatable (no "five years pass in an instant")
- Scene-level coherence (connected shots should describe connected events)

#### Pacing Compression / Highlights
Remove shots to produce a condensed version:
- Keep only shots with `audio_detected.bucket == "speech"` → dialogue-only edit
- Keep only shots with `duration_s > 5` → longer, more cinematic moments
- Keep every Nth shot → mechanical skip edit (like a film viewed on fast-forward)
- Keep shots matching specific `mood` values → mood-curated edit

Stitch handles arbitrary subsets of clips; no structural changes needed.

#### Shot Reversal
Reverse the order of all entries in `prompts.json`. The film plays backwards structurally. With audio, this means the spoken dialogue also plays in reverse chronological order — a jarring but interesting art experiment.

---

### 4. Audio Alterations

#### Music Genre Swap
Modify `sound` fields to describe different music styles. E.g., if the original says "sweeping orchestral score", replace with "lo-fi hip-hop beat", "blues harmonica", or "heavy metal riff". MMAudio and ElevenLabs interpret these descriptions directly; they're not constrained to reproduce the original score.

#### Silent Film
Strip all `dialogue` fields (set to null) and remove speech references from `sound` fields. With a matching aesthetic transform (desaturate, add grain via `color_palette` edit), this produces a silent-era reconstruction.

---

## Implementation Approach

### Field Replacement Script

For aesthetic transforms, a short Python script suffices:

```python
import json

with open("output/film/prompts.json") as f:
    prompts = json.load(f)

for entry in prompts:
    d = entry["description"]
    d["lighting"] = "harsh directional light, deep chiaroscuro shadows"
    d["color_palette"] = "black, white, and charcoal grey"
    # ... etc

with open("output/film_noir/prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)
```

Then run decode on the new output directory:
```bash
python pipeline.py media/film.mp4 -o output/film_noir --strategy runpod-wan --skip encode1 encode2
```

### LLM Transform Script

For narrative transforms:

1. Load `prompts.json`
2. Batch shots (e.g., 20 at a time to fit context)
3. For each batch, send to Claude/Gemini with a system prompt defining the desired transform and schema preservation rules
4. Reassemble and write `prompts_alt.json`
5. Validate all required fields are present and non-empty

Key constraint: the LLM must return valid JSON matching the original schema. Using structured output (Gemini `response_schema` or Claude tool use) makes this reliable.

---

## Recommendations (Ranked by Achievability x Interest)

| Idea | Effort | Interest | Cost to Generate |
|------|--------|----------|-----------------|
| Alternate dialogue | Low | Very High | Full decode (audio only if video cached) |
| Film noir preset | Low | High | Full decode |
| Weather overlay | Very Low | Medium | Full decode |
| Genre shift (LLM) | Medium | High | Full decode |
| LLM plot rewrite | High | Very High | Full decode |
| Time-of-day shift | Low | Medium | Full decode |
| Subject substitution | Low | Medium | Full decode |
| Highlights reel | Low | Medium | Partial decode |
| Silent film | Very Low | Low-Medium | Partial decode (no speech) |

**Suggested first experiment**: Film noir preset on Star Wars IV. It requires zero LLM work, runs entirely as a JSON transform, and the contrast between the source material's bright sci-fi look and a hard-boiled noir reconstruction would be visually striking and easy to evaluate.

**Most interesting long-term**: Alternate dialogue. It reuses all existing video clips (no regeneration needed if clips are cached), only re-running speech synthesis. This makes iteration fast and cheap. The dialogue rewrite + re-stitch cycle could run in minutes for a feature film.

---

## Open Questions

1. **Visual coherence of aesthetic presets**: Does applying the same lighting/palette to all shots produce a cohesive look, or does per-shot variation in the base description create jarring inconsistency across cuts? Would need to test on a short clip sequence first.

2. **Subject substitution fidelity**: How much does renaming/describing a character differently affect generation? Video models may still produce recognisable settings and environments even if the subject description changes. The reverse — setting stays the same but character looks different — is probably achievable.

3. **LLM rewrite granularity**: For plot rewrites, should context be per-shot, per-scene, or whole-film? Per-shot risks incoherence; whole-film context may exceed context limits for long films. Scene-batching (using the manifest's scene boundaries) is probably the right granularity.

4. **Dialogue rewrite quality**: LLM-written dialogue will be generic without knowing the characters. Providing the original dialogue as context ("the original line was X, rewrite it as Y style") would produce more grounded results than rewriting from scratch.
