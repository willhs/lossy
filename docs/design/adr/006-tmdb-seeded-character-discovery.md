---
id: adr-006-tmdb-seeded-character-discovery
type: decision
purpose: "Record the decision to use TMDB cast data to seed stage 3 character discovery."
scope: ["design", "encoder", "stage3"]
non_goals: []
tags: ["adr", "encoder", "character-registry", "tmdb"]
related: ["design/character-registry.md", "design/portrait-generation.md"]
---

# Context

Stage 3 builds a character registry by sending all shot subject descriptions to Gemini in a single call. For Star Wars EP IV (~2,012 shots), this produced only 9 characters and missed significant characters: Jawas (46 shots, ~2.3% of film), Uncle Owen (7 shots), Aunt Beru (9 shots). About 45% of shots describe characters generically ("small hooded creatures", "old farmer") without explicit names — Gemini can't reliably identify these without prior knowledge of the cast.

The single 2,000-line Gemini call was also expensive and hit context limits on very long films.

# Decision

When `--tmdb-id` is provided, use a two-step supervised discovery pipeline:

1. **Text match** (free): Word-boundary regex match of TMDB character names against shot subjects. Pre-assigns shots where the character is explicitly named.
2. **Supervised Gemini** (for unmatched shots only): Send remaining shots to Gemini in batches of 200, with the TMDB cast list in the system prompt as context. Gemini can now map "small hooded creatures" → "Jawas" because it knows Jawas are in the cast.

The top 30 cast members by billing order are fetched from TMDB. Character names are normalized (strip `(voice)`, `(uncredited)`, ` / alternate` annotations).

Without `--tmdb-id`, or if the TMDB fetch fails, stage 3 falls back to the existing unsupervised single-call path unchanged.

Supports both movies (`--tmdb-type movie`, default) and TV series (`--tmdb-type tv`).

# Consequences

**Good:**
- Star Wars EP IV: 16 characters found (vs 9 before), including all previously missing characters
- Reduces Gemini input size significantly — only unmatched shots go to the API (869 text-matched, 1143 to Gemini for Star Wars IV)
- Batching at 200 shots/call prevents context limit issues on long films
- TV series work: BCS S01E01 correctly identified Jimmy McGill and Kim Wexler from visual descriptions alone
- Fully backwards-compatible — no change to behaviour without `--tmdb-id`

**Bad:**
- Requires a TMDB API key (free, but one more credential to manage)
- TMDB cast list is film-level, not scene-level — supporting characters with few appearances may not be in the top 30 cast
- TV episode credits (`/tv/{id}/credits`) return series-level cast, not episode-specific cast

# Alternatives Considered

- **Episode-specific TV credits** (`/tv/{id}/season/{s}/episode/{e}/credits`): More accurate for TV but requires parsing episode info from the filename/user. Deferred — series-level credits are sufficient for now.
- **Send all shots to Gemini with cast list** (no text match pre-filter): Simpler but doesn't reduce API costs or call size.
- **Fine-tune Gemini on cast photos**: Best accuracy but not feasible without per-film training data.
