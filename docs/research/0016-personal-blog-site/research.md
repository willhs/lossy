---
id: "0016"
type: research
purpose: "Investigate options for a personal blog site focused on weird/unusual software topics."
scope: ["blog", "personal-site"]
non_goals: ["implementation", "content strategy"]
tags: ["research", "blog", "web"]
related: []
---

# Personal Blog Site — Research

## Question

What is the right platform, tech stack, hosting, and structural approach for a personal technical blog focused on weird software — esoteric languages, unusual architectures, vintage computing, niche tools?

## Context

Goal is to build a custom personal site to publish technical writing and project write-ups, replacing Substack as canonical home. Key tension: blog vs wiki/digital garden. Needs to showcase software skills, support rich media, and be cheap to run.

## Scope

- **Investigate**: Platform options, notable technical blogs for reference, blog vs garden tradeoff, hosting options, domain/path strategy
- **Out of scope**: Content strategy, actual writing, cross-posting automation

---

## Findings

### 1. Notable Technical Blogs in the HN Ecosystem

These are the most-referenced personal technical blogs in the Hacker News community (2024–2025):

| Blog | Domain | Focus | What makes it stand out |
|------|--------|--------|--------------------------|
| Simon Willison | simonwillison.net | AI, LLMs, web dev | Prolific daily writing + link curation; custom Django blog |
| Julia Evans | jvns.ca | Linux, SQL, systems | Zines, hand-drawn explainers; high signal-to-noise |
| Fabien Sanglard | fabiensanglard.net | Game dev, systems | Deep technical dives; rare but long-form |
| Bart Ciechanowski | ciechanow.ski | Physics/CS explainers | Extraordinary interactive scroll-driven animations; very rare posts |
| Brendan Gregg | brendangregg.com | Performance, systems | Reference-grade depth; flame graphs inventor |
| Josh Comeau | joshwcomeau.com | Frontend, CSS, React | Heavily interactive demos baked into posts |
| Maggie Appleton | maggieappleton.com | Visual, design, tech | Design-forward; digital garden hybrid; high craftsmanship |
| Rachel | rachelbythebay.com | Infra, sysadmin | Blunt, opinionated; no frills; longevity |
| Eli Bendersky | eli.thegreenplace.net | Compilers, PLs | Long-form technical depth; consistent for 15+ years |
| Jeff Geerling | jeffgeerling.com | Raspberry Pi, hardware | Blog companion to 1M-subscriber YouTube channel |
| Neal Agarwal | neal.fun | Interactive art/web | Not a blog — interactive experiments; very distinctive |
| Brandur Leach | brandur.org | Postgres, Go, infra | Polished long-form; obsessive editorial quality |

**Weird-software adjacent reference sites:**
- **esoteric.codes** (Daniel Temkin) — interviews creators of esolangs, code art, constraint-based programming. Runs on custom PiranhaCMS/.NET. Conceptual framing, not just technical. Migrated from Tumblr 2017.
- **esolangs.org** — wiki for esoteric languages

**Key patterns from top blogs:**
- Long-form wins over frequency for technical audiences
- Interactive/visual elements dramatically increase shareability (Ciechanowski, Comeau)
- Having a clear, ownable niche beats broad coverage
- Plain, fast designs age better than trendy ones

---

### 2. Platform Options

#### Option A: Static Site Generator + Cloudflare Pages (Recommended baseline)

| SSG | Build speed | Template language | JavaScript | Themes | Best for |
|-----|-------------|-------------------|-----------|--------|----------|
| **Hugo** | Fastest (1000p in ~2s) | Go templates | None by default | Many | Writers who want zero JS, max speed |
| **Astro** | Fast | JSX/components | Islands (opt-in) | Good | Developers who want interactive islands |
| **11ty** | Fast | Flexible (Nunjucks, Liquid, etc.) | None by default | Minimal | Developers who want full control |

**Hugo**: Best if you want to just write and publish. Go templates are awkward to customize but themes are abundant. Zero client-side JS.

**Astro**: Best if you want interactive elements (animated demos, embedded charts) alongside static content. Island architecture means you can add React/Svelte components per-post without sacrificing static performance. Strong content collections support. Growing community.

**11ty**: Best if you want maximum flexibility with minimal opinions. Steeper setup but nothing is hidden from you.

#### Option B: Ghost

- **Hosted (ghost.io)**: $9–$25/mo. Polished editor, newsletters built-in, native cross-posting. No code required. Limited customization.
- **Self-hosted on budgie**: Free (infra cost only). Full control. Maintenance overhead — upgrades, backups, Docker. Ghost is Node.js, fits existing budgie setup.
- Good for writing-first focus; less good for interactive technical posts or showing off custom code.

#### Option C: Bear Blog

- Extremely minimal. Free tier. No themes, no JS, just writing.
- Great for writing, bad for showcasing software skills or rich media.

#### Option D: Fully Custom

- Maximum control and expression. Can make it genuinely weird (fitting the theme).
- Highest effort. Could use any stack: Next.js, SvelteKit, custom Go/Haskell server, etc.
- Risk: builds the blog instead of writing the blog.

---

### 3. Blog vs Digital Garden

**Traditional blog:** chronological, polished before publishing, SEO-optimized, follows news/events rhythm.

**Digital garden:** non-linear, notes evolve over time, "tended" rather than "posted," bi-directional links, explicit maturity labels (seedling → evergreen).

**Hybrid approaches** (most interesting people do this):
- Maggie Appleton: essays + garden with explicit growth states
- Tom Critchlow: personal wiki on GitHub Pages alongside blog posts
- Andy Matuschak: working notes as public thinking environment

**Key question**: Do you want to write polished articles (blog), or maintain an evolving reference (garden)? The hybrid is more work but more distinctive.

For a weird-software site: a garden makes sense for things like "my notes on Brainfuck interpreters" that grow over time, while a blog suits "I implemented X in Piet and here's what I learned."

---

### 4. Hosting Options

| Platform | Free tier bandwidth | CDN nodes | Build minutes | Custom domain | Notes |
|----------|-------------------|-----------|---------------|---------------|-------|
| **Cloudflare Pages** | Unlimited | 300+ PoPs | 500/mo | Yes (free) | Best for static; zero egress cost |
| **Netlify** | 100 GB/mo | ~30 PoPs | 300/mo | Yes (free) | Mature; forms/identity features |
| **Vercel** | 100 GB/mo | ~70 PoPs | 6000 min/mo | Yes (free) | Best for Next.js; less ideal for Hugo/Astro |
| **GitHub Pages** | ~100 GB soft | GitHub CDN | Jekyll only | Yes (free) | Simple push-to-deploy; limited build control |

**Recommendation**: Cloudflare Pages for static sites — unlimited bandwidth, largest CDN, free, no vendor lock-in beyond Pages.

---

### 5. Domain / URL Strategy

- **Subdomain of willhs.me** (e.g., `blog.willhs.me` or `weird.willhs.me`): Fast to set up, keeps identity unified.
- **Own domain** (e.g., `weird.software` or similar): Stronger brand, better if this becomes a standalone identity.

**URL path conventions for SEO:**
- Posts: `/posts/slug` or `/YYYY/slug` (year prefix adds context but makes reorganization harder)
- Tags/categories: `/tags/esoteric-languages`
- Avoid: dates in slugs unless historically relevant content

---

## Conclusions

After iterating with the user and a second-opinion review, the framing converged significantly from the original brief:

1. **Real purpose** is partly a *practice-of-sharing* goal — the user typically keeps ideas private and wants to share more. Audience target is humble: even one friend, colleague, or local tech person getting value = success. Authenticity is the constraint.

2. **Real content shape** is practitioner build-in-public AI/software experiments, not high-concept cultural commentary. Planned posts include "my lil football coach" (Glory Lab), agentic dev workflow notes, and "angels and demons" (Judge Your Life). Quirky-because-the-projects-are-quirky, not because the framing is weird-coded.

3. **Brand**: name-branded site (not "Weird Software"). The corrected purpose makes the case decisive: a practice-of-sharing site for concrete work needs the most durable, least performative brand available — the user's name. Closest analogs: Geoffrey Litt, Linus Lee, Simon Willison, Nicholas Carlini — all name-branded build-in-public people.

4. **Frame**: "working notebook" — not portfolio. Lower pressure, better matches the habit being built. Portfolio implies curation and stale "About me" pages.

5. **"Weird Software"**: shrinks to a post title, tag, or lightweight category. Don't brand ahead of evidence. Promote later only if a real body of work emerges.

6. **Real failure mode**: publishing friction, not weak branding. Optimize for "next 10 posts go up easily," not for cleverness.

7. **Tech**: Astro + Cloudflare Pages remains the right baseline — supports interactive demos via islands (for future need), low-friction publishing (markdown in git → push to deploy), free unlimited hosting, RSS out of the box.

---

## Recommendations

**v1 spec:**

- **Identity**: name-branded site (use willhs.me as a starting subdomain like `notes.willhs.me`, or use willhs.me itself if not already in use; can move to standalone domain later if it earns it)
- **Frame**: working notebook — engineer sharing experiments, tools, and project writeups
- **Tagline**: signals "engineer sharing experiments and project writeups," not a thesis about weirdness
- **Tech**: Astro + Cloudflare Pages
- **Content shape**: `/posts/` (project writeups + workflow notes + reflections), `/about/` (one page), nothing else in v1
- **Tags**: lightweight tag system; "weird-software" is one tag among others
- **Publishing workflow**: markdown files in git, push to deploy, no CMS, no scheduling, no review pipeline
- **RSS feed**: yes (Astro `@astrojs/rss`)
- **Cross-posting**: optional later; canonical URLs always live on the site
- **Interactive demos**: use Astro islands when a post wants one; not required for v1
- **Inaugural post**: one of the planned project writeups (e.g. Glory Lab / "my lil football coach")

**Single highest-leverage move for v1**: design the site and workflow around publishing the next 10 posts easily, not around expressing a clever brand idea.

**Out of scope for v1** (defer to later iterations):
- Digital garden / `/notes/` section with bidirectional links
- Custom standalone domain
- Cross-posting automation
- Newsletter integration
- "Weird Software" as a series brand

**Alternative**: Hugo + Cloudflare Pages if you want simpler setup and pure writing focus with no interactivity.

**Avoid for this use case**: Ghost hosted (monthly cost, limited customization), Bear Blog (too minimal for showcasing software skills), Substack (no custom domain on free tier, limited control).
