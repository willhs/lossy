---
id: experiment-0003
type: experiment
purpose: "A/B test comparing original prose prompts vs Wan-optimized cinematography prompts on shots 10-24."
tags: ["experiment", "prompts", "wan", "video-generation"]
related: ["../../tasks/0004-prompt-reformatting/task.md", "./0001-first-e2e-decode-test.md"]
---

# Experiment 0003: Prompt Reformatting A/B Test

## Hypothesis

Wan-optimized prompts (front-loaded subject/action, cinematography vocabulary, no metadata labels) will produce more accurate and visually coherent video clips compared to the current prose-style prompts.

## Setup

- **Source**: Star Wars EP IV, shots 10-24
- **Model**: Replicate wan-video/wan-2.2-t2v-fast, 832x480, 81 frames @ 16fps
- **Control**: Current `format_prompt()` -- prose with "Color palette:", "Mood:" labels (existing clips from experiment 0001)
- **Treatment**: `_format_prompt_wan()` -- Subject > Action > Camera > Style, cinematography terms
- **Cost**: ~$0.75 for treatment group (control reuses existing clips)

## Prompt Comparison

### Shot 10

**Control prompt:**
> Cinematic extreme wide shot, slow zoom out. The camera slowly zooms out, revealing more of the vastness of space. The two moons remain in their relative positions, while the planet's curvature becomes more pronounced as the zoom progresses. Harsh, direct light from an unseen star source, creating strong contrast between illuminated and shadowed areas on the moons and planet. Color palette: Deep blacks of space, speckled with white stars, contrasted by the muted blues, grays, and earthy tones of the moons and planet.. Mood: Vast, awe-inspiring, and slightly desolate. Deep space, with a planet and its moons in the background.

**Treatment prompt:**
> two moons of varying sizes, and a planet's atmosphere and surface are visible in the distance. The camera slowly zooms out, revealing more of the vastness of space. The two moons remain in their relative positions, while the planet's curvature becomes more pronounced as the zoom progresses. Extreme Wide shot, slow dolly out. Deep space, with a planet and its moons in the background. Harsh, direct light from an unseen star source, creating strong contrast between illuminated and shadowed areas on the moons and planet. Deep blacks of space, speckled with white stars, contrasted by the muted blues, grays, and earthy tones of the moons and planet. tones. Vast, awe-inspiring, and slightly desolate atmosphere.

### Shot 11

**Control prompt:**
> Cinematic extreme wide shot, static. The spaceship fires a red laser beam from its front array of engines towards the lower left of the frame. The planet and moons remain static in the background. Harsh, direct light from an unseen source, creating strong highlights on the spaceship and casting deep shadows. The planet is illuminated from the side, revealing atmospheric details. Color palette: Deep blacks of space, muted blues and browns of the planet, grey of the spaceship, bright red of the laser, and white of the stars and moons.. Mood: Tense, action-oriented, epic. Deep space, with a planet and its moons in the background.

**Treatment prompt:**
> a large, grey spaceship with multiple glowing engines, a planet with swirling clouds, two moons, a red laser beam. The spaceship fires a red laser beam from its front array of engines towards the lower left of the frame. The planet and moons remain static in the background. Extreme Wide shot, static. Deep space, with a planet and its moons in the background. Harsh, direct light from an unseen source, creating strong highlights on the spaceship and casting deep shadows. The planet is illuminated from the side, revealing atmospheric details. Deep blacks of space, muted blues and browns of the planet, grey of the spaceship, bright red of the laser, and white of the stars and moons. tones. Tense, action-oriented, epic atmosphere.

### Shot 12

**Control prompt:**
> Cinematic extreme wide shot, static. The spacecraft slowly moves further away from the camera, shrinking in size and becoming more distant relative to the moons and the planet. Harsh, direct light from an unseen source (likely a star) illuminates the spacecraft and the moons, creating strong highlights and deep shadows. The planet's atmosphere is subtly lit from below. Color palette: Deep blacks of space, muted blues and browns of the planet, grey of the moons, and bright orange-yellow from the spacecraft's engines.. Mood: Vast, isolated, and awe-inspiring, with a sense of impending journey or departure.. Deep space, with a large planet and two moons in the background.

**Treatment prompt:**
> A large, textured planet dominates the lower third of the frame, with a crescent moon to its left and a smaller, distant moon above and to the right. A small, blocky spacecraft with multiple glowing engines is positioned in the upper right quadrant, moving away from the camera. The spacecraft slowly moves further away from the camera, shrinking in size and becoming more distant relative to the moons and the planet. Extreme Wide shot, static. Deep space, with a large planet and two moons in the background. Harsh, direct light from an unseen source (likely a star) illuminates the spacecraft and the moons, creating strong highlights and deep shadows. The planet's atmosphere is subtly lit from below. Deep blacks of space, muted blues and browns of the planet, grey of the moons, and bright orange-yellow from the spacecraft's engines. tones. Vast, isolated, and awe-inspiring, with a sense of impending journey or departure. atmosphere.

### Shot 13

**Control prompt:**
> Cinematic extreme wide shot, slow zoom out and slight tilt down. The Star Destroyer, initially distant, dramatically enters the frame from the top right and appears to descend towards the planet. The small transport ship remains relatively static in comparison, positioned below the Star Destroyer. Harsh, direct light from an unseen source illuminates the Star Destroyer, creating sharp contrasts and highlighting its metallic surface. The planet and moons are dimly lit, suggesting they are in shadow or illuminated by reflected light. Color palette: Deep blacks of space, muted blues and browns of the planet, grey and white of the spacecraft, with small red accents on the transport.. Mood: Ominous, awe-inspiring, and tense, emphasizing the overwhelming power of the Star Destroyer.. Deep space, high above a planet with visible atmospheric layers and two moons.

**Treatment prompt:**
> A massive, dagger-shaped Imperial Star Destroyer looms in the upper right quadrant, dwarfing a small, blocky transport ship. Two moons and a planet's atmosphere are visible in the lower half of the frame. The Star Destroyer, initially distant, dramatically enters the frame from the top right and appears to descend towards the planet. The small transport ship remains relatively static in comparison, positioned below the Star Destroyer. Extreme Wide shot, slow dolly out and slight tilt down. Deep space, high above a planet with visible atmospheric layers and two moons. Harsh, direct light from an unseen source illuminates the Star Destroyer, creating sharp contrasts and highlighting its metallic surface. The planet and moons are dimly lit, suggesting they are in shadow or illuminated by reflected light. Deep blacks of space, muted blues and browns of the planet, grey and white of the spacecraft, with small red accents on the transport. tones. Ominous, awe-inspiring, and tense, emphasizing the overwhelming power of the Star Destroyer. atmosphere.

### Shot 14

**Control prompt:**
> Cinematic extreme wide shot, slow zoom out. The camera slowly zooms out, revealing more of the vastness of space and the scale of the Star Destroyer relative to the planet and moon. The ship remains static in its position. Harsh, direct light from an unseen star illuminates the top and side of the Star Destroyer, creating strong contrasts and deep shadows on its hull. The planet below is lit from the side, with its atmosphere glowing. Color palette: Dominant colors are the cool grays and whites of the Star Destroyer, the muted oranges and browns of the planet's surface, the deep blues and blacks of space, and the pale gray of the moon.. Mood: ominous, imposing, awe-inspiring, vast. Deep space, with a large planet and its moon in the background. The Star Destroyer is positioned in orbit or high atmosphere.

**Treatment prompt:**
> A massive Imperial Star Destroyer hangs in space, positioned above a swirling, atmospheric planet and its moon. The underside of the ship is visible, with its iconic dagger shape pointing towards the viewer. The camera slowly zooms out, revealing more of the vastness of space and the scale of the Star Destroyer relative to the planet and moon. The ship remains static in its position. Extreme Wide shot, slow dolly out. Deep space, with a large planet and its moon in the background. The Star Destroyer is positioned in orbit or high atmosphere. Harsh, direct light from an unseen star illuminates the top and side of the Star Destroyer, creating strong contrasts and deep shadows on its hull. The planet below is lit from the side, with its atmosphere glowing. Dominant colors are the cool grays and whites of the Star Destroyer, the muted oranges and browns of the planet's surface, the deep blues and blacks of space, and the pale gray of the moon. tones. ominous, imposing, awe-inspiring, vast atmosphere.

### Shot 15

**Control prompt:**
> Cinematic wide shot, static. A smaller, blocky ship detaches from the underside of the Star Destroyer and begins to move away. Harsh, direct lighting from an unseen source illuminates the Star Destroyer, creating strong highlights and shadows. The planet below is dimly lit, with a visible atmospheric glow along its curve. Color palette: Deep blacks of space, the grey and white of the Star Destroyer, and the reddish-brown and blue hues of the planet.. Mood: ominous, imposing, vast. Orbiting a desolate, rocky planet in outer space.

**Treatment prompt:**
> A large, wedge-shaped Imperial Star Destroyer is positioned in space above a planet. A smaller, blocky ship detaches from the underside of the Star Destroyer and begins to move away. Wide shot, static. Orbiting a desolate, rocky planet in outer space. Harsh, direct lighting from an unseen source illuminates the Star Destroyer, creating strong highlights and shadows. The planet below is dimly lit, with a visible atmospheric glow along its curve. Deep blacks of space, the grey and white of the Star Destroyer, and the reddish-brown and blue hues of the planet. tones. ominous, imposing, vast atmosphere.

### Shot 16

**Control prompt:**
> Cinematic medium wide shot, zoom out. The camera zooms out, revealing more of the space environment and the planet below. The Rebel ship's engine continues to burn brightly, and the Star Destroyer remains in its position relative to the planet. Harsh, direct lighting from an unseen source, creating strong highlights on the ships and casting deep shadows. The engine exhaust provides a bright, localized light source. Color palette: Deep blacks of space, muted oranges and browns of the planet, bright yellows and reds from the engine exhaust, and the metallic grays of the ships.. Mood: Tense, dramatic, action-oriented, with a sense of pursuit or escape.. Orbit above a desolate, reddish-brown planet with a thin blue atmosphere, set against the backdrop of deep space and stars.

**Treatment prompt:**
> A large Imperial Star Destroyer is in the upper right quadrant of the frame. In the foreground, a Rebel ship's engine is visible on the left, emitting bright yellow and red exhaust. The camera zooms out, revealing more of the space environment and the planet below. The Rebel ship's engine continues to burn brightly, and the Star Destroyer remains in its position relative to the planet. Medium Wide shot, dolly out. Orbit above a desolate, reddish-brown planet with a thin blue atmosphere, set against the backdrop of deep space and stars. Harsh, direct lighting from an unseen source, creating strong highlights on the ships and casting deep shadows. The engine exhaust provides a bright, localized light source. Deep blacks of space, muted oranges and browns of the planet, bright yellows and reds from the engine exhaust, and the metallic grays of the ships. tones. Tense, dramatic, action-oriented, with a sense of pursuit or escape. atmosphere.

### Shot 17

**Control prompt:**
> Cinematic wide shot, zoom out. The Star Destroyer fires a green laser beam from its ventral side. The camera zooms out, revealing more of the planet's atmosphere and the vastness of space. Harsh, direct lighting from an unseen source, casting sharp shadows on the Star Destroyer and illuminating the planet's surface. Color palette: Deep blacks of space, metallic grays of the Star Destroyer, vibrant green of the laser, and muted reds and oranges of the planet.. Mood: Ominous, powerful, and awe-inspiring.. Orbit above a desolate, alien planet in outer space.

**Treatment prompt:**
> A massive Imperial Star Destroyer is positioned in space above a reddish-brown planet. The Star Destroyer fires a green laser beam from its ventral side. The camera zooms out, revealing more of the planet's atmosphere and the vastness of space. Wide shot, dolly out. Orbit above a desolate, alien planet in outer space. Harsh, direct lighting from an unseen source, casting sharp shadows on the Star Destroyer and illuminating the planet's surface. Deep blacks of space, metallic grays of the Star Destroyer, vibrant green of the laser, and muted reds and oranges of the planet. tones. Ominous, powerful, and awe-inspiring. atmosphere.

### Shot 18

**Control prompt:**
> Cinematic medium wide shot, zoom out. The shot begins with a close view of the spacecraft. A bright green laser blast strikes the ship, followed by a massive explosion that engulfs the rear section of the vessel in pinkish smoke and debris. The camera zooms out slightly throughout the sequence, revealing more of the starfield. Harsh, directional lighting from an unseen source illuminates the spacecraft, creating strong highlights and shadows. The explosion provides its own intense, localized light. Color palette: Dominant colors are deep black for space, stark white and red for the ship, with vibrant green from the laser and a dramatic pink/orange from the explosion.. Mood: Intense, chaotic, destructive, and action-packed.. Deep space, with a clear view of a starfield.

**Treatment prompt:**
> A large, white and red spacecraft with intricate details, seen from a slightly low angle, against a backdrop of stars. The shot begins with a close view of the spacecraft. A bright green laser blast strikes the ship, followed by a massive explosion that engulfs the rear section of the vessel in pinkish smoke and debris. The camera zooms out slightly throughout the sequence, revealing more of the starfield. Medium Wide shot, dolly out. Deep space, with a clear view of a starfield. Harsh, directional lighting from an unseen source illuminates the spacecraft, creating strong highlights and shadows. The explosion provides its own intense, localized light. Dominant colors are deep black for space, stark white and red for the ship, with vibrant green from the laser and a dramatic pink/orange from the explosion. tones. Intense, chaotic, destructive, and action-packed. atmosphere.

### Shot 19-24

Interior corridor shots (C-3PO, R2-D2, Stormtroopers). See decode output for full prompts.

## Results

### Per-Shot Observations

| Shot | Control | Treatment | Notes |
|------|---------|-----------|-------|
| 10   |         |           |       |
| 11   |         |           |       |
| 12   |         |           |       |
| 13   |         |           |       |
| 14   |         |           |       |
| 15   |         |           |       |
| 16   |         |           |       |
| 17   |         |           |       |
| 18   |         |           |       |
| 19   |         |           |       |
| 20   |         |           |       |
| 21   |         |           |       |
| 22   |         |           |       |
| 23   |         |           |       |
| 24   |         |           |       |

### Summary

[Overall observations -- which approach produced better results and why]

## Recommendation

[Keep new formatter / revert / iterate further]
