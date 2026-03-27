---
id: task-0011
type: spec
purpose: "Add named scene range definitions that produce separate stitched videos for key scenes"
tags: ["stitch", "pipeline", "scenes"]
related: []
created: 2026-03-26
updated: 2026-03-26
---

# Scene Range Stitching

## Goal

Support defining named scene ranges per media output and automatically stitch each range as a separate output video during the stitch stage. This lets users extract key scenes (e.g. "hologram scene: shots 328-360") as standalone clips from a full decode run.

## Context

The primary output being developed is Star Wars IV using the `runpod-wan` video strategy with `runpod-mmaudio` audio (producing `reconstructed_runpod-wan+runpod-mmaudio.mp4`). Scene extraction targets this output.

The pipeline currently stitches all decoded clips into a single `reconstructed_<strategy>.mp4`. There's a `--start-index` flag but no end-index, no named ranges, and no multi-output support.

For longer films, specific scenes are more interesting to review than the full reconstruction. The metadata (prompts.json) already contains subjects, dialogue, and timecodes per shot -- enough to identify scenes. What's missing is a way to define scene ranges and have the stitch stage produce separate videos for each.

## Scope

### Must Do
- Define a `scenes.json` format in the output directory for named shot ranges
- Add `--end-index` support to stitch filtering (complement to existing `--start-index`)
- Stitch stage produces `scene_<name>_<strategy>.mp4` for each defined range
- Scene videos include audio/speech tracks when those options are active
- Pipeline auto-stitches scenes when `scenes.json` exists

### Might Do
- A CLI flag to stitch only specific named scenes (skip full reconstruct)
- Include scene timecode info in stitch output summary

### Won't Do
- Auto-detection of scene boundaries (this is manual/curated)
- A GUI or interactive scene selector
- Modifying the encode stage

## Success Criteria

- [ ] `scenes.json` with named ranges produces separate video files per scene
- [ ] Scene videos have correct audio/speech when those options are enabled
- [ ] Pipeline runs scene stitching automatically after full stitch
- [ ] Existing stitch behaviour unchanged when no `scenes.json` present
