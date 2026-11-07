"""Render the pilot benchmark with LTX-2.5 distilled on the pilot pod's ComfyUI.

The API-format graph mirrors the official ComfyUI blueprint
(Comfy-Org/workflow_templates blueprints/text_to_video_ltx_2_5.json): stage 1
at half resolution on the 8-sigma distilled schedule, x2 latent upsample, then
a 3-sigma refinement pass at full resolution. The prompt enhancer is left OFF
so every model sees the same production prompt.

Variants (the cheaper-levers sweep) change one knob each:
    --variant base        two-stage, 1280x704 out, DiffVAE (template default)
    --variant convvae     same, conv VAE (faster decode)
    --variant onestage    stage 1 only at 640x352 (no upsample/refine)

    uv run python tools/model_pilot/run_ltx25.py --variant base
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import config  # noqa: E402
import runpod_pod  # noqa: E402

FPS = 25
OUT_W, OUT_H = 1280, 704  # matches runpod-ltx2; /2 = 640x352, both /32
STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
NEG = "worst quality, blurry, distorted, low resolution, watermark, subtitles"

UNET = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
TE = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VAE = {"diff": "ltx-2.5-video-vae-bf16.safetensors", "conv": "ltx-2.5-video-vae-conv-bf16.safetensors"}
AVAE = "ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"


def frames_for(target_s: float) -> int:
    """LTX needs frames % 8 == 1; clamp to 25..121 like runpod-ltx2."""
    n = max(25, min(121, round(target_s * FPS)))
    return (n - 1) // 8 * 8 + 1 if (n - 1) % 8 < 4 else ((n - 1) // 8 + 1) * 8 + 1


def workflow(prompt: str, seed: int, length: int, variant: str) -> dict:
    two_stage = variant != "onestage"
    vae = VAE["conv" if variant == "convvae" else "diff"]
    w1, h1 = (OUT_W // 2, OUT_H // 2)
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "ltxv", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["2", 0]}},
        "7": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": float(FPS)}},
        "8": {"class_type": "EmptyLTXVLatentVideo",
              "inputs": {"width": w1, "height": h1, "length": length, "batch_size": 1}},
        "9": {"class_type": "LTXVEmptyLatentAudio",
              "inputs": {"audio_vae": ["4", 0], "frames_number": length, "frame_rate": FPS, "batch_size": 1}},
        "10": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["8", 0], "audio_latent": ["9", 0]}},
        "11": {"class_type": "LTXVDualCFGGuider",
               "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1],
                          "video_cfg": 1.0, "audio_cfg": 1.0}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "14": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE1_SIGMAS}},
        "15": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["13", 0], "guider": ["11", 0], "sampler": ["12", 0],
                          "sigmas": ["14", 0], "latent_image": ["10", 0]}},
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}},
    }
    video_lat, audio_lat = ["16", 0], ["16", 1]
    if two_stage:
        wf.update({
            "20": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPSCALER}},
            "21": {"class_type": "LTXVLatentUpsampler",
                   "inputs": {"samples": ["16", 0], "upscale_model": ["20", 0], "vae": ["3", 0]}},
            "22": {"class_type": "LTXVConcatAVLatent",
                   "inputs": {"video_latent": ["21", 0], "audio_latent": ["16", 1]}},
            "23": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
            "24": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE2_SIGMAS}},
            "25": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["23", 0], "guider": ["11", 0], "sampler": ["12", 0],
                              "sigmas": ["24", 0], "latent_image": ["22", 0]}},
            "26": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["25", 0]}},
        })
        video_lat, audio_lat = ["26", 0], ["26", 1]
    wf.update({
        "30": {"class_type": "VAEDecodeTiled",
               "inputs": {"samples": video_lat, "vae": ["3", 0], "tile_size": 512, "overlap": 64,
                          "temporal_size": 64, "temporal_overlap": 16}},
        "31": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": audio_lat, "audio_vae": ["4", 0]}},
        "32": {"class_type": "CreateVideo", "inputs": {"images": ["30", 0], "audio": ["31", 0], "fps": float(FPS)}},
        "33": {"class_type": "SaveVideo",
               "inputs": {"video": ["32", 0], "filename_prefix": "pilot_ltx25", "format": "mp4", "codec": "h264"}},
    })
    return wf


def _first_output(history: dict) -> dict | None:
    for node_out in history.get("outputs", {}).values():
        for key in ("images", "videos", "gifs", "video"):
            items = node_out.get(key)
            if items:
                return items[0]
    return None


def main():
    ap = argparse.ArgumentParser(description="Render the pilot benchmark with LTX-2.5")
    ap.add_argument("--variant", default="base", choices=["base", "convvae", "onestage"])
    ap.add_argument("--bench", default="output/pilot0025/benchmark.json")
    ap.add_argument("--dir", default="output/pilot0025/encode")
    ap.add_argument("--only", type=int, nargs="*", help="shot indices (default: all)")
    ap.add_argument("--tag", default="", help="suffix for the clips dir, e.g. -4090 for a GPU sweep")
    a = ap.parse_args()

    config.load_env()
    bench = json.load(open(a.bench))
    out_dir = os.path.join(os.path.dirname(a.bench), "clips", f"ltx25-{a.variant}{a.tag}")
    os.makedirs(out_dir, exist_ok=True)
    tpath = os.path.join(out_dir, "timings.json")
    timings = json.load(open(tpath)) if os.path.exists(tpath) else {}

    s = runpod_pod.RunPodSession(a.dir, keep_pod=45)
    if not s.reconnect():
        sys.exit("no pod")
    s.mark_clean_exit()

    for shot in bench["shots"]:
        idx = shot["index"]
        if a.only and idx not in a.only:
            continue
        dest = os.path.join(out_dir, f"{idx:04d}.mp4")
        if os.path.exists(dest):
            continue
        n = frames_for(shot["target_s"])
        t0 = time.time()
        hist = s.submit_workflow(workflow(shot["prompt"], idx, n, a.variant), timeout=1200)
        dt_s = time.time() - t0
        out = _first_output(hist) if hist else None
        data = s.download_output(out) if out else None
        if not data:
            print(f"  {idx}: FAILED after {dt_s:.0f}s")
            timings[str(idx)] = {"ok": False, "seconds": dt_s}
        else:
            open(dest, "wb").write(data)
            timings[str(idx)] = {"ok": True, "seconds": round(dt_s, 1), "frames": n,
                                 "out_s": round(n / FPS, 2)}
            print(f"  {idx}: {dt_s:.1f}s for {n} frames")
        json.dump(timings, open(tpath, "w"), indent=1)


if __name__ == "__main__":
    main()
