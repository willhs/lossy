#!/usr/bin/env python3
"""Standalone MMAudio V2 inference script for running on a RunPod pod via SSH.

This bypasses ComfyUI entirely, running MMAudio inference directly via Python.
Designed to run concurrently alongside ComfyUI (which handles video generation).

Usage (on the pod):
    python3 /tmp/mmaudio_standalone.py \
        --prompt "Wind blowing through trees, birds chirping" \
        --duration 5.0 \
        --seed 42 \
        --output /tmp/audio_output.flac

    # With custom model directory (default: ComfyUI models dir)
    python3 /tmp/mmaudio_standalone.py \
        --prompt "Heavy rain on a tin roof" \
        --duration 10.0 \
        --models-dir /workspace/runpod-slim/ComfyUI/models/mmaudio \
        --output /tmp/rain.flac

Requirements on pod:
    - mmaudio package (pip install mmaudio)
    - torch with CUDA
    - torchaudio
    - Internet access (downloads ~5GB of model weights on first run)

Based on: https://github.com/hkchengrex/MMAudio/blob/main/demo.py
"""

import argparse
import gc
import os
import sys
import time

import torch
import torchaudio

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_vram_mb() -> int:
    """Get current GPU VRAM usage in MiB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() // (1024 * 1024)
    return 0


def get_peak_vram_mb() -> int:
    """Get peak GPU VRAM usage in MiB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() // (1024 * 1024)
    return 0


@torch.inference_mode()
def run_inference(
    prompt: str,
    duration: float,
    seed: int,
    output_path: str,
    models_dir: str,
    steps: int = 25,
    cfg_strength: float = 4.5,
) -> bool:
    """Run MMAudio text-to-audio inference and save as FLAC.

    Uses the official MMAudio evaluation pipeline. Downloads standard .pth
    weights via model.download_if_needed() on first run (FeaturesUtils uses
    torch.load() internally, so safetensors won't work for VAE/synchformer).

    The models_dir argument is currently unused but reserved for a future
    optimization where we load Kijai safetensors directly (bypassing
    FeaturesUtils, similar to how ComfyUI-MMAudio does it).

    Returns True on success, False on failure.
    """
    from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate,
                                    setup_eval_logging)
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.networks import MMAudio, get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils

    setup_eval_logging()

    vram_before = get_vram_mb()
    print(f"VRAM before load: {vram_before} MiB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Find the large_44k_v2 model config
    model_name = "large_44k_v2"
    if model_name not in all_model_cfg:
        available = list(all_model_cfg.keys())
        print(f"Available model configs: {available}")
        for name in available:
            if "large" in name and "44k" in name:
                model_name = name
                break
        else:
            print("Error: No suitable model config found.")
            return False

    model: ModelConfig = all_model_cfg[model_name]
    seq_cfg = model.seq_cfg

    # Download standard .pth weights if not cached.
    # FeaturesUtils uses torch.load() internally for VAE/synchformer,
    # so we can't pass safetensors paths directly. The standard download
    # path is simplest for the prototype.
    print("Checking model weights (will download ~5GB on first run)...")
    model.download_if_needed()

    # Load the main network
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    print(f"Loaded main model from {model.model_path}")

    vram_after_net = get_vram_mb()
    print(f"VRAM after net load: {vram_after_net} MiB")

    # Set up FlowMatching sampler
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=steps)

    # Load feature utils (VAE, synchformer, CLIP)
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False,
    )
    feature_utils = feature_utils.to(device, dtype).eval()

    vram_after_features = get_vram_mb()
    print(f"VRAM after feature utils: {vram_after_features} MiB")

    # Text-to-audio mode: no video input
    clip_frames = None
    sync_frames = None

    # Set duration and update sequence lengths
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    # Generate audio
    print(f"Generating {duration}s audio: {prompt[:80]}...")
    t_start = time.time()

    audios = generate(
        clip_frames,
        sync_frames,
        [prompt],
        negative_text=[""],
        feature_utils=feature_utils,
        net=net,
        fm=fm,
        rng=rng,
        cfg_strength=cfg_strength,
    )

    t_elapsed = time.time() - t_start
    peak_vram = get_peak_vram_mb()
    print(f"Inference: {t_elapsed:.1f}s, peak VRAM: {peak_vram} MiB")

    # Save as FLAC
    audio = audios.float().cpu()[0]
    torchaudio.save(output_path, audio, seq_cfg.sampling_rate)
    print(f"Saved: {output_path} (sample rate: {seq_cfg.sampling_rate})")

    # Cleanup
    del net, feature_utils, fm, audios, audio
    gc.collect()
    torch.cuda.empty_cache()

    vram_final = get_vram_mb()
    print(f"VRAM after cleanup: {vram_final} MiB")
    print(f"Peak VRAM (total session): {peak_vram} MiB")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Standalone MMAudio V2 text-to-audio inference"
    )
    parser.add_argument("--prompt", required=True, help="Text prompt for audio generation")
    parser.add_argument("--duration", type=float, required=True, help="Audio duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", required=True, help="Output FLAC file path")
    parser.add_argument(
        "--models-dir",
        default="/workspace/runpod-slim/ComfyUI/models/mmaudio",
        help="Directory containing MMAudio safetensors model files",
    )
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=4.5, help="CFG strength")
    args = parser.parse_args()

    print("=== MMAudio Standalone Inference ===")
    print(f"Prompt:   {args.prompt[:80]}...")
    print(f"Duration: {args.duration}s")
    print(f"Seed:     {args.seed}")
    print(f"Steps:    {args.steps}, CFG: {args.cfg}")
    print(f"Models:   {args.models_dir}")
    print(f"Output:   {args.output}")
    print()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    try:
        ok = run_inference(
            prompt=args.prompt,
            duration=args.duration,
            seed=args.seed,
            output_path=args.output,
            models_dir=args.models_dir,
            steps=args.steps,
            cfg_strength=args.cfg,
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
