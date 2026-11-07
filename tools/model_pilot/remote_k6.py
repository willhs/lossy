"""Runs ON THE POD: render the pilot benchmark with Kandinsky 6.0 Pro distill.

Diffusers Kandinsky6TI2VAPipeline, 10 steps, guidance 1.0 (model card). The
29B transformer is ~58GB in bf16, more than a 48GB A6000 holds, so weights are
stored in FP8 and computed in bf16 (layerwise casting) with model CPU offload.
Native 480x864, 24fps, with the model's own synced audio kept in the file.

K6 is trained on 5s clips. We ask for the shot's own length first and fall back
to the native 121 frames if the pipeline rejects it; either way the timing is
recorded per shot, because K6's cost per shot depends on which one applies.

    python remote_k6.py benchmark.json /workspace/k6/pro-distill /workspace/k6/out
"""

import gc
import json
import os
import sys
import time

import torch
from diffusers import Kandinsky6TI2VAPipeline
from diffusers.utils import encode_video

FPS = 24
NATIVE_FRAMES = 121


def frames_for(target_s: float) -> int:
    n = max(25, min(NATIVE_FRAMES, round(target_s * FPS)))
    return (n - 1) // 4 * 4 + 1  # 4k+1, the usual causal-VAE constraint


def main():
    bench_path, model_dir, out_dir = sys.argv[1:4]
    native_only = "--native" in sys.argv
    os.makedirs(out_dir, exist_ok=True)
    bench = json.load(open(bench_path))
    tpath = os.path.join(out_dir, "timings.json")
    timings = json.load(open(tpath)) if os.path.exists(tpath) else {}

    t0 = time.time()
    pipe = Kandinsky6TI2VAPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    # FP8 storage, bf16 compute: the 29B transformer is ~58GB in bf16, more
    # than a 48GB A6000 holds; in FP8 it is ~29GB and stays resident on GPU.
    pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

    # Memory plan. The pod container is capped at 50GB RAM (cgroup memory.max)
    # even on a 503GB host. enable_model_cpu_offload() parks every idle
    # component in that RAM, and the process was OOM-killed at VAE decode,
    # silently, three times. So no offload at all: encode every prompt up
    # front with the text encoders on GPU, drop them, then keep transformer +
    # VAEs resident on GPU for the whole run.
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    embeds = {}
    with torch.no_grad():
        for shot in bench["shots"]:
            pe, ppe, *_ = pipe.encode_prompt(prompt=shot["prompt"], do_classifier_free_guidance=False,
                                             device=torch.device("cuda"))
            embeds[shot["index"]] = (pe.cpu(), ppe.cpu())
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    gc.collect()
    torch.cuda.empty_cache()
    for name in ("transformer", "vae", "audio_vae", "vocoder"):
        getattr(pipe, name).to("cuda")
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    timings["_load_seconds"] = round(time.time() - t0, 1)
    print(f"loaded + encoded {len(embeds)} prompts in {timings['_load_seconds']}s", flush=True)

    def render(idx, n):
        pe, ppe = embeds[idx]
        return pipe(prompt_embeds=pe.to("cuda"), pooled_prompt_embeds=ppe.to("cuda"),
                    height=480, width=864, num_frames=n, num_inference_steps=10, guidance_scale=1.0,
                    generator=torch.Generator("cpu").manual_seed(idx))

    for shot in bench["shots"]:
        idx = shot["index"]
        dest = os.path.join(out_dir, f"{idx:04d}.mp4")
        if os.path.exists(dest):
            continue
        n = NATIVE_FRAMES if native_only else frames_for(shot["target_s"])
        t1 = time.time()
        out = None
        try:
            try:
                out = render(idx, n)
            except Exception as e:  # shorter-than-native rejected -> native
                if n == NATIVE_FRAMES:
                    raise
                print(f"{idx}: {n} frames rejected ({e!r:.200}); retrying at {NATIVE_FRAMES}", flush=True)
                n = NATIVE_FRAMES
                out = render(idx, n)
            encode_video(out.frames[0], fps=FPS, output_path=dest,
                         audio=out.audio[0][None], audio_sample_rate=pipe.audio_sample_rate)
            dt = time.time() - t1
            timings[str(idx)] = {"ok": True, "seconds": round(dt, 1), "frames": n, "out_s": round(n / FPS, 2)}
            print(f"{idx}: {dt:.1f}s for {n} frames", flush=True)
        except Exception as e:
            timings[str(idx)] = {"ok": False, "seconds": round(time.time() - t1, 1), "error": repr(e)[:300]}
            print(f"{idx}: FAILED {e!r:.300}", flush=True)
        json.dump(timings, open(tpath, "w"), indent=1)
        out = None
        gc.collect()
        torch.cuda.empty_cache()
    print("K6_RENDER_DONE", flush=True)


if __name__ == "__main__":
    main()
