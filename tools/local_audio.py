"""Local (Apple Silicon / CPU) audio strategies that plug into decode.run_audio.

Sidesteps RunPod/ComfyUI entirely: MMAudio (SFX) and MusicGen (music) run on
this machine via PyTorch (MPS if available, else CPU). Same generate() contract
and file naming as the hosted strategies, so decode.run_audio + stitch work
unchanged.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

# MMAudio's eval_utils.generate runs under torch.inference_mode(), which on
# recent torch trips "Inference tensors cannot be saved for backward" on MPS.
# no_grad gives the same speed without the inference-tensor restriction.
torch.inference_mode = torch.no_grad

from strategies_audio import AudioStrategy, _split_duration, filter_speech_from_sound, AudioClipResult  # noqa: E402


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalMMAudioStrategy(AudioStrategy):
    """MMAudio V2 (large_44k_v2) text-to-audio, run locally. SFX/ambience."""

    name = "mmaudio-local"
    MIN_DURATION = 1
    MAX_DURATION = 10
    NUM_STEPS = 25
    CFG = 4.5

    def __init__(self, device=None, model_name="medium_44k"):
        self.device = device or pick_device()
        # fp16 on MPS: float32 needs ~30GB (OOMs on 24GB unified). CPU stays fp32.
        self.dtype = torch.float16 if self.device == "mps" else torch.float32
        # large_44k_v2 OOMs on 24GB MPS; medium fits with better quality than small.
        self.model_name = model_name
        self._loaded = False

    def _target_durations(self, target_s: float) -> list[float]:
        return _split_duration(target_s, self.MIN_DURATION, self.MAX_DURATION)

    def _load(self):
        if self._loaded:
            return
        from mmaudio.eval_utils import all_model_cfg, setup_eval_logging
        from mmaudio.model.networks import get_my_mmaudio
        from mmaudio.model.utils.features_utils import FeaturesUtils
        setup_eval_logging()
        cfg = all_model_cfg[self.model_name]
        cfg.download_if_needed()
        self.cfg = cfg
        self.seq_cfg = cfg.seq_cfg
        print(f"  [MMAudio] loading on {self.device}...")
        net = get_my_mmaudio(cfg.model_name).to(self.device, self.dtype).eval()
        net.load_weights(torch.load(cfg.model_path, map_location=self.device, weights_only=True))
        self.net = net
        self.feature_utils = FeaturesUtils(
            tod_vae_ckpt=cfg.vae_path,
            synchformer_ckpt=cfg.synchformer_ckpt,
            enable_conditions=True,
            mode=cfg.mode,
            bigvgan_vocoder_ckpt=cfg.bigvgan_16k_path,
            need_vae_encoder=False,
        ).to(self.device, self.dtype).eval()
        self._loaded = True
        print("  [MMAudio] ready.")

    def generate(self, sound_description, audio_dir, shot_index, target_duration_s, seed=None):
        filtered = filter_speech_from_sound(sound_description)
        if filtered is None:
            print(f"  Shot {shot_index}: sound is entirely speech, skipping MMAudio")
            return []
        sound_description = filtered
        self._load()

        import torchaudio
        from mmaudio.eval_utils import generate as mm_generate
        from mmaudio.model.flow_matching import FlowMatching

        durations = self._target_durations(target_duration_s)
        results = []
        for part_idx, duration in enumerate(durations):
            clip_name = (f"{shot_index:04d}.flac" if len(durations) == 1
                         else f"{shot_index:04d}-{part_idx + 1:02d}.flac")
            clip_path = os.path.join(audio_dir, clip_name)
            try:
                rng = torch.Generator(device=self.device)
                rng.manual_seed((seed or 0) + part_idx)
                fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=self.NUM_STEPS)
                self.seq_cfg.duration = float(duration)
                self.net.update_seq_lengths(
                    self.seq_cfg.latent_seq_len, self.seq_cfg.clip_seq_len, self.seq_cfg.sync_seq_len)
                audios = mm_generate(
                    None, None, [sound_description], negative_text=[""],
                    feature_utils=self.feature_utils, net=self.net, fm=fm, rng=rng,
                    cfg_strength=self.CFG,
                )
                audio = audios.float().cpu()[0]
                torchaudio.save(clip_path, audio, self.seq_cfg.sampling_rate)
                results.append(AudioClipResult(path=clip_path, actual_duration_s=duration, cost=0.0))
                print(f"  Shot {shot_index} part {part_idx+1}/{len(durations)} -> {clip_name} ({duration}s)")
            except Exception as e:
                print(f"  Error MMAudio {clip_name}: {e}")
                return []
        return results


class LocalMusicGenStrategy(AudioStrategy):
    """MusicGen (facebook/musicgen-large) via transformers, run locally. Music shots."""

    name = "musicgen-local"
    uses_music_field = True
    MIN_DURATION = 2
    MAX_DURATION = 30
    MODEL_ID = "facebook/musicgen-large"

    def __init__(self, device=None):
        self.device = device or pick_device()
        self._loaded = False

    def _target_durations(self, target_s: float) -> list[float]:
        return _split_duration(target_s, self.MIN_DURATION, self.MAX_DURATION)

    def _load(self):
        if self._loaded:
            return
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        dtype = torch.float16 if self.device == "mps" else torch.float32
        print(f"  [MusicGen] loading {self.MODEL_ID} on {self.device} ({dtype})...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        model = MusicgenForConditionalGeneration.from_pretrained(self.MODEL_ID, dtype=dtype)
        self.model = model.to(self.device).eval()
        self.sr = model.config.audio_encoder.sampling_rate
        self._loaded = True
        print("  [MusicGen] ready.")

    def generate(self, sound_description, audio_dir, shot_index, target_duration_s, seed=None):
        if not sound_description:
            return []
        self._load()
        import torchaudio

        durations = self._target_durations(target_duration_s)
        results = []
        for part_idx, duration in enumerate(durations):
            clip_name = (f"{shot_index:04d}.flac" if len(durations) == 1
                         else f"{shot_index:04d}-{part_idx + 1:02d}.flac")
            clip_path = os.path.join(audio_dir, clip_name)
            try:
                # MusicGen: 50 tokens/sec; max_new_tokens controls length.
                max_new = int(round(duration * 50))
                inputs = self.processor(text=[sound_description], padding=True, return_tensors="pt").to(self.device)
                if seed is not None:
                    torch.manual_seed(seed + part_idx)
                with torch.no_grad():
                    wav = self.model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=max_new)
                audio = wav[0].float().cpu()  # (channels, samples)
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                torchaudio.save(clip_path, audio, self.sr)
                results.append(AudioClipResult(path=clip_path, actual_duration_s=duration, cost=0.0))
                print(f"  Shot {shot_index} part {part_idx+1}/{len(durations)} -> {clip_name} ({duration}s)")
            except Exception as e:
                print(f"  Error MusicGen {clip_name}: {e}")
                return []
        return results
