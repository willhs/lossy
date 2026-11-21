"""Keyframe-first character continuity (research 0029, arm K, with its two fixes).

Idea: instead of asking the video model to remember a face, draw each shot's first
frame with an image editor that is shown the character's reference sheet, then let
LTX-2.5 run image-to-video from that frame. This module owns the body of knowledge
that makes that safe to run unattended; the strategy (``RunPodLtx25KeyframeStrategy``
in ``strategies_video.py``) only wires it to the pod.

The data shape:

* ``ShotRoute``: per shot, either a keyframe shot (``characters`` are the people
  whose sheets the editor sees) or a text shot with the reason it falls back. A
  keyframe route always names at least one character, a text route always a reason.
* ``Sheet``: one accepted character sheet (image + a face signature).
* ``KeyframeGenerator`` / ``FaceEmbedder``: the two things this needs from outside
  the process. The generator is an interface so the trial's fal klein call
  (``FalKleinGenerator``) and a later klein on the pod are interchangeable; the
  embedder is an interface so the distinctness gate is testable without models.

The two fixes the trial asked for (research 0029 "Problems found"):

1. Sheet wardrobe leaked into shots. Sheets are head-and-shoulders only, with no
   clothing in frame, AND the keyframe prompt puts the scene (which carries the
   shot's wardrobe) first and tells the editor to take only face, hair, skin, age
   and build from the sheet.
2. Generated sheets invented look-alikes (Jerry and Kramer: face cosine 0.78).
   Sheets are made in prominence order and each new one is compared with every
   accepted sheet; one that is too close is reseeded, and if it stays too close
   it is dropped (that character's shots use the text path) rather than shipped.

Nothing here touches RunPod directly. ``FalKleinGenerator`` makes paid fal calls;
``PodKleinGenerator`` runs the same model in ComfyUI through a ``ComfyRunner`` the
strategy builds over its pod session (research 0029, "Self-hosted klein on the pod").
"""
from __future__ import annotations

import enum
import hashlib
import json
import os
import tempfile
import time
import zlib
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

# Face cosine between two different characters' sheets above which the pair is
# "the same-looking person" (research 0029: Jerry/Kramer 0.78; K's cross-character
# mean 0.45; the doc says "over ~0.5"). SFace's own same-person line is ~0.36, so
# 0.5 is the looser of the two; a sheet only has to differ visibly, not be certain.
LOOKALIKE_COSINE = 0.5
SHEET_ATTEMPTS = 3              # first draw + two reseeds
SHEET_SIZE = (1024, 1024)       # the trial's 2x2 sheet
CAP_PER_SHOT = 2                # sheets the editor sees: the identity mixin's cap

SHEET_PROMPT = (
    "Character reference sheet: one photograph divided into a 2x2 grid on a plain mid-grey studio "
    "background, showing the same real person's head four times. Every view is framed tightly on "
    "the head and neck, so no clothing, jewellery or accessories are visible. Top left: "
    "front-facing. Top right: three-quarter view from the left. Bottom left: side profile. "
    "Bottom right: three-quarter view from the right. Neutral expression, soft even studio "
    "lighting, face and hair exactly as described. The person: {block} Photorealistic 35mm film "
    "photograph, natural skin texture, no text, no labels.")


class Fallback(str, enum.Enum):
    """Why a shot is decoded from text alone. The first three are known from the
    plan; the last three only happen while the keyframes are being made."""

    NO_CHARACTERS = "no_characters"          # nobody the shot map knows is in the shot
    DESIGN_CHARACTER = "design_character"    # a keep_name character (droid, Vader, trooper) is in it
    NO_DESCRIPTION = "no_description"        # a character has no description to draw a sheet from
    SHEET_UNAVAILABLE = "sheet_unavailable"  # a sheet could not be made, or stayed a look-alike
    KEYFRAME_FAILED = "keyframe_failed"      # the editor call failed or the budget was spent
    UPLOAD_FAILED = "upload_failed"          # the keyframe could not be copied to the pod


@dataclass(frozen=True)
class ShotRoute:
    shot: int
    characters: tuple[str, ...]
    fallback: Fallback | None

    def __post_init__(self):
        if (self.fallback is None) != bool(self.characters):
            raise ValueError("a keyframe route needs characters and a text route needs a reason "
                             f"(shot {self.shot}: {self.characters!r}, {self.fallback!r})")

    @property
    def keyframe(self) -> bool:
        return self.fallback is None


def plan_route(shot: int, names: list[str], chars_by_name: dict,
               cap: Callable[[list[str]], list[str]]) -> ShotRoute:
    """Decide one shot. ``names`` is every character the shot map puts in it.

    A shot is a keyframe shot only if every character in it is a person we can draw
    a sheet for. A design character anywhere in the shot sends the whole shot to
    text: the editor has no sheet for a droid, would invent one from words, and the
    video model would then be locked to that invention for the whole clip.
    ``cap`` picks the people whose sheets are shown, the same ones whose identity
    blocks the text prompt carries.
    """
    if not names:
        return ShotRoute(shot, (), Fallback.NO_CHARACTERS)
    chars = [chars_by_name.get(n) for n in names]
    if any(c is None for c in chars):
        return ShotRoute(shot, (), Fallback.NO_DESCRIPTION)
    if any(c.get("keep_name") for c in chars):
        return ShotRoute(shot, (), Fallback.DESIGN_CHARACTER)
    shown = tuple(cap(list(names)))[:CAP_PER_SHOT]
    if any(not (chars_by_name[n].get("description") or "").strip() for n in shown):
        return ShotRoute(shot, (), Fallback.NO_DESCRIPTION)
    return ShotRoute(shot, shown, None)


def sheet_prompt(description: str) -> str:
    """The sheet's text-to-image prompt, from the identity description only (never a name)."""
    return SHEET_PROMPT.format(block=description.strip())


def keyframe_prompt(video_prompt: str, n_sheets: int) -> str:
    """The editor prompt for a shot's first frame.

    The scene comes first, because it names the shot's wardrobe, and the sheets are
    reduced to "face, hair, skin, age and build". The trial put the clothes
    instruction last and many frames kept the sheet's grey t-shirt anyway.
    """
    if n_sheets == 1:
        who = "The person in this frame is the person in the reference sheet (image 1)."
    elif n_sheets == 2:
        who = ("The two people in this frame are the person in reference sheet 1 and the person "
               "in reference sheet 2; keep each one distinct.")
    else:
        raise ValueError(f"the editor is shown 1 or 2 sheets, got {n_sheets}")
    return (f"A single film frame, not a grid. Clothing, props, setting and lighting come only from "
            f"this scene: {video_prompt} {who} From the sheet take ONLY the face, hair, skin, age and "
            "build. Do not copy any clothing, background or layout from the sheet.")


# --------------------------------------------------------------- interfaces

class KeyframeError(Exception):
    """One image could not be made. The shot (or sheet attempt) falls back."""


class KeyframeBudgetExceeded(KeyframeError):
    """The generator's spend cap is reached; no further calls will be made."""


class KeyframeSetupError(Exception):
    """A prerequisite is missing (key, models). Stops the run before any pod exists."""


class KeyframeGenerator(Protocol):
    id: str   # stamped into cache keys, so a changed generator regenerates

    def preflight(self) -> None:
        """Raise KeyframeSetupError if a call could not possibly succeed. Free."""

    def make_sheet(self, prompt: str, seed: int, size: tuple[int, int]) -> bytes:
        """A text-to-image sheet (PNG bytes). Raises KeyframeError."""

    def make_keyframe(self, prompt: str, sheets: list[bytes], seed: int,
                      size: tuple[int, int]) -> bytes:
        """An edit of the given sheets into one frame (PNG bytes). Raises KeyframeError."""


class FaceEmbedder(Protocol):
    def preflight(self) -> None:
        """Raise KeyframeSetupError if the models are missing. Free."""

    def faces(self, png: bytes) -> list[np.ndarray]:
        """Unit-length embedding of every face found in the image (empty if none)."""


# ------------------------------------------------------------------- sheets

@dataclass(frozen=True)
class Sheet:
    name: str
    png: bytes
    seed: int
    signature: np.ndarray        # unit centroid of the sheet's face embeddings
    max_cosine: float | None     # to the most similar earlier accepted sheet (None: first)


@dataclass(frozen=True)
class SheetSet:
    sheets: dict[str, Sheet]
    dropped: dict[str, str]      # name -> why no sheet was kept

    def __contains__(self, name: str) -> bool:
        return name in self.sheets


def sheet_seed(name: str, attempt: int) -> int:
    """Stable per character, a new draw per attempt."""
    return zlib.crc32(name.encode()) % 1_000_000 + attempt


def _signature(embedder: FaceEmbedder, png: bytes) -> np.ndarray | None:
    faces = embedder.faces(png)
    if not faces:
        return None
    m = np.mean(faces, axis=0)
    return m / np.linalg.norm(m)


def cache_key(*parts: str) -> str:
    """What a cached image was made from; any part changing invalidates it."""
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


class DiskCache:
    """``<dir>/<stem>.png`` plus ``<stem>.json`` holding the key it was made from.

    A file whose key no longer matches (the description, prompt or generator
    changed after a re-encode) is treated as absent, never reused.
    """

    def __init__(self, directory: str):
        self.dir = directory

    def path(self, stem: str) -> str:
        return os.path.join(self.dir, f"{stem}.png")

    def load(self, stem: str, key: str) -> tuple[bytes, dict] | None:
        meta_path = os.path.join(self.dir, f"{stem}.json")
        try:
            meta = json.load(open(meta_path))
            png = open(self.path(stem), "rb").read()
        except (OSError, ValueError):
            return None
        return (png, meta) if meta.get("key") == key and png else None

    def save(self, stem: str, key: str, png: bytes, **meta) -> None:
        os.makedirs(self.dir, exist_ok=True)
        with open(self.path(stem), "wb") as f:
            f.write(png)
        with open(os.path.join(self.dir, f"{stem}.json"), "w") as f:
            json.dump({"key": key, **meta}, f, indent=1)


def make_sheets(names: list[str], descriptions: dict[str, str], generator: KeyframeGenerator,
                embedder: FaceEmbedder, cache: DiskCache | None = None, *,
                threshold: float = LOOKALIKE_COSINE, attempts: int = SHEET_ATTEMPTS,
                log: Callable[[str], None] = print) -> SheetSet:
    """One sheet per name, in the order given (most prominent first), each distinct.

    A cached sheet is reused as is. A new one is compared with every accepted sheet;
    over ``threshold`` it is redrawn with a new seed, up to ``attempts`` draws, and
    then dropped. A draw with no detectable face, or a failed call, also uses up an
    attempt. A spent budget ends the loop for everyone left.
    """
    accepted: dict[str, Sheet] = {}
    dropped: dict[str, str] = {}
    out_of_budget = False
    for name in names:
        prompt = sheet_prompt(descriptions[name])
        key = cache_key(generator.id, prompt)
        cached = cache.load(f"{name}_sheet", key) if cache else None
        if cached:
            sig = _signature(embedder, cached[0])
            if sig is not None:
                accepted[name] = Sheet(name, cached[0], cached[1].get("seed", 0), sig,
                                       cached[1].get("max_cosine"))
                log(f"  sheet {name}: cached")
                continue
        if out_of_budget:
            dropped[name] = "budget spent before this sheet"
            continue
        why = "no draw attempted"
        for attempt in range(attempts):
            seed = sheet_seed(name, attempt)
            try:
                png = generator.make_sheet(prompt, seed, SHEET_SIZE)
            except KeyframeBudgetExceeded as e:
                out_of_budget, why = True, f"budget: {e}"
                break
            except KeyframeError as e:
                why = f"generation failed: {e}"
                continue
            sig = _signature(embedder, png)
            if sig is None:
                why = "no face found in the sheet"
                continue
            worst = max(((float(np.dot(sig, o.signature)), o.name) for o in accepted.values()),
                        default=(None, None))
            if worst[0] is not None and worst[0] > threshold:
                why = f"look-alike of {worst[1]} (face cosine {worst[0]:.2f} > {threshold})"
                log(f"  sheet {name}: attempt {attempt + 1} {why}; reseeding")
                continue
            accepted[name] = Sheet(name, png, seed, sig, worst[0])
            if cache:
                cache.save(f"{name}_sheet", key, png, seed=seed, attempt=attempt + 1,
                           max_cosine=worst[0], generator=generator.id)
            log(f"  sheet {name}: ok (seed {seed}, attempt {attempt + 1}"
                f"{'' if worst[0] is None else f', closest sheet {worst[0]:.2f}'})")
            break
        if name not in accepted:
            dropped[name] = why
            log(f"  sheet {name}: DROPPED ({why}); its shots use text")
    return SheetSet(accepted, dropped)


# ------------------------------------------------------------ fal klein (K)

class FalKleinGenerator:
    """The trial's generator: FLUX.2 [klein] 4B on fal (Apache-2.0 weights).

    Prices are fal list prices read 2026-11-15, with the edit counted on input and
    output megapixels (an upper bound). ``cap_usd`` is a hard stop on the estimated
    spend of this instance's calls: ``capped_run`` measures the RunPod balance only,
    so fal spend would otherwise be uncapped. Every call is appended to ``cost_log``.
    """

    id = "fal-flux2-klein-4b"
    T2I = "fal-ai/flux-2/klein/4b"
    EDIT = "fal-ai/flux-2/klein/4b/edit"
    PRICE_T2I_PER_MP = 0.005
    PRICE_EDIT_PER_MP = 0.01
    ATTEMPTS = 3

    def __init__(self, cap_usd: float, cost_log: str | None = None):
        self.cap_usd = cap_usd
        self.cost_log = cost_log
        # Spend already logged counts against the cap: capped_run restarts decode
        # after a failure, and each restart is a new process that would otherwise
        # get a fresh budget. To spend more on the same output dir, raise the cap.
        self.spent = self._logged_spend(cost_log)
        self._uploads: dict[str, str] = {}

    @staticmethod
    def _logged_spend(cost_log: str | None) -> float:
        try:
            return sum(json.loads(line)["usd"] for line in open(cost_log) if line.strip())
        except (OSError, TypeError, ValueError, KeyError):
            return 0.0

    @classmethod
    def price_sheet(cls, size=SHEET_SIZE) -> float:
        return cls.PRICE_T2I_PER_MP * size[0] * size[1] / 1e6

    @classmethod
    def price_keyframe(cls, n_sheets: int, size: tuple[int, int]) -> float:
        mp = size[0] * size[1] / 1e6 + n_sheets * SHEET_SIZE[0] * SHEET_SIZE[1] / 1e6
        return cls.PRICE_EDIT_PER_MP * mp

    def preflight(self) -> None:
        if not os.environ.get("FAL_KEY"):
            raise KeyframeSetupError("FAL_KEY is not set (the keyframe editor runs on fal).")
        try:
            import fal_client  # noqa: F401
            import httpx  # noqa: F401
        except ImportError as e:
            raise KeyframeSetupError(f"fal client not installed: {e}") from e

    def make_sheet(self, prompt: str, seed: int, size: tuple[int, int]) -> bytes:
        args = {"prompt": prompt, "seed": seed, "image_size": {"width": size[0], "height": size[1]},
                "output_format": "png"}
        return self._call(self.T2I, args, self.price_sheet(size), "sheet", seed)

    def make_keyframe(self, prompt: str, sheets: list[bytes], seed: int,
                      size: tuple[int, int]) -> bytes:
        price = self.price_keyframe(len(sheets), size)
        self._check_budget(price)
        args = {"prompt": prompt, "image_urls": [self._upload(s) for s in sheets], "seed": seed,
                "image_size": {"width": size[0], "height": size[1]}, "output_format": "png"}
        return self._call(self.EDIT, args, price, "keyframe", seed)

    def _check_budget(self, price: float) -> None:
        if self.spent + price > self.cap_usd:
            raise KeyframeBudgetExceeded(
                f"fal cap ${self.cap_usd:.2f} reached (${self.spent:.3f} spent, next call ${price:.3f})")

    def _upload(self, png: bytes) -> str:
        digest = hashlib.sha256(png).hexdigest()
        if digest not in self._uploads:
            import fal_client
            try:
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, f"sheet_{digest[:12]}.png")
                    with open(path, "wb") as f:
                        f.write(png)
                    self._uploads[digest] = fal_client.upload_file(path)
            except Exception as e:
                raise KeyframeError(f"sheet upload failed: {e}") from e
        return self._uploads[digest]

    def _call(self, model: str, args: dict, price: float, kind: str, seed: int) -> bytes:
        import fal_client
        import httpx
        self._check_budget(price)
        err = None
        for attempt in range(self.ATTEMPTS):
            try:
                res = fal_client.subscribe(model, arguments=args, with_logs=False)
                if any(res.get("has_nsfw_concepts") or []):
                    raise RuntimeError("safety checker blanked the image")
                r = httpx.get(res["images"][0]["url"], follow_redirects=True, timeout=120)
                r.raise_for_status()
                self.spent += price     # billed once, on the call that returned an image
                if self.cost_log:
                    os.makedirs(os.path.dirname(self.cost_log) or ".", exist_ok=True)
                    with open(self.cost_log, "a") as f:
                        f.write(json.dumps({"kind": kind, "model": model, "seed": seed,
                                            "usd": round(price, 4)}) + "\n")
                return r.content
            except Exception as e:   # transient API errors
                err = e
                time.sleep(5 * (attempt + 1))
        raise KeyframeError(f"{model} failed after {self.ATTEMPTS} tries: {err}")


# ------------------------------------------------------- klein on the pod

# The three files the official ComfyUI "FLUX.2 Klein 4B Distilled" edit template loads
# (Comfy-Org/workflow_templates, read 2026-11-22). Pinned by sha256 in strategies_video.
KLEIN_UNET = "flux-2-klein-4b-fp8.safetensors"
KLEIN_TEXT_ENCODER = "qwen_3_4b.safetensors"
KLEIN_VAE = "flux2-vae.safetensors"
KLEIN_STEPS = 4      # the distilled model's schedule (the template's Flux2Scheduler)


def klein_graph(prompt: str, seed: int, size: tuple[int, int], refs: Sequence[str] = ()) -> dict:
    """ComfyUI API graph for FLUX.2 klein 4B. No ``refs``: text-to-image (a sheet).
    One or two ``refs`` (filenames in the pod's ComfyUI input dir): a reference edit
    (a keyframe), each image VAE-encoded and attached with ``ReferenceLatent``.

    Node and input names are ComfyUI v0.39.2's (the version the pod upgrades to). CFG
    is 1 in the template, so ``BasicGuider`` (no negative) samples the same thing.
    """
    w, h = size
    if w % 16 or h % 16:
        raise ValueError(f"klein latents need sides divisible by 16, got {w}x{h}")
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": KLEIN_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": KLEIN_TEXT_ENCODER, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "Flux2Scheduler", "inputs": {"steps": KLEIN_STEPS, "width": w, "height": h}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }
    cond = ["4", 0]
    for i, name in enumerate(refs):
        load, enc, ref = f"2{i}0", f"2{i}1", f"2{i}2"
        g[load] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[enc] = {"class_type": "VAEEncode", "inputs": {"pixels": [load, 0], "vae": ["3", 0]}}
        g[ref] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": cond, "latent": [enc, 0]}}
        cond = [ref, 0]
    g["9"] = {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": cond}}
    g["10"] = {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["8", 0], "guider": ["9", 0], "sampler": ["7", 0],
                          "sigmas": ["6", 0], "latent_image": ["5", 0]}}
    g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}}
    g["12"] = {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "klein"}}
    return g


class ComfyRunner(Protocol):
    """What the pod generator needs from a pod. The strategy adapts its RunPodSession."""

    def ensure_ready(self) -> None:
        """Boot the pod and load the weights if not done yet. Raises if it cannot."""

    def upload(self, png: bytes) -> str | None:
        """Put an image in ComfyUI's input dir; its filename, or None on failure."""

    def run(self, graph: dict) -> bytes | None:
        """Run a graph to completion; the first saved image's bytes, or None on failure."""


class PodKleinGenerator:
    """FLUX.2 klein 4B in ComfyUI on the same RunPod pod that then runs LTX-2.5.

    The pod is brought up on the first call, not on construction, so a run whose
    sheets and keyframes are all cached boots nothing here. ``used_pod`` tells the
    strategy to unload klein before LTX loads. Every image's wall time goes to
    ``timing_log`` (the pod bills by the second; there is no per-call price).
    """

    id = "pod-flux2-klein-4b-fp8"

    def __init__(self, runner: ComfyRunner, timing_log: str | None = None):
        self.runner = runner
        self.timing_log = timing_log
        self.used_pod = False
        self._uploads: dict[str, str] = {}

    def preflight(self) -> None:
        if not os.environ.get("RUNPOD_API_KEY"):
            raise KeyframeSetupError("RUNPOD_API_KEY is not set (the keyframe editor runs on the pod).")

    def _ready(self) -> None:
        if not self.used_pod:
            self.runner.ensure_ready()
            self.used_pod = True

    def make_sheet(self, prompt: str, seed: int, size: tuple[int, int]) -> bytes:
        self._ready()
        return self._run(klein_graph(prompt, seed, size), "sheet", seed)

    def make_keyframe(self, prompt: str, sheets: list[bytes], seed: int,
                      size: tuple[int, int]) -> bytes:
        self._ready()
        return self._run(klein_graph(prompt, seed, size, [self._upload(s) for s in sheets]),
                         "keyframe", seed)

    def _upload(self, png: bytes) -> str:
        digest = hashlib.sha256(png).hexdigest()
        if digest not in self._uploads:
            name = self.runner.upload(png)
            if not name:
                raise KeyframeError("sheet upload to the pod failed")
            self._uploads[digest] = name
        return self._uploads[digest]

    def _run(self, graph: dict, kind: str, seed: int) -> bytes:
        t0 = time.time()
        try:
            png = self.runner.run(graph)
        except Exception as e:   # a dropped connection to the pod: this image only
            raise KeyframeError(f"klein on the pod failed: {e}") from e
        if not png:
            raise KeyframeError("klein on the pod returned no image")
        if self.timing_log:
            os.makedirs(os.path.dirname(self.timing_log) or ".", exist_ok=True)
            with open(self.timing_log, "a") as f:
                f.write(json.dumps({"kind": kind, "seed": seed,
                                    "seconds": round(time.time() - t0, 2)}) + "\n")
        return png


# ---------------------------------------------------- face embedder (the gate)

class OpenCVFaceEmbedder:
    """OpenCV-zoo YuNet + SFace on CPU, exactly as the trial's scorer measured faces.

    ``models_dir`` holds ``yunet.onnx`` and ``sface.onnx`` (opencv_zoo,
    face_detection_yunet and face_recognition_sface). Loaded on first use.
    """

    FILES = ("yunet.onnx", "sface.onnx")

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self._det = self._rec = None

    def preflight(self) -> None:
        missing = [f for f in self.FILES if not os.path.isfile(os.path.join(self.models_dir, f))]
        if missing:
            raise KeyframeSetupError(
                f"face models missing in {self.models_dir!r}: {', '.join(missing)}. The sheet "
                "distinctness gate needs them: download face_detection_yunet and "
                "face_recognition_sface from github.com/opencv/opencv_zoo and set "
                "LOSSY_FACE_MODELS to that directory.")

    def faces(self, png: bytes) -> list[np.ndarray]:
        import cv2
        if self._det is None:
            self._det = cv2.FaceDetectorYN.create(
                os.path.join(self.models_dir, "yunet.onnx"), "", (320, 320), 0.6)
            self._rec = cv2.FaceRecognizerSF.create(os.path.join(self.models_dir, "sface.onnx"), "")
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return []
        h, w = img.shape[:2]
        self._det.setInputSize((w, h))
        _, found = self._det.detect(img)
        out = []
        for f in (found if found is not None else []):
            emb = self._rec.feature(self._rec.alignCrop(img, f)).flatten()
            out.append(emb / np.linalg.norm(emb))
        return out


# --------------------------------------------------------------- keyframes

@dataclass(frozen=True)
class KeyframeJob:
    shot: int
    characters: tuple[str, ...]
    prompt: str          # the editor prompt (keyframe_prompt of the shot's first-part text)


def make_keyframes(jobs: list[KeyframeJob], sheets: SheetSet, generator: KeyframeGenerator,
                   cache: DiskCache, size: tuple[int, int],
                   log: Callable[[str], None] = print) -> tuple[dict[int, str], dict[int, Fallback]]:
    """First frames for the jobs: ``({shot: png path}, {shot: why it has none})``.

    A shot whose character has no sheet, or whose editor call fails, is simply left
    out of the paths: it decodes from text, as it would without this strategy. The
    seed is the shot index, so a rerun draws the same frame; a cached frame is reused
    only if its prompt, generator and sheets are unchanged. Once the budget is spent
    no further calls are made.
    """
    paths: dict[int, str] = {}
    failed: dict[int, Fallback] = {}
    out_of_budget = False
    for job in jobs:
        if any(c not in sheets for c in job.characters):
            failed[job.shot] = Fallback.SHEET_UNAVAILABLE
            continue
        pngs = [sheets.sheets[c].png for c in job.characters]
        key = cache_key(generator.id, job.prompt, *(hashlib.sha1(p).hexdigest() for p in pngs))
        stem = f"kf_{job.shot:04d}"
        if cache.load(stem, key):
            paths[job.shot] = cache.path(stem)
            continue
        if out_of_budget:
            failed[job.shot] = Fallback.KEYFRAME_FAILED
            continue
        try:
            png = generator.make_keyframe(job.prompt, pngs, job.shot, size)
        except KeyframeBudgetExceeded as e:
            out_of_budget = True
            failed[job.shot] = Fallback.KEYFRAME_FAILED
            log(f"  keyframes: {e}; the remaining shots use text")
            continue
        except KeyframeError as e:
            failed[job.shot] = Fallback.KEYFRAME_FAILED
            log(f"  keyframe {job.shot}: {e}; this shot uses text")
            continue
        cache.save(stem, key, png, shot=job.shot, characters=list(job.characters),
                   generator=generator.id)
        paths[job.shot] = cache.path(stem)
        log(f"  keyframe {job.shot:04d} ({'+'.join(job.characters)})")
    return paths, failed
