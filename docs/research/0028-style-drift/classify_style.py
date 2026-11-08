"""Classify the rendering style of one still per generated clip (research 0028).

Usage (from the repo root):
    LOSSY_GEMINI_BACKEND=vertex LOSSY_ENCODE_MODEL=gemini-3.1-flash-lite \
        uv run python docs/research/0028-style-drift/classify_style.py FRAMES_DIR OUT_JSONL

FRAMES_DIR holds one JPEG per clip (<clip-stem>.jpg, e.g. 0008-02.jpg).
Resumable: clips already in OUT_JSONL are skipped. Prints token usage at the end.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.getcwd())
import config  # noqa: E402
from google.genai import types  # noqa: E402

PROMPT = """You are labelling the RENDERING STYLE of a single video frame, not its content.
Pick exactly one style:
- photoreal: looks like live-action footage of real people/places shot on a camera
- 3d_animation: CG cartoon / Pixar-like / stylised 3D characters (exaggerated features, plastic skin)
- 2d_animation: hand-drawn or flat 2D cartoon
- other: anything else (painting, black frame, abstract, text card)
Also give confidence 0-1 and a reason of at most 12 words.
If the frame looks like a recognisable real TV show or film other than a sitcom set, name it in lookalike, else empty."""

SCHEMA = {
    "type": "object",
    "properties": {
        "style": {"type": "string", "enum": ["photoreal", "3d_animation", "2d_animation", "other"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "lookalike": {"type": "string"},
    },
    "required": ["style", "confidence", "reason", "lookalike"],
}


def main(frames_dir: str, out_path: str) -> None:
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["clip"] for line in f if line.strip()}
    todo = sorted(p for p in os.listdir(frames_dir) if p.endswith(".jpg") and p[:-4] not in done)
    client = config.gemini_client(timeout_ms=60_000)
    usage = {"in": 0, "out": 0}

    def one(name: str) -> dict:
        with open(os.path.join(frames_dir, name), "rb") as f:
            img = f.read()
        err: Exception | None = None
        for _ in range(3):
            try:
                resp = client.models.generate_content(
                    model=config.ENCODE_MODEL,
                    contents=[types.Part.from_bytes(data=img, mime_type="image/jpeg"), PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", response_schema=SCHEMA, temperature=0),
                )
                rec = json.loads(resp.text or "{}")
                m = resp.usage_metadata
                if m is not None:
                    usage["in"] += m.prompt_token_count or 0
                    usage["out"] += (m.candidates_token_count or 0) + (m.thoughts_token_count or 0)
                return {"clip": name[:-4], **rec}
            except Exception as e:  # retry transient API errors
                err = e
        return {"clip": name[:-4], "style": "error", "confidence": 0, "reason": str(err)[:200], "lookalike": ""}

    with ThreadPoolExecutor(8) as pool, open(out_path, "a") as out:
        for rec in pool.map(one, todo):
            out.write(json.dumps(rec) + "\n")
    print(f"classified {len(todo)} frames with {config.ENCODE_MODEL} via {config.GEMINI_BACKEND}; "
          f"tokens in={usage['in']} out={usage['out']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
