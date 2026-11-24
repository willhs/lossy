"""Run `encode.py stage4` with Gemini reached through fal's any-llm endpoint.

Why: on 24 Nov 2026 the AI Studio key's project returned 429 with `quota_limit_value: 0` for
every model, and there are no Vertex credentials in the job. Stage 4 only needs text in and JSON
out, so this swaps the Gemini client for a stand-in that calls fal (`fal-ai/any-llm`, FAL_KEY).
`run_stage4` itself, its prompt, and the registry gate on names are unchanged.

    op run --env-file .env.tpl -- uv run python docs/research/0030-sw4-full-run-preflight/stage4_via_fal.py RUN_DIR [MODEL]

MODEL defaults to google/gemini-2.5-flash. It is recorded in speakers.json as "model".
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import fal_client  # noqa: E402

import encode  # noqa: E402


class _Response:
    def __init__(self, text):
        self.text = text


class _Models:
    def __init__(self, model):
        self.model = model

    def generate_content(self, model, contents, config):
        prompt = contents[0].parts[0].text
        last = None
        for attempt in range(4):
            try:
                out = fal_client.subscribe("fal-ai/any-llm", arguments={
                    "model": self.model,
                    "system_prompt": config.system_instruction,
                    "prompt": prompt,
                })
                text = (out.get("output") or "").strip()
                if text.startswith("```"):
                    text = text.strip("`").removeprefix("json").strip()
                return _Response(text)
            except Exception as e:  # network / 5xx: retry, then let the batch be skipped
                last = e
                time.sleep(2 * (attempt + 1))
        print(f"  fal any-llm failed: {last}")
        return _Response("")


class _Client:
    def __init__(self, model):
        self.models = _Models(model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output_dir")
    p.add_argument("model", nargs="?", default="google/gemini-2.5-flash")
    a = p.parse_args()
    encode.ENCODE_MODEL = a.model
    encode._gemini_client = lambda api_key=None: _Client(a.model)
    os.environ.setdefault("GEMINI_API_KEY", "unused")
    encode.run_stage4(argparse.Namespace(output_dir=a.output_dir))


if __name__ == "__main__":
    main()
