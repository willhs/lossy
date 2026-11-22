"""tools/regen_identity.py rewrites descriptions and nothing else."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import regen_identity  # noqa: E402


class FakeModels:
    def __init__(self):
        self.prompts = []

    def generate_content(self, model, contents, config):
        text = contents[0].parts[0].text
        self.prompts.append(text)
        return SimpleNamespace(text="A real man, mid-30s, natural skin texture.")


class FakeTypes:
    class Content:
        def __init__(self, role, parts):
            self.parts = parts

    class Part:
        @staticmethod
        def from_text(text):
            return SimpleNamespace(text=text)

    class GenerateContentConfig:
        def __init__(self, **kw):
            pass


def _data():
    return {"characters": [
        {"name": "han_solo", "display_name": "Han Solo", "description": "old",
         "shots": [1, 2], "keep_name": True},
        {"name": "ghost", "display_name": "Ghost", "description": "keep me", "shots": [9]},
    ]}


def test_only_description_changes_and_input_is_not_mutated():
    data = _data()
    models = FakeModels()
    client = SimpleNamespace(models=models)
    out = regen_identity.regenerate(client, FakeTypes, "m", data,
                                    {1: "a man", 2: "a man again"}, "SPEC")
    han = out["characters"][0]
    assert han["description"].startswith("A real man")
    assert han["shots"] == [1, 2] and han["keep_name"] is True and han["name"] == "han_solo"
    assert data["characters"][0]["description"] == "old"
    assert models.prompts[0].endswith("Write SPEC")


def test_character_with_no_subjects_keeps_its_description_and_makes_no_call():
    models = FakeModels()
    out = regen_identity.regenerate(SimpleNamespace(models=models), FakeTypes, "m",
                                    _data(), {1: "a man"}, "SPEC")
    assert out["characters"][1]["description"] == "keep me"
    assert len(models.prompts) == 1


def test_sample_shots_spreads_across_the_whole_list():
    picked = regen_identity.sample_shots(list(range(1000)), 20)
    assert len(picked) == 20 and picked[0] == 0 and picked[-1] == 999
    assert regen_identity.sample_shots([5, 3], 20) == [3, 5]
