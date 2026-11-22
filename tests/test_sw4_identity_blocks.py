"""The SW4 photographic spec's Chewbacca and Leia blocks (research 0030, 23 Nov test)."""
import json
import re
from pathlib import Path

SPEC = (Path(__file__).parent.parent / "docs/research/0030-sw4-full-run-preflight/data"
        / "sw4_characters_photographic.json")


def _block(name):
    chars = {c["name"]: c for c in json.loads(SPEC.read_text())["characters"]}
    return chars[name]["description"].lower()


def test_chewbacca_reads_as_wookiee_not_ape():
    d = _block("chewbacca")
    assert "wookiee" in d and "bandolier" in d
    for word in ("ape", "primate", "mane", "simian", "monkey", "gorilla", "beast", "claw"):
        assert not re.search(rf"\b{word}", d), word


def test_leia_is_covered_to_the_neck():
    d = _block("princess_leia_organa")
    assert "white" in d and "long-sleeved" in d and "high" in d and "neckline" in d
    assert "shoulders" in d and "covered" in d
    assert "bare" not in d
