"""Tests for /api/blog_selections endpoints in tools/serve.py.

Spins up the actual handler on an ephemeral port to exercise the HTTP layer.
"""
import http.client
import json
import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import serve  # noqa: E402


@pytest.fixture
def server(tmp_path, monkeypatch):
    selections = tmp_path / "selections.json"
    selections.write_text(json.dumps({"entries": []}))
    monkeypatch.setattr(serve, "SELECTIONS_PATH", str(selections))

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    httpd = serve.ThreadedHTTPServer(("127.0.0.1", port), serve.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    yield port, str(selections)
    httpd.shutdown()


@pytest.mark.req("REQ-010")
def test_get_returns_json(server):
    port, _ = server
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("GET", "/api/blog_selections")
    r = c.getresponse()
    assert r.status == 200
    assert r.getheader("Content-Type") == "application/json"
    assert json.loads(r.read())["entries"] == []


@pytest.mark.req("REQ-011")
def test_post_upserts(server):
    port, path = server
    body = json.dumps({
        "film": "star_wars_iv_v2", "shot_idx": 42,
        "strategy_a": "runpod-wan", "strategy_b": "runpod-vace",
        "label": "x", "tag": "funny_halluc",
    })
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("POST", "/api/blog_selections", body=body,
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200
    on_disk = json.load(open(path))
    assert on_disk["entries"][0]["shot_idx"] == 42


@pytest.mark.req("REQ-012")
def test_post_invalid_returns_400_and_preserves_disk(server):
    port, path = server
    before = open(path).read()
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("POST", "/api/blog_selections", body="{}",
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 400
    assert open(path).read() == before


@pytest.mark.req("REQ-013")
def test_delete_removes(server):
    port, path = server
    # Seed one entry
    with open(path, "w") as f:
        json.dump({"entries": [{
            "film": "star_wars_iv_v2", "shot_idx": 42,
            "strategy_a": "runpod-wan", "strategy_b": "runpod-vace",
            "label": "x",
        }]}, f)
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("DELETE", "/api/blog_selections?film=star_wars_iv_v2&shot_idx=42")
    assert c.getresponse().status == 200
    assert json.load(open(path))["entries"] == []


@pytest.mark.req("REQ-013")
def test_delete_missing_returns_404(server):
    port, _ = server
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("DELETE", "/api/blog_selections?film=nope&shot_idx=1")
    assert c.getresponse().status == 404
