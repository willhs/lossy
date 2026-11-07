"""The Gemini backend switch: AI Studio (API key) vs Vertex AI (ADC)."""

from unittest.mock import MagicMock, patch

import pytest

import config


def test_aistudio_backend_requires_api_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_BACKEND", "aistudio")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        config.gemini_client()


def test_aistudio_backend_builds_key_client(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_BACKEND", "aistudio")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with patch("google.genai.Client") as client:
        config.gemini_client(1000)
    assert client.call_args.kwargs["api_key"] == "k"
    assert "vertexai" not in client.call_args.kwargs


def test_vertex_backend_uses_adc_and_pins_quota_project(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_BACKEND", "vertex")
    monkeypatch.setattr(config, "VERTEX_PROJECT", "my-proj")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # no key needed
    creds = MagicMock()
    with patch("google.auth.default", return_value=(creds, "other-proj")), \
            patch("google.genai.Client") as client:
        config.gemini_client()
    creds.with_quota_project.assert_called_once_with("my-proj")
    kw = client.call_args.kwargs
    assert kw["vertexai"] is True and kw["project"] == "my-proj"


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_BACKEND", "bogus")
    with pytest.raises(SystemExit):
        config.gemini_client()
