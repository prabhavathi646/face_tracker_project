"""Tests for configuration loading and validation."""

import json
import os

import pytest

from src.config import Config, ConfigError


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_load_real_config():
    """The shipped config.json must load and expose expected values."""
    cfg = Config(os.path.join(PROJECT_ROOT, "config.json"))
    assert cfg.get("detection", "frame_skip") == 2
    assert cfg.get("recognition", "similarity_threshold") > 0
    assert cfg.get("video_source", "type") in ("file", "rtsp")


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        Config(str(tmp_path / "nope.json"))


def test_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ConfigError):
        Config(str(bad))


def test_utf8_bom_is_tolerated(tmp_path):
    """Windows editors often save JSON with a BOM; loading must still work."""
    import os as _os
    good = _os.path.join(PROJECT_ROOT, "config.json")
    with open(good, "rb") as src:
        payload = src.read()
    bom_file = tmp_path / "bom.json"
    bom_file.write_bytes(b"\xef\xbb\xbf" + payload)
    cfg = Config(str(bom_file))
    assert cfg.get("detection", "frame_skip") == 2


def test_missing_section_raises(tmp_path):
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"detection": {"frame_skip": 1}}), encoding="utf-8")
    with pytest.raises(ConfigError):
        Config(str(incomplete))


def test_get_default_returns_default(config):
    assert config.get("detection", "does_not_exist", default=99) == 99


def test_resolve_path_makes_absolute(config):
    resolved = config.resolve_path("data/sample_video.mp4")
    assert os.path.isabs(resolved)