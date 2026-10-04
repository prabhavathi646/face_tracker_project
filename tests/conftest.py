"""Shared pytest fixtures for the face tracker test suite."""

import os
import sys

import numpy as np
import pytest

# Make the project root importable when running pytest from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config  # noqa: E402
from src.database import Database  # noqa: E402


@pytest.fixture
def config(tmp_path) -> Config:
    """Load the real config.json but redirect all storage to a temp dir."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = Config(os.path.join(project_root, "config.json"))
    cfg._data["database"]["path"] = str(tmp_path / "test.db")
    cfg._data["storage"]["entries_dir"] = str(tmp_path / "entries")
    cfg._data["storage"]["exits_dir"] = str(tmp_path / "exits")
    cfg._data["storage"]["registered_faces_dir"] = str(tmp_path / "registered")
    return cfg


@pytest.fixture
def db(config) -> Database:
    """Fresh in-memory-like SQLite database (file lives in tmp_path)."""
    database = Database(config.get("database", "path"))
    yield database
    database.close()


@pytest.fixture
def sample_crop() -> np.ndarray:
    """A small synthetic BGR 'face' image for save/write tests."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """A normalized random 512-d embedding (ArcFace-style)."""
    rng = np.random.default_rng(7)
    vec = rng.standard_normal(512).astype(np.float32)
    return vec / np.linalg.norm(vec)