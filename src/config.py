"""Configuration loader for the Intelligent Face Tracker project.

All tunable parameters live in ``config.json`` at the project root. This
module loads that file once, validates presence of required sections, and
exposes convenient dotted access through the :class:`Config` class so the
rest of the code never hard-codes paths or thresholds.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict


class ConfigError(Exception):
    """Raised when the configuration file is missing or malformed."""


class Config:
    """Loads and provides structured access to ``config.json``.

    Example
    -------
    >>> cfg = Config("config.json")
    >>> cfg.get("detection", "frame_skip")
    2
    """

    REQUIRED_SECTIONS = (
        "video_source",
        "detection",
        "recognition",
        "tracking",
        "database",
        "storage",
        "logging",
        "processing",
    )

    def __init__(self, config_path: str = "config.json") -> None:
        self.config_path = config_path
        self._data: Dict[str, Any] = self._load(config_path)
        self._validate()
        # project_root is the directory containing config.json; used to
        # resolve all relative paths declared inside the config file.
        self.project_root = os.path.dirname(os.path.abspath(config_path)) or "."

    @staticmethod
    def _load(config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise ConfigError(f"Configuration file not found: {config_path}")
        try:
            # utf-8-sig tolerates files saved with a BOM (common on Windows)
            # while behaving identically for plain UTF-8.
            with open(config_path, "r", encoding="utf-8-sig") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc

    def _validate(self) -> None:
        missing = [s for s in self.REQUIRED_SECTIONS if s not in self._data]
        if missing:
            raise ConfigError(f"Missing required config sections: {missing}")

    def get(self, *keys: str, default: Any = None) -> Any:
        """Fetch a nested configuration value.

        ``cfg.get("detection", "frame_skip")`` returns
        ``config["detection"]["frame_skip"]`` or ``default`` if absent.
        """
        node: Any = self._data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def resolve_path(self, relative_path: str) -> str:
        """Resolve a config-declared path relative to the project root."""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.normpath(os.path.join(self.project_root, relative_path))

    @property
    def raw(self) -> Dict[str, Any]:
        """Return the full configuration dictionary (read-only use)."""
        return self._data
