"""Load configuration: YAML defaults (config/defaults.yaml) overridden by
runtime settings (data/settings.json), per SHOULDER_SURFING_ALERT.md §9."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = REPO_ROOT / "config" / "defaults.yaml"
SETTINGS_PATH = REPO_ROOT / "data" / "settings.json"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict[str, Any]:
    """Load config/defaults.yaml, then apply data/settings.json overrides if present."""
    with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        config = _deep_merge(config, overrides)

    return config


def save_settings(overrides: dict[str, Any]) -> None:
    """Persist runtime overrides to data/settings.json (dashboard writes here)."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    merged = _deep_merge(existing, overrides)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)