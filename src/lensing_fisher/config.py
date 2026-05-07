"""YAML config loader with environment-variable expansion."""

import os
import re
from pathlib import Path

import yaml

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env(obj):
    """Recursively expand ``${VAR}`` in string values using os.environ."""
    if isinstance(obj, str):
        def repl(m: re.Match) -> str:
            var = m.group(1)
            if var not in os.environ:
                raise KeyError(f"Environment variable {var!r} referenced in config but not set")
            return os.environ[var]
        return _ENV_PATTERN.sub(repl, obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


REQUIRED_TOP_LEVEL = ("likelihoods", "cosmology", "camb", "bins", "output")


def load(path: str | os.PathLike) -> dict:
    """Load and validate a Fisher driver YAML config.

    Returns a plain dict with environment variables expanded. Raises if any
    top-level required section is missing.
    """
    text = Path(path).read_text()
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path!s} did not parse to a mapping")
    cfg = _expand_env(cfg)

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"config {path!s} missing required top-level keys: {missing}")

    return cfg
