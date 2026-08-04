"""YAML config loader with environment-variable expansion and ``!include``."""

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


class _IncludeLoader(yaml.SafeLoader):
    """SafeLoader supporting ``!include <path>`` and ``!include <path>#<key>``.

    Paths are resolved relative to the including file. The ``#key`` form splices
    in a single top-level key, which is how a Fisher config pulls in a shared
    dataset definition::

        likelihoods: !include datasets/spa.yaml#likelihoods

    The same file is then loadable by the Cobaya wrapper, so the chain and the
    Fisher cannot drift apart.
    """

    def __init__(self, stream):
        self._root = Path(getattr(stream, "name", ".")).parent
        super().__init__(stream)


def _include(loader: _IncludeLoader, node: yaml.Node):
    spec = str(loader.construct_scalar(node))
    path_part, _, key = spec.partition("#")

    path = Path(path_part)
    if not path.is_absolute():
        path = loader._root / path
    if not path.exists():
        raise FileNotFoundError(f"!include target not found: {path}")

    with open(path) as f:
        data = yaml.load(f, _IncludeLoader)

    if key:
        if not isinstance(data, dict) or key not in data:
            raise KeyError(f"!include {path_part}: no top-level key {key!r}")
        return data[key]
    return data


_IncludeLoader.add_constructor("!include", _include)


REQUIRED_TOP_LEVEL = ("likelihoods", "cosmology", "camb", "bins", "output")


def load_raw(path: str | os.PathLike) -> dict:
    """Load a YAML config with ``!include`` and ``${VAR}`` expansion, no validation.

    Used when only part of a config is wanted — e.g. the Cobaya wrapper reading
    the ``likelihoods:`` block out of a Fisher config, or a dataset-only file
    shared between the two.
    """
    with open(path) as f:
        cfg = yaml.load(f, _IncludeLoader)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path!s} did not parse to a mapping")
    return _expand_env(cfg)


def load(path: str | os.PathLike) -> dict:
    """Load and validate a Fisher driver YAML config.

    Returns a plain dict with environment variables expanded. Raises if any
    top-level required section is missing.
    """
    cfg = load_raw(path)

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"config {path!s} missing required top-level keys: {missing}")

    return cfg
