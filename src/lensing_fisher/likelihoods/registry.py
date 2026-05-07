"""Backend registry: maps a backend name (``clipy``, ``candl``) to a factory
that takes a kwargs dict from YAML and returns a likelihood object."""

from typing import Callable, Any

LIKELIHOOD_FACTORIES: dict[str, Callable[..., Any]] = {}


def register(name: str):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in LIKELIHOOD_FACTORIES:
            raise ValueError(f"Likelihood backend {name!r} already registered")
        LIKELIHOOD_FACTORIES[name] = fn
        return fn
    return decorator


def get_factory(name: str) -> Callable[..., Any]:
    if name not in LIKELIHOOD_FACTORIES:
        raise KeyError(
            f"Unknown likelihood backend {name!r}. "
            f"Registered: {sorted(LIKELIHOOD_FACTORIES)}"
        )
    return LIKELIHOOD_FACTORIES[name]
