"""candl likelihood factory."""

import importlib

from .registry import register


def _resolve_dotted(path: str):
    """Resolve a dotted attribute path like ``spt_candl_data.SPT3G_D1_TnE_lite``."""
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(
            f"dataset must be a dotted path like 'spt_candl_data.SPT3G_D1_TnE_lite', got {path!r}"
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise AttributeError(f"{module_name!r} has no attribute {attr!r}") from e


@register("candl")
def make_candl_likelihood(*, dataset: str, **kwargs):
    """Build a ``candl.Like(<dataset>)`` instance.

    Parameters
    ----------
    dataset
        Dotted path to the dataset object, e.g.
        ``"spt_candl_data.SPT3G_D1_TnE_lite"`` or ``"candl_data.ACT_DR6_TTTEEE"``.
    """
    import candl  # local import: candl is optional

    data_obj = _resolve_dotted(dataset)
    return candl.Like(data_obj, **kwargs)
