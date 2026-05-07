"""clipy (Planck clik) likelihood factory."""

from .registry import register


@register("clipy")
def make_clipy_likelihood(*, clik_path: str, all_priors: bool = True, crop=None, **kwargs):
    """Build a ``clipy.clik_candl`` instance.

    Parameters
    ----------
    clik_path
        Path to the .clik directory.
    all_priors
        Forwarded to clipy.
    crop
        Optional list of crop strings (e.g. ``["crop TT 0 1000 strict"]``).
    """
    import clipy  # local import: clipy is optional

    extra = {}
    if crop is not None:
        extra["crop"] = list(crop)
    extra.update(kwargs)
    return clipy.clik_candl(clik_path, all_priors=all_priors, **extra)
