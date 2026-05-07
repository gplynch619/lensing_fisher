from .local_lens import build_local_lens_theory

try:
    from .template_lensing import TemplateLensingCAMB
except ImportError:
    TemplateLensingCAMB = None  # cobaya/camb not installed; theory submodule still importable

__all__ = ["build_local_lens_theory", "TemplateLensingCAMB"]
