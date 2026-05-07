"""Lightweight import + config-load smoke tests. No CAMB / candl runtime required."""

import os
from pathlib import Path

import pytest


def test_fisher_import():
    from lensing_fisher import FisherMatrix
    assert FisherMatrix is not None


def test_registry_has_backends():
    from lensing_fisher.likelihoods import LIKELIHOOD_FACTORIES
    assert "clipy" in LIKELIHOOD_FACTORIES
    assert "candl" in LIKELIHOOD_FACTORIES


def test_config_loads_example(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANCK_CLIK_BASELINE", "/tmp/fake_planck")
    monkeypatch.setenv("MNU_HUNTER_ROOT", str(tmp_path))

    from lensing_fisher import config

    example = Path(__file__).parent.parent / "examples" / "lensing_sensitivity.yaml"
    cfg = config.load(example)

    for required in ("likelihoods", "cosmology", "camb", "bins", "output"):
        assert required in cfg

    assert cfg["bins"]["parametrization"] == "local_lens"
    assert cfg["likelihoods"]["spt3g_D1"]["backend"] == "candl"
    # env-var expansion happened
    assert "/tmp/fake_planck" in cfg["likelihoods"]["planck_highl_ttteee"]["clik_path"]


def test_config_missing_section_raises(tmp_path):
    from lensing_fisher import config

    bad = tmp_path / "bad.yaml"
    bad.write_text("likelihoods: {}\ncosmology: {}\n")
    with pytest.raises(ValueError, match="missing required top-level keys"):
        config.load(bad)


def test_fisher_matrix_serial_quadratic():
    """FisherMatrix on a quadratic log-likelihood should recover the Hessian."""
    import numpy as np
    from lensing_fisher import FisherMatrix

    # logL = -0.5 * sum_i ((x_i - mu_i) / sigma_i)^2 → Fisher diag = 1 / sigma_i^2
    sigmas = np.array([1.0, 2.0, 0.5])
    fid = {"a": 0.0, "b": 0.0, "c": 0.0}

    def logl(p):
        x = np.array([p["a"], p["b"], p["c"]])
        return -0.5 * float(np.sum((x / sigmas) ** 2))

    fm = FisherMatrix(logl, fid, use_central_differences=True)
    F = fm.compute_fisher_matrix(step_size=np.array([0.01, 0.01, 0.01]),
                                 adaptive=False, relative=False)
    expected = np.diag(1.0 / sigmas**2)
    assert np.allclose(F, expected, atol=1e-6)
