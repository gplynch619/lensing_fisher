"""Parameter ties, and the agreement between the Fisher's and Cobaya's version.

A_planck is tied to A_act — one shared calibration across Planck and ACT. The
Fisher expresses this in the dataset YAML and Cobaya expresses it as
``value: 'lambda A_act: A_act'`` in the chain YAML. Those are two files, so they
can drift; if they do, the chain and the Fisher silently vary different numbers
of parameters and sigma(A_template) stops being comparable to the kernel.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from lensing_fisher import config
from lensing_fisher.driver import assemble_parameters
from lensing_fisher.likelihoods import combined_loglike

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(autouse=True)
def _fake_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PLANCK_CLIK_BASELINE", "/tmp/fake_planck")
    monkeypatch.setenv("MNU_HUNTER_ROOT", str(tmp_path))
    monkeypatch.setenv("LENSING_FISHER_ROOT", str(EXAMPLES.parent))


COSMO = {"fiducial": {"H0": 67.0, "tau": 0.054}, "step_sizes_relative": {"H0": 0.01, "tau": 0.01}}
NUISANCE = {"A_planck": 1.0, "A_act": 1.0, "P_act": 1.0}
EDGES = np.array([2.0, 100.0, 500.0])


def test_tied_parameter_is_not_in_the_fisher_vector():
    """A tied parameter would make the matrix singular if it kept a row."""
    free = assemble_parameters(COSMO, {}, NUISANCE, EDGES, tied={"A_planck": "A_act"})
    assert "A_planck" not in free.names
    assert free.nuisance_names == ["A_act", "P_act"]

    untied = assemble_parameters(COSMO, {}, NUISANCE, EDGES)
    assert "A_planck" in untied.names
    assert len(untied.names) == len(free.names) + 1


def test_unknown_tie_target_raises():
    with pytest.raises(ValueError, match="tied_parameters references"):
        assemble_parameters(COSMO, {}, NUISANCE, EDGES, tied={"A_nope": "A_act"})


def test_combined_loglike_fills_the_tied_value():
    seen = []

    class _Like:
        def log_like(self, pars):
            seen.append(dict(pars))
            return -1.0

    def fake_params_to_logl(lk, theory):
        return lambda pars: lk.log_like(pars)

    import candl.tools
    original = candl.tools.get_params_to_logl_func
    candl.tools.get_params_to_logl_func = fake_params_to_logl
    try:
        fn = combined_loglike([_Like()], theory=None, tied={"A_planck": "A_act"})
        fn({"A_act": 1.007, "P_act": 1.0})
    finally:
        candl.tools.get_params_to_logl_func = original

    assert seen[0]["A_planck"] == 1.007, "tied parameter not filled in from its source"


@pytest.mark.parametrize("dataset,chain", [
    ("up_planck_act_spt", "chain_up_planck_act_spt"),
])
def test_dataset_tie_matches_the_cobaya_chain(dataset, chain):
    ties = config.load_raw(EXAMPLES / "datasets" / f"{dataset}.yaml")["tied_parameters"]
    text = (EXAMPLES / f"{chain}.yaml").read_text()

    for target, source in ties.items():
        # Cobaya form:  A_planck:\n    value: 'lambda A_act: A_act'
        pattern = rf"{target}:\s*\n\s*value:\s*'lambda {source}:\s*{source}'"
        assert re.search(pattern, text), (
            f"{chain}.yaml does not tie {target} to {source} the way "
            f"{dataset}.yaml does"
        )
        # ... and the tied parameter must not also carry a prior, which would
        # make it free again on the chain side only.
        assert not re.search(rf"{target}:\s*\n\s*prior:", text)


def test_fisher_config_pulls_the_tie_from_the_dataset():
    """One source of truth: the tie travels with the data definition."""
    cfg = config.load(EXAMPLES / "fisher_spa.yaml")
    shared = config.load_raw(EXAMPLES / "datasets" / "spa.yaml")
    assert cfg["tied_parameters"] == shared["tied_parameters"] == {"A_planck": "A_act"}


def test_planck_only_set_has_no_tie():
    """UP-P has no ACT, so A_planck is genuinely free there."""
    up = config.load_raw(EXAMPLES / "datasets" / "up_planck.yaml")
    assert "tied_parameters" not in up
    text = (EXAMPLES / "chain_up_planck.yaml").read_text()
    assert re.search(r"A_planck:\s*\n\s*prior:", text)
    assert "A_act" not in text
