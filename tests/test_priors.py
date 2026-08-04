"""Internal likelihood priors, and the tau double-count they cause.

ACT DR6 and SPT-3G D1 both ship a stand-in ``tau`` prior applied inside
``log_like``. Any combination that also includes a low-ell EE likelihood must
drop them or tau is counted two or three times over, which tightens sigma(A_s)
and so biases the lensing amplitude.
"""

import numpy as np
import pytest

from lensing_fisher import config
from lensing_fisher.likelihoods import drop_parameter_priors

from pathlib import Path

EXAMPLES = Path(__file__).parent.parent / "examples"


class _Prior:
    def __init__(self, *par_names):
        self.par_names = list(par_names)


class _Like:
    def __init__(self, *priors):
        self.priors = list(priors)
        self.rebuilt = 0

    def init_priors(self):
        self.rebuilt += 1


def test_drop_rebuilds_the_compiled_priors():
    like = _Like(_Prior("tau"), _Prior("A_act"))
    drop_parameter_priors(like, ["tau"], label="act")
    assert like.rebuilt == 1


def test_drops_only_the_named_prior():
    like = _Like(_Prior("tau"), _Prior("A_act"))
    drop_parameter_priors(like, ["tau"], label="act")
    assert [p.par_names for p in like.priors] == [["A_act"]]


def test_unmatched_name_raises_rather_than_no_op():
    """A drop_priors that quietly does nothing looks identical to one that worked."""
    like = _Like(_Prior("A_act"))
    with pytest.raises(ValueError, match="carries no internal prior"):
        drop_parameter_priors(like, ["tau"], label="act")


def test_joint_prior_is_not_silently_gutted():
    like = _Like(_Prior("tau", "A_act"))
    with pytest.raises(ValueError, match="joint over"):
        drop_parameter_priors(like, ["tau"], label="act")


@pytest.mark.parametrize("dataset", ["spa", "up_planck_act_spt"])
def test_lowl_ee_sets_drop_tau_on_every_candl_likelihood(dataset, monkeypatch, tmp_path):
    """The config-level guard: sroll2 present => no candl tau prior survives."""
    monkeypatch.setenv("PLANCK_CLIK_BASELINE", "/tmp/fake_planck")
    monkeypatch.setenv("MNU_HUNTER_ROOT", str(tmp_path))
    likes = config.load_raw(EXAMPLES / "datasets" / f"{dataset}.yaml")["likelihoods"]

    assert "planck_lowl_ee" in likes, "test premise: this set constrains tau from data"
    candl_entries = {n: s for n, s in likes.items() if s.get("backend") == "candl"}
    assert candl_entries, "test premise: this set has candl likelihoods"
    for name, spec in candl_entries.items():
        assert "tau" in spec.get("drop_priors", []), (
            f"{dataset}: {name} keeps its internal tau prior alongside sroll2"
        )


def test_planck_only_set_needs_no_dropping(monkeypatch, tmp_path):
    monkeypatch.setenv("PLANCK_CLIK_BASELINE", "/tmp/fake_planck")
    monkeypatch.setenv("MNU_HUNTER_ROOT", str(tmp_path))
    likes = config.load_raw(EXAMPLES / "datasets" / "up_planck.yaml")["likelihoods"]
    assert all(s["backend"] == "clipy" for s in likes.values())


def _tau_response(like, centre=0.0566, delta=0.01):
    """How much log_like moves for a shift in tau, with the bandpowers held fixed.

    tau is not a theory parameter here — ``Dl`` is passed in — so any response at
    all is the internal prior and nothing else.
    """
    lo, hi = int(like.ell_min), int(like.ell_max) + 1
    ell = np.arange(lo, hi)
    dl = {"ell": ell, "TT": np.full(ell.size, 1000.0), "TE": np.full(ell.size, 10.0),
          "EE": np.full(ell.size, 40.0), "BB": np.zeros(ell.size)}
    pars = {n: 1.0 for n in like.required_nuisance_parameters}
    pars["Dl"] = dl
    return (float(like.log_like({**pars, "tau": centre}))
            - float(like.log_like({**pars, "tau": centre + delta})))


@pytest.mark.parametrize("dataset,centre,sigma", [
    ("candl_data.ACT_DR6_TTTEEE", 0.0566, 0.0058),
    ("spt_candl_data.SPT3G_D1_TnE_lite", 0.0510, 0.0060),
])
def test_drop_priors_removes_the_tau_constraint_the_comments_quote(dataset, centre, sigma):
    """End to end through make_candl, which is the only supported entry point.

    Exercised as two separate likelihoods rather than one mutated in place:
    candl caches its compiled log-likelihood on first call, so dropping priors
    after an evaluation silently fails. make_candl drops at construction.
    """
    pytest.importorskip("candl")
    from lensing_fisher.likelihoods import make_candl

    try:
        kept = make_candl(dataset=dataset)
        dropped = make_candl(dataset=dataset, drop_priors=["tau"])
    except Exception as exc:  # data package not installed
        pytest.skip(f"{dataset} unavailable: {exc}")

    # The prior that is there by default has the width the dataset comment quotes.
    response = _tau_response(kept, centre=centre)
    assert 0.01 / np.sqrt(2.0 * response) == pytest.approx(sigma, abs=1e-4)

    # And after dropping, tau does not enter the likelihood at all.
    assert _tau_response(dropped, centre=centre) == pytest.approx(0.0, abs=1e-9)

    # Calibration priors are untouched — those are genuine nuisance constraints.
    assert [p.par_names for p in dropped.priors] == [
        p.par_names for p in kept.priors if p.par_names != ["tau"]
    ]
