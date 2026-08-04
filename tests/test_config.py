"""Config loading and the shared dataset definitions. No CAMB / candl needed."""

from pathlib import Path

import numpy as np
import pytest

from lensing_fisher import config

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(autouse=True)
def _fake_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PLANCK_CLIK_BASELINE", "/tmp/fake_planck")
    monkeypatch.setenv("MNU_HUNTER_ROOT", str(tmp_path))


def test_backends_registered():
    from lensing_fisher.likelihoods import BACKENDS
    assert set(BACKENDS) == {"candl", "clipy"}


def test_missing_section_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("likelihoods: {}\ncosmology: {}\n")
    with pytest.raises(ValueError, match="missing required top-level keys"):
        config.load(bad)


def test_include_splices_shared_dataset():
    """The Fisher config and the Cobaya chain must read one dataset definition."""
    cfg = config.load(EXAMPLES / "fisher_spa.yaml")
    shared = config.load_raw(EXAMPLES / "datasets" / "spa.yaml")

    assert cfg["likelihoods"] == shared["likelihoods"]
    assert cfg["likelihoods"]["act_dr6"]["ell_cuts"]["TE"] == [600, 8500]
    # env expansion happened inside the included file
    assert "/tmp/fake_planck" in cfg["likelihoods"]["planck_highl_ttteee"]["clik_path"]


def test_include_rejects_missing_file_and_key(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("likelihoods: !include nope.yaml#likelihoods\n")
    with pytest.raises(FileNotFoundError, match="!include target not found"):
        config.load_raw(bad)

    (tmp_path / "src.yaml").write_text("something_else: {}\n")
    bad2 = tmp_path / "bad2.yaml"
    bad2.write_text("likelihoods: !include src.yaml#likelihoods\n")
    with pytest.raises(KeyError, match="no top-level key"):
        config.load_raw(bad2)


def test_up_pas_dataset_matches_the_documented_ell_split():
    """Guards the dataset: Planck TT<=1000 TE/EE<=600, ACT and SPT added to 1000."""
    likes = config.load_raw(EXAMPLES / "datasets" / "up_planck_act_spt.yaml")["likelihoods"]

    assert "crop TT 0 1000 strict" in likes["planck_highl_ttteee"]["crop"]
    assert "crop TE 0 600 strict" in likes["planck_highl_ttteee"]["crop"]
    # ACT contributes all three spectra over 600-1000. TT overlaps Planck there
    # deliberately, per the DR6 prescription.
    assert likes["act_dr6"]["ell_cuts"] == {
        "TT": [600, 1000], "TE": [600, 1000], "EE": [600, 1000]}
    # SPT TT from its own start; TE/EE from the 600 polarisation handoff.
    assert likes["spt3g_D1"]["ell_cuts"] == {
        "TT": [0, 1000], "TE": [600, 1000], "EE": [600, 1000]}
    # Everything capped at 1000 is what makes this set "unlensed".
    assert all(hi == 1000
               for spec in ("act_dr6", "spt3g_D1")
               for _, hi in likes[spec]["ell_cuts"].values())
    assert "planck_lowl_ee" in likes


def test_up_planck_is_up_pas_minus_act_and_spt():
    """The two unlensed sets must differ by the added data and nothing else.

    That is the whole point of rebuilding the original Planck-only set on the
    clipy stack: if the shared Planck block drifted between the two files, a
    shift in the predicted lensing power would no longer be attributable to ACT
    and SPT.
    """
    p = config.load_raw(EXAMPLES / "datasets" / "up_planck.yaml")["likelihoods"]
    pas = config.load_raw(EXAMPLES / "datasets" / "up_planck_act_spt.yaml")["likelihoods"]

    assert set(pas) - set(p) == {"act_dr6", "spt3g_D1"}
    for name in p:
        assert p[name] == pas[name], f"{name} differs between UP-P and UP-PAS"


def test_spa_fisher_lmax_covers_the_data():
    """ACT DR6 bandpowers reach ell~6126; the reference run used lmax 2500."""
    cfg = config.load(EXAMPLES / "fisher_spa.yaml")
    assert cfg["camb"]["set_for_lmax"]["lmax"] >= 6200
