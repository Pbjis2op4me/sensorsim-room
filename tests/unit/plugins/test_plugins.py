"""Third-party data that does not ship, installed as plug-ins: scene-material absorption, the
absorption catalogue for the room fit, pinned validation datasets, and the outdoor ambient fit."""
import json

import numpy as np
import pytest
import yaml

from sensorsim.acoustics.room import fit
from sensorsim.core import schema



@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv(fit.CATALOGUE_ENV, raising=False)
    fit.use_catalogue(None)
    yield
    fit.use_catalogue(None)


def _catalogue(folder, alpha=(.05, .06, .07, .08, .09, .1)):
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"materials.yaml").write_text(yaml.safe_dump(
        {"materials": {"concrete": {"absorption_profile_refs": ["concrete_test"]}}}))
    (folder/"assemblies_measured.yaml").write_text(yaml.safe_dump({"absorption_profiles": {"concrete_test": {
        "frequency_hz": {"nominal": [125, 250, 500, 1000, 2000, 4000]},
        "alpha": {"nominal": list(alpha), "source": "synthetic test values", "validity": "test"}}}}))
    return folder


@pytest.mark.parametrize("route", ["function", "environment"])
def test_a_catalogue_outside_the_package_sets_the_fit_priors(clean, tmp_path, monkeypatch, route):
    folder = _catalogue(tmp_path/"catalogue")
    if route == "function":
        fit.use_catalogue(folder)
    else:
        monkeypatch.setenv(fit.CATALOGUE_ENV, str(folder))
    alpha, evidence = fit.material_prior("concrete")
    np.testing.assert_allclose(alpha[:6], [.05, .06, .07, .08, .09, .1])
    assert evidence["source"] == "synthetic test values"
    assert set(fit.catalogue_fingerprint()["files"]) == set(fit.CATALOGUE_FILES)


def test_a_chosen_catalogue_folder_without_its_files_is_an_error_not_a_silent_fallback(clean, tmp_path):
    fit.use_catalogue(tmp_path/"empty")
    with pytest.raises(FileNotFoundError, match="Absorption catalogue"):
        fit.material_prior("concrete")


def test_pinned_downloads_are_accepted_and_changes_are_caught(tmp_path):
    from tests.validation.bench_room_fit.run import verified_sha
    from tests.validation.pin_data import pin
    data = tmp_path/"room_acoustics"
    (data/"meshrir").mkdir(parents=True)
    member = data/"meshrir"/"S1-M3969_npy.zip"
    member.write_bytes(b"stand-in for the downloaded archive")
    pin(data)
    assert len(verified_sha(data, member)) == 64
    with pytest.raises(FileExistsError):
        pin(data)
    member.write_bytes(b"a different download")
    with pytest.raises(ValueError, match="changed"):
        verified_sha(data, member)


def test_no_dataset_fitted_ambient_defaults_ship():
    assert not hasattr(schema, "DirectionalAmbientSpec")      # the outdoor scenario ambient model is not shipped
