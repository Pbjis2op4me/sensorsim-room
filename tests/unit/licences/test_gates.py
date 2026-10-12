"""WP9 policy gates: source-linked manifest, device-folder and import allowlists."""

import hashlib
import json
from pathlib import Path

import pytest

from sensorsim import licensing


ROOT = Path(__file__).resolve().parents[3]
# Package data: the same files the wheel installs (sensorsim/licences/).
REGISTER = ROOT / "src" / "sensorsim" / "licences" / "register.yaml"
SCHEMA = ROOT / "src" / "sensorsim" / "licences" / "audio_pack_manifest.schema.json"


def _manifest(registry_id="freesound-cc-by-4", license_id="CC-BY-4.0", license_url=None, content=b"clip"):
    rows = licensing.validate_register(REGISTER)
    row = rows[registry_id]
    return {
        "schema_version": 1,
        "purpose": "commercial_training",
        "items": [{
            "freesound_id": "123456",
            "uploader": "example-uploader",
            "source_url": "https://freesound.org/sounds/123456/",
            "registry_id": registry_id,
            "license_id": license_id,
            "license_url": license_url or row["url"],
            "modifications": ["resampled to 16 kHz", "trimmed leading silence"],
            "path": "audio/clip.wav",
            "sha256": hashlib.sha256(content).hexdigest(),
            "verified_on": "2026-10-08",
        }],
    }


def test_registry_has_source_linked_current_rows_for_all_assets():
    rows = licensing.validate_register(REGISTER)
    assert len(rows) >= 70
    assert {"python-numpy", "python-scipy", "python-torch", "python-netcdf4", "gpuRIR"} <= rows.keys()
    assert rows["iso-266-preferred-frequencies"]["disposition"] == "never"
    assert all(row["evidence_quote"].strip() and row["url"].startswith("https://") for row in rows.values())
    assert {asset_id for asset_id, row in rows.items() if row.get("manifestable")} == {
        "freesound-cc0", "freesound-cc-by-4",
    }


def test_audio_manifest_schema_contains_required_per_item_provenance():
    schema = json.loads(SCHEMA.read_text())
    assert licensing.audio_pack_manifest_schema() == schema   # package-data accessor reads the same file
    item_schema = schema["properties"]["items"]["items"]
    required = set(item_schema["required"])
    assert {"freesound_id", "uploader", "license_url", "modifications"} <= required
    assert item_schema["properties"]["registry_id"]["enum"] == [
        "freesound-cc0", "freesound-cc-by-4",
    ]


def test_cc_by_manifest_is_accepted_and_payload_hash_is_checked(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    payload = b"proxy fixture only"
    (audio / "clip.wav").write_bytes(payload)
    manifest = _manifest(content=payload)
    assert licensing.validate_commercial_manifest(manifest, REGISTER, audio_root=tmp_path) == 1


def test_commercial_manifest_rejects_non_audio_asset_even_if_ship_licensed():
    manifest = _manifest("python-numpy", "BSD-3-Clause")
    with pytest.raises(licensing.LicenseGateError, match="not an approved audio license class"):
        licensing.validate_commercial_manifest(manifest, REGISTER)


def test_cc_by_nc_manifest_is_rejected_as_commercial_negative_control():
    manifest = _manifest("freesound-cc-by-nc-4", "CC-BY-NC-4.0")
    with pytest.raises(licensing.LicenseGateError, match="prohibited disposition 'never'"):
        licensing.validate_commercial_manifest(manifest, REGISTER)


def test_manifest_rejects_unregistered_asset_and_mismatched_sound_id():
    manifest = _manifest()
    manifest["items"][0]["registry_id"] = "not-registered"
    with pytest.raises(licensing.LicenseGateError, match="unregistered asset"):
        licensing.validate_commercial_manifest(manifest, REGISTER)
    manifest = _manifest()
    manifest["items"][0]["source_url"] = "https://freesound.org/sounds/123/"
    with pytest.raises(licensing.LicenseGateError, match="does not match source_url"):
        licensing.validate_commercial_manifest(manifest, REGISTER)


@pytest.mark.parametrize("unsafe_path", [".", "../outside.wav", "C:/outside.wav", "audio\\clip.wav"])
def test_manifest_rejects_nonportable_or_escaping_paths(unsafe_path):
    manifest = _manifest()
    manifest["items"][0]["path"] = unsafe_path
    with pytest.raises(licensing.LicenseGateError, match="safe relative path"):
        licensing.validate_commercial_manifest(manifest, REGISTER)


def test_exact_device_folder_is_accepted(tmp_path):
    for name in licensing.DEVICE_FILES:
        (tmp_path / name).write_text("proxy fixture only")
    assert licensing.validate_device_folder(tmp_path) == sorted(licensing.DEVICE_FILES)


def test_unregistered_wav_device_folder_negative_control_is_rejected(tmp_path):
    for name in licensing.DEVICE_FILES:
        (tmp_path / name).write_text("proxy fixture only")
    (tmp_path / "extra.wav").write_bytes(b"unregistered")
    with pytest.raises(licensing.LicenseGateError, match="unexpected entries.*extra.wav"):
        licensing.validate_device_folder(tmp_path)


def test_current_source_passes_and_gpu_rir_import_negative_control_fails(tmp_path):
    assert licensing.scan_for_forbidden_imports(ROOT / "src" / "sensorsim", REGISTER) == []
    (tmp_path / "bad.py").write_text("import gpuRIR\n")
    with pytest.raises(licensing.LicenseGateError, match="forbidden import gpuRIR"):
        licensing.scan_for_forbidden_imports(tmp_path, REGISTER)


def test_constant_dynamic_gpu_rir_import_is_also_rejected(tmp_path):
    (tmp_path / "bad.py").write_text(
        "import importlib as loader\nloader.import_module('gpuRIR')\n"
    )
    with pytest.raises(licensing.LicenseGateError, match="forbidden import gpuRIR"):
        licensing.scan_for_forbidden_imports(tmp_path, REGISTER)


def test_builtins_module_gpu_rir_import_is_also_rejected(tmp_path):
    (tmp_path / "bad.py").write_text(
        "import builtins as runtime\nruntime.__import__('gpuRIR')\n"
    )
    with pytest.raises(licensing.LicenseGateError, match="forbidden import gpuRIR"):
        licensing.scan_for_forbidden_imports(tmp_path, REGISTER)


def test_register_and_schema_resolve_as_package_data():
    from importlib.resources import files
    assert licensing._default_registry_path() == files("sensorsim").joinpath("licences", "register.yaml")
    assert licensing._default_registry_path().is_file() and SCHEMA.is_file()
    assert len(licensing.validate_register()) == len(licensing.validate_register(REGISTER))
