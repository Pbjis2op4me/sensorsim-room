"""Fail-closed checks for audio-pack manifests and device export folders."""

from __future__ import annotations

import ast
from datetime import date
from functools import lru_cache
import hashlib
from importlib.resources import files
from importlib.resources.abc import Traversable
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
from urllib.parse import urlparse, unquote
import unicodedata

import yaml


class LicenseGateError(ValueError):
    """An artifact does not satisfy the repository's recorded license policy."""


DEVICE_FILES = frozenset({
    "edge.py", "model.npz", "LICENSE", "NOTICE", "DEPLOY.md", "SHA256SUMS",
})

_ASSET_FIELDS = frozenset({
    "id", "name", "version_or_hash", "url", "license_id", "evidence_quote",
    "checked_on", "disposition",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _resource(name: str) -> Traversable:
    """Package data in ``sensorsim/licences``: the same lookup in a checkout, a wheel and an sdist build."""
    return files("sensorsim").joinpath("licences", name)


def _default_registry_path() -> Traversable:
    # One lookup path. A build that omits the resource fails closed in load_register.
    return _resource("register.yaml")


def audio_pack_manifest_schema() -> dict:
    """The per-item commercial audio-pack manifest schema shipped with the package."""
    return json.loads(_resource("audio_pack_manifest.schema.json").read_text(encoding="utf-8"))


def load_register(registry_path: str | Path | None = None) -> dict:
    path = Path(registry_path) if registry_path is not None else _default_registry_path()
    try:
        registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LicenseGateError(f"cannot read license register {path}: {exc}") from exc
    if not isinstance(registry, dict) or not isinstance(registry.get("assets"), list):
        raise LicenseGateError("license register must contain an assets list")
    return registry


def validate_register(registry_path: str | Path | None = None,
                      *, checked_on_or_after: str = "2026-09-08") -> dict[str, dict]:
    """Validate the evidence fields and return rows keyed by unique asset id."""
    registry = load_register(registry_path)
    rows: dict[str, dict] = {}
    issues: list[str] = []
    for index, row in enumerate(registry["assets"]):
        if not isinstance(row, dict):
            issues.append(f"assets[{index}] is not a mapping")
            continue
        missing = _ASSET_FIELDS - row.keys()
        if missing:
            issues.append(f"assets[{index}] missing {sorted(missing)}")
            continue
        asset_id = row["id"]
        if not isinstance(asset_id, str) or not asset_id:
            issues.append(f"assets[{index}] has an empty/non-string id")
            continue
        if asset_id in rows:
            issues.append(f"duplicate asset id {asset_id!r}")
            continue
        if not isinstance(row["version_or_hash"], str) or not row["version_or_hash"].strip():
            issues.append(f"{asset_id}: version_or_hash is empty")
        parsed = urlparse(str(row["url"]))
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            issues.append(f"{asset_id}: source URL is invalid")
        if not isinstance(row["license_id"], str) or not row["license_id"].strip():
            issues.append(f"{asset_id}: license_id is empty")
        if not isinstance(row["evidence_quote"], str) or not row["evidence_quote"].strip():
            issues.append(f"{asset_id}: evidence_quote is empty")
        checked_on = row["checked_on"]
        if not isinstance(checked_on, str) or not _DATE_RE.fullmatch(checked_on):
            issues.append(f"{asset_id}: checked_on must be YYYY-MM-DD")
        else:
            try:
                date.fromisoformat(checked_on)
                if checked_on < checked_on_or_after:
                    issues.append(f"{asset_id}: stale check date {checked_on}")
            except ValueError:
                issues.append(f"{asset_id}: invalid check date {checked_on}")
        if row["disposition"] not in {"ship", "plug-in", "internal", "never"}:
            issues.append(f"{asset_id}: invalid disposition {row['disposition']!r}")
        rows[asset_id] = row
    forbidden = registry.get("never_imports", [])
    if not isinstance(forbidden, list) or any(not isinstance(name, str) or not name for name in forbidden):
        issues.append("never_imports must be a list of nonempty import names")
    if issues:
        raise LicenseGateError("invalid license register: " + "; ".join(issues))
    return rows


def _read_manifest(manifest: dict | str | Path) -> dict:
    if isinstance(manifest, dict):
        return manifest
    path = Path(manifest)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LicenseGateError(f"cannot read manifest {path}: {exc}") from exc
    try:
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise LicenseGateError(f"cannot parse manifest {path}: {exc}") from exc


def validate_commercial_manifest(manifest: dict | str | Path,
                                 registry_path: str | Path | None = None,
                                 *, audio_root: str | Path | None = None) -> int:
    """Accept only per-item Freesound records mapped to ship/plug-in registry rows.

    This checks a recorded attestation against the register. It does not make a
    network request to verify that a Freesound page still reports the attested
    uploader or license; that remains a review-time responsibility.
    """
    data = _read_manifest(manifest)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise LicenseGateError("manifest schema_version must be 1")
    if data.get("purpose") != "commercial_training":
        raise LicenseGateError("manifest purpose must be commercial_training")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise LicenseGateError("commercial training manifest must contain at least one item")
    rows = validate_register(registry_path)
    required = {
        "freesound_id", "uploader", "source_url", "registry_id", "license_id",
        "license_url", "modifications", "path", "sha256", "verified_on",
    }
    allowed = required
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise LicenseGateError(f"items[{index}] must be an object")
        missing, extra = required - item.keys(), item.keys() - allowed
        if missing:
            raise LicenseGateError(f"items[{index}] missing {sorted(missing)}")
        if extra:
            raise LicenseGateError(f"items[{index}] has unsupported fields {sorted(extra)}")
        sound_id = str(item["freesound_id"])
        if not sound_id.isdigit():
            raise LicenseGateError(f"items[{index}] has invalid Freesound ID")
        if not isinstance(item["uploader"], str) or not item["uploader"].strip():
            raise LicenseGateError(f"items[{index}] uploader is required")
        source = urlparse(str(item["source_url"]))
        if source.scheme != "https" or source.netloc not in {"freesound.org", "www.freesound.org"}:
            raise LicenseGateError(f"items[{index}] source_url must be a Freesound sound page")
        if not re.search(rf"/sounds/{re.escape(sound_id)}/?$", source.path):
            raise LicenseGateError(f"items[{index}] Freesound ID does not match source_url")
        checked_on = item["verified_on"]
        try:
            if not isinstance(checked_on, str) or not _DATE_RE.fullmatch(checked_on):
                raise ValueError
            date.fromisoformat(checked_on)
        except ValueError as exc:
            raise LicenseGateError(f"items[{index}] verified_on must be a valid YYYY-MM-DD date") from exc
        if not isinstance(item["modifications"], (str, list)) or (
            isinstance(item["modifications"], list)
            and any(not isinstance(change, str) for change in item["modifications"])
        ):
            raise LicenseGateError(f"items[{index}] modifications must be text or a list of text")
        raw_path = item["path"]
        if (not isinstance(raw_path, str) or not raw_path.strip()
                or "\\" in raw_path or "\x00" in raw_path
                or re.match(r"^[A-Za-z]:", raw_path)):
            raise LicenseGateError(f"items[{index}] path must be a safe relative path")
        relpath = PurePosixPath(raw_path)
        if (relpath.is_absolute() or not relpath.parts
                or any(part in {"", ".", ".."} for part in relpath.parts)):
            raise LicenseGateError(f"items[{index}] path must be a safe relative path")
        if not isinstance(item["sha256"], str) or not _SHA256_RE.fullmatch(item["sha256"]):
            raise LicenseGateError(f"items[{index}] sha256 must be lowercase hex SHA-256")
        registry_id = item["registry_id"]
        row = rows.get(registry_id)
        if row is None:
            raise LicenseGateError(f"items[{index}] references unregistered asset {registry_id!r}")
        if row["disposition"] not in {"ship", "plug-in"}:
            raise LicenseGateError(
                f"items[{index}] asset {registry_id!r} has prohibited disposition {row['disposition']!r}"
            )
        if row.get("manifestable") is not True or row.get("kind") != "license-class":
            raise LicenseGateError(
                f"items[{index}] asset {registry_id!r} is not an approved audio license class"
            )
        if item["license_id"] != row["license_id"] or item["license_url"] != row["url"]:
            raise LicenseGateError(f"items[{index}] license does not match its registered license class")
        if audio_root is not None:
            root = Path(audio_root).resolve()
            audio_path = (root / relpath).resolve()
            if root not in audio_path.parents or not audio_path.is_file():
                raise LicenseGateError(f"items[{index}] audio path is missing or escapes audio_root")
            digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise LicenseGateError(f"items[{index}] SHA-256 mismatch for {relpath}")
    return len(items)


def validate_device_folder(folder: str | Path) -> list[str]:
    """Require the exact, flat, six-file device release contract."""
    path = Path(folder)
    try:
        entries = list(path.iterdir())
    except OSError as exc:
        raise LicenseGateError(f"cannot scan device folder {path}: {exc}") from exc
    names = {entry.name for entry in entries}
    unexpected = sorted(names - DEVICE_FILES)
    missing = sorted(DEVICE_FILES - names)
    nonfiles = sorted(entry.name for entry in entries if not entry.is_file() or entry.is_symlink())
    if unexpected or missing or nonfiles:
        details = []
        if unexpected:
            details.append(f"unexpected entries {unexpected}")
        if missing:
            details.append(f"missing entries {missing}")
        if nonfiles:
            details.append(f"non-regular files {nonfiles}")
        raise LicenseGateError("device folder violates allowlist: " + "; ".join(details))
    return sorted(names)


def _module_is_forbidden(module: str, forbidden: set[str]) -> str | None:
    for name in forbidden:
        if module == name or module.startswith(name + "."):
            return name
    return None


def _literal_string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _forbidden_imports(tree: ast.AST, forbidden: set[str]) -> list[tuple[int, str]]:
    importlib_names = {"importlib"}
    import_module_names: set[str] = set()
    builtin_import_names = {"__import__"}
    builtins_module_names = {"builtins"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or "importlib")
                elif alias.name == "builtins":
                    builtins_module_names.add(alias.asname or "builtins")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        import_module_names.add(alias.asname or alias.name)
            elif node.module == "builtins":
                for alias in node.names:
                    if alias.name == "__import__":
                        builtin_import_names.add(alias.asname or alias.name)

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        module: str | None = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                banned = _module_is_forbidden(alias.name, forbidden)
                if banned:
                    found.append((node.lineno, banned))
        elif isinstance(node, ast.ImportFrom) and node.module:
            banned = _module_is_forbidden(node.module, forbidden)
            if banned:
                found.append((node.lineno, banned))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and (
                node.func.id in import_module_names or node.func.id in builtin_import_names
            ):
                if node.args:
                    module = _literal_string(node.args[0])
            elif (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                  and isinstance(node.func.value, ast.Name) and node.func.value.id in importlib_names):
                if node.args:
                    module = _literal_string(node.args[0])
            elif (isinstance(node.func, ast.Attribute) and node.func.attr == "__import__"
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id in builtins_module_names):
                if node.args:
                    module = _literal_string(node.args[0])
            banned = _module_is_forbidden(module, forbidden) if module else None
            if banned:
                found.append((node.lineno, banned))
    return found


def scan_for_forbidden_imports(source_root: str | Path,
                               registry_path: str | Path | None = None) -> list[str]:
    """AST-scan Python sources and reject imports registered as never-use packages."""
    registry = load_register(registry_path)
    forbidden = {str(name) for name in registry.get("never_imports", [])}
    forbidden.update(
        str(row.get("import_name"))
        for row in registry["assets"]
        if row.get("disposition") == "never" and row.get("import_name")
    )
    root = Path(source_root)
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            violations.append(f"{path}: cannot parse Python source: {exc}")
            continue
        for lineno, module in _forbidden_imports(tree, forbidden):
            violations.append(f"{path}:{lineno}: forbidden import {module}")
    if violations:
        raise LicenseGateError("forbidden imports found: " + "; ".join(violations))
    return []


# Metadata cannot recover stripped ancestry or authenticate a declaration.
# Scientific validity/evidence notes describe algorithms, not imported assets.
_NON_LINEAGE_KEYS = frozenset({"validity", "evidence_note", "environment", "source_sha256"})
_NON_AUDIO_KINDS = frozenset({
    "technical-document", "documentation", "numeric-standard-table",
    "historical-mathematical-reference", "python-package", "validation-tool",
})
_DESCRIPTIVE_WORDS = frozenset({"path", "file", "filename", "label", "uploader", "owner", "author", "group", "reviewer", "notes", "comment"})
_PROVENANCE_WORDS = frozenset({
    "licence", "license", "rights", "source", "sources", "dataset", "datasets",
    "url", "urls", "doi", "citation", "citations", "database", "databasename",
    "attribution", "corpus", "collection", "title", "origin", "references",
})
# Corpus aliases assert ancestry only in declared provenance. A descriptive name
# or arbitrary ancestor directory, however distinctive, is not a data declaration.
_DISTINCTIVE_NAMES = {
    "fsd50k-packaged": ("fsd50k",), "esc50": ("esc50", "esc 50"),
    "esc10": ("esc10",),
    "musan-audio": ("musan",), "meshrir-v1-2": ("meshrir",),
    "dechorate-hdf5-v2": ("dechorate",), "reverbdb-rirs": ("reverbdb",),
    "tau-srir": ("tau srir",), "tau-nigens-2021": ("tau nigens",),
    "starss22": ("starss22",),
    "bras-v3": ("bras v3", "bras cr", "bras cr2", "bras rs", "bras rs1", "bras genelec",
                "bras 1 scene descriptions", "bras 2 source and receiver descriptions"),
}
_DECLARED_ALIASES = {
    "arni-rir": ("arni",), "bras-v3": ("bras",),
    "chime-home": ("chime home",), "esc10": ("esc 10",), "desed-audio": ("desed",),
    "freesound-sampling-plus": ("sampling plus",),
}


def _normalise_source(text: str) -> str:
    text = unicodedata.normalize("NFKC", unquote(text)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _declared_provenance(key: str) -> bool:
    if set(_normalise_source(key).split()) & _DESCRIPTIVE_WORDS:
        return False
    return (key in {"registry_id", "restricted_assets"}
            or bool(set(_normalise_source(key).split()) & _PROVENANCE_WORDS))


@lru_cache(maxsize=1)
def _restricted_identifiers(register_text: str):
    """Cache parsing, never resource availability: read the register on each check.

    Names/aliases need a declared provenance field. URLs retain their host AND
    path: generic hosts (Zenodo, GitHub, Freesound) alone identify no dataset.
    """
    registry = yaml.safe_load(register_text)
    if not isinstance(registry, dict) or not isinstance(registry.get("assets"), list) or not registry["assets"]:
        raise ValueError("lineage register requires a nonempty assets list")
    patterns = []
    for row in registry["assets"]:
        if row["disposition"] not in {"never", "internal"} or row.get("kind") in _NON_AUDIO_KINDS:
            continue
        asset = row["id"]
        declared = {asset, row["name"], *_DECLARED_ALIASES.get(asset, ()),
                    *_DISTINCTIVE_NAMES.get(asset, ())}
        if row["license_id"].startswith("LicenseRef-"):
            declared.add(row["license_id"])
        url = urlparse(row["url"])
        distinctive = {url.netloc + url.path}
        # Qualified BRAS product/scene names retain specific data identity in
        # model/profile names. Bare corpus aliases above require provenance.
        if asset == "bras-v3":
            distinctive.update(_DISTINCTIVE_NAMES[asset])
        # DOI and contextual Zenodo record IDs survive resolver/API spelling.
        for doi in re.findall(r"10\.\d{4,9}/[^\s?#]+", row["url"], re.I):
            distinctive.add(doi)
            # A terminal .N is a version only after an already numeric record
            # identifier (depositonce-6726.3); zenodo.6576203 is the record itself.
            distinctive.add(re.sub(r"(?<=\d)\.\d+$", "", doi))
        match = re.search(r"zenodo\.org/(?:api/)?records?/(\d+)", row["url"])
        if match:
            distinctive.update({"zenodo " + match[1], "zenodo records " + match[1],
                                "zenodo record " + match[1], "zenodo api records " + match[1],
                                "10.5281/zenodo." + match[1]})
            distinctive.update("zenodo.org/" + route + match[1]
                               for route in ("record/", "records/", "api/records/"))
        patterns.append((asset, tuple(sorted({_normalise_source(v) for v in declared if v})),
                        tuple(sorted({_normalise_source(v) for v in distinctive if v}))))
    return tuple(patterns)


# Recognised data-file names/layouts, not corpus words in project directories.
# These signals retain a restriction; they cannot authenticate or clear data.
_DATA_FILE_PATTERNS = {
    "dechorate-hdf5-v2": (
        r"dechorate_rir_processed\.hdf5",
        r"dechorate_room\d{6}_src\d+_arr\d+_mics\d+-\d+\.sofa",
    ),
    "meshrir-v1-2": (r"(?:meshrir_)?s(?:1-m3969|32-m441)_npy\.zip",),
    "fsd50k-packaged": (r"fsd50k\.(?:dev_audio|eval_audio|ground_truth|metadata|doc)\.zip",),
    "bras-v3": (
        r"1[ _]scene[ _]descriptions-cr2\.zip",
        r"2[ _]source[ _]and[ _]receiver[ _]descriptions-genelec[ _]8020c\.zip",
        r"cr2_rirs_genelec8020c\.sofa",
    ),
}
_PATH_WORDS = frozenset({"path", "file", "filename"})
_DATASET_WORDS = frozenset({"dataset", "datasets", "database", "databasename", "corpus", "collection"})
_LOCATION_SUFFIXES = frozenset({
    ".hdf5", ".h5", ".sofa", ".zip", ".wav", ".flac", ".ogg", ".mp3",
    ".aif", ".aiff", ".opus", ".m4a", ".aac", ".npy", ".npz", ".csv",
    ".json", ".yaml", ".yml",
})


def _location(raw, key, *, is_path=False, declared=False):
    """Return portable path components, keeping explicit DOI/URL declarations."""
    value = unicodedata.normalize("NFKC", unquote(raw)).casefold()
    if re.match(r"^(?:https?://|10\.\d{4,9}/)", value):
        return None
    # A slash alone in provenance (e.g. "dEchorate/v1") does not assert a
    # filesystem location. Other source fields may still hold explicit paths.
    if (is_path or set(_normalise_source(key).split()) & _PATH_WORDS
            or value.startswith(("/", "./", "../", "\\"))
            or re.match(r"^[a-z]:[/\\]", value)
            or (not declared and ("/" in value or "\\" in value)
                and not any(c.isspace() for c in value))
            or Path(value).suffix in _LOCATION_SUFFIXES):
        return tuple(part for part in value.replace("\\", "/").split("/") if part)
    return None


def _data_file_assets(parts):
    if not parts:
        return
    leaf = parts[-1]
    for asset, patterns in _DATA_FILE_PATTERNS.items():
        if any(re.fullmatch(pattern, leaf) for pattern in patterns):
            yield asset, leaf
    # A declared BRAS scene plus an immediate data file is stronger than a
    # user directory named bras, bras_cr2 or bras-research elsewhere in a path.
    if (len(parts) >= 3 and parts[-3] == "bras" and parts[-2] in {"cr2", "rs1", "rs2"}
            and Path(leaf).suffix in {".wav", ".sofa", ".hdf5"}):
        yield "bras-v3", "/".join(parts[-3:])


def _collect_lineage(*values):
    try:
        patterns = _restricted_identifiers(_default_registry_path().read_text(encoding="utf-8"))
    except (OSError, KeyError, TypeError, ValueError, AttributeError, yaml.YAMLError) as exc:
        raise LicenseGateError(f"cannot read lineage register: {exc}") from exc
    restricted: set[str] = set()
    matches: set[tuple[str, str, str]] = set()

    def mark(asset, token, field):
        restricted.add(asset)
        matches.add((asset, token, field or "<root>"))

    def visit(value, key="", field="", declared=False, dataset_declared=False):
        if key in _NON_LINEAGE_KEYS:
            return
        descriptive = bool(set(_normalise_source(key).split()) & _DESCRIPTIVE_WORDS)
        declared = (declared or _declared_provenance(key)) and not descriptive
        # Schema-defined dataset identities outrank path-spelling heuristics,
        # including nested identifiers. Explicit path/description fields reset
        # that context so an incidental directory still cannot invent ancestry.
        dataset_declared = (dataset_declared or key == "registry_id"
                            or bool(set(_normalise_source(key).split()) & _DATASET_WORDS)) and not descriptive
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        if isinstance(value, dict):
            if value.get("bras_derived"):
                mark("bras-v3", "bras_derived=true", f"{field}.bras_derived".lstrip("."))
            if value.get("internal_validation_only"):
                mark("internal-validation-only", "internal_validation_only=true",
                     f"{field}.internal_validation_only".lstrip("."))
            for k, v in value.items():
                visit(v, str(k).lower(), f"{field}.{k}".lstrip("."), declared, dataset_declared)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                visit(v, key, f"{field}[{i}]", declared, dataset_declared)
        elif isinstance(value, (str, Path)):
            raw = str(value)
            normal = _normalise_source(raw)
            # A known geographic name, even inside a source/citation field.
            normal = re.sub(r"\bbras d or\b", "", normal)
            padded = " " + normal + " "
            location = _location(raw, key, is_path=isinstance(value, Path), declared=declared)
            if location is not None:
                for asset, token in _data_file_assets(location):
                    mark(asset, token, field)
            # Full licence phrases and SPDX identifiers are distinctive anywhere.
            # Bare terms such as noncommercial in free-form notes are not declarations.
            words = normal.replace("share alike", "sharealike").replace("non commercial", "noncommercial")
            words = words.replace("no derivatives", "noderivatives").replace("no derivs", "noderivs")
            for term, spdx in (("sharealike", "SA"), ("noncommercial", "NC"),
                               ("noderivatives", "ND"), ("noderivs", "ND")):
                if re.search(r"\b" + term + r"\b", words) and (declared or "attribution" in words.split()):
                    mark("CC-BY-" + spdx, term, field)
            for cc in re.finditer(r"\b(?:cc\s*by|creativecommons org licenses by)(?:\s*(?:nc|nd|sa))+\b", normal):
                terms = re.sub(r"^(?:cc\s*by|creativecommons org licenses by)", "", cc.group())
                for term in re.findall(r"nc|nd|sa", terms):
                    mark("CC-BY-" + term.upper(), cc.group(), field)
            if declared and normal in {"nc", "nd", "sa"}:
                mark("CC-BY-" + normal.upper(), normal, field)
            for asset, names, identifiers in patterns:
                # Only explicit dataset declarations may override a location;
                # ordinary source paths still cannot assert ancestor identity.
                tokens = identifiers if location is None else ()
                if declared and (dataset_declared or location is None):
                    tokens += names
                for token in tokens:
                    if " " + token + " " in padded:
                        mark(asset, token, field)
            if key == "restricted_assets" and raw.strip():
                mark(raw, raw, field)

    for value in values:
        visit(value)
    lineage = ({"bras_derived": "bras-v3" in restricted, "internal_validation_only": True,
                "restricted_assets": sorted(restricted)} if restricted else {})
    return lineage, sorted(matches)


def lineage_metadata(*values) -> dict:
    """Union retained restricted ancestry; an empty result is not licence clearance.

    Register names/aliases match only declared provenance fields. Dataset/registry
    declarations retain ancestry regardless of path-like spelling. Locations
    match recognised data files/layouts, never arbitrary ancestor names. Explicit
    DOI/URL, qualified product and licence identifiers retain restrictions. Documents,
    standard tables, packages and tools are not data ancestry. Stripped or unknown
    provenance cannot be recovered from audio content by this metadata matcher.
    """
    return _collect_lineage(*values)[0]


def require_commercial_lineage(*values, context: str) -> None:
    """Reject declared internal-only ancestry before training/render export."""
    lineage, matches = _collect_lineage(*values)
    if lineage:
        detail = "BRAS-derived; " if lineage["bras_derived"] else ""
        reasons = "; ".join(f"{token!r} in {field} ({asset})" for asset, token, field in matches)
        raise LicenseGateError(f"{context}: {detail}internal-validation-only lineage is prohibited "
                               f"for training and export: {lineage['restricted_assets']}; matched {reasons}")


def validate_training_lineage(manifest: str | Path, rows=None) -> None:
    """Check each render record and its run-level provenance, including old datasets."""
    manifest = Path(manifest)
    if rows is None:
        try:
            rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        except (OSError, ValueError) as exc:
            raise LicenseGateError(f"cannot read training lineage: {exc}") from exc
    require_commercial_lineage(rows, context="training manifest")
    run = manifest.parent / "run.json"
    if run.is_file():
        _validate_lineage_json(run)


def _validate_lineage_json(path: Path) -> None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise LicenseGateError(f"cannot read lineage from {path.name}: {exc}") from exc
    require_commercial_lineage(value, context=path.name)


def validate_project_lineage(output: str | Path, model_metadata: dict) -> None:
    """Check model and retained project provenance before any device files are written."""
    output = Path(output)
    require_commercial_lineage(model_metadata, context="model export")
    for relative in ("fit/room_model.json", "fit/report.json", "dataset/run.json"):
        path = output / relative
        if path.is_file():
            _validate_lineage_json(path)
    manifest = output / "dataset" / "manifest.jsonl"
    if manifest.is_file():
        validate_training_lineage(manifest)
