# License registry and release gates

[`src/sensorsim/licences/register.yaml`](../src/sensorsim/licences/register.yaml) records source-linked license evidence and the programme disposition for software, optional tools, audio packs, and measured room responses. [`src/sensorsim/licences/audio_pack_manifest.schema.json`](../src/sensorsim/licences/audio_pack_manifest.schema.json) defines the per-file record for a future commercial training pack.

No CC BY dataset or audio pack is bundled or approved for redistribution by this source release. Before using an external CC BY item, retain the creator and any designated attribution parties, the source title when supplied, source URL/DOI and edition, licence name and link, supplied copyright and disclaimer notices, and a statement of changes in a separate attribution record that travels with the item and any required downstream attribution. The uploader field alone is not a complete attribution record. The current MeshRIR, Arni, ReverbDB and FUSS register entries are licence/disposition references, not complete attribution records; complete them from the publisher's records before reuse. Do not invent missing creators or source notices.
The current manifest schema has no full attribution field and forbids extra item properties, so use a companion record until the schema and its validator are amended together. This does not authorize new external data in the repository.

Commercial packs admit only Freesound CC0 and CC BY 4.0 items. Keep the Freesound sound ID, uploader, source page, license URL, modifications, file path, SHA-256, and date checked for every file. CC BY attribution must travel with the training run and model package. Whether attribution must accompany weights trained on CC BY audio is unresolved; get legal advice before distributing such weights.

The library validator `licensing.validate_commercial_manifest` for audio-pack manifests (called by `negatives.clips.load_clip` for external CC0 / CC BY clips) checks the submitted provenance against this register. It does not contact Freesound or independently authenticate an uploader's license selection. A human release review must reopen the source page and re-check the per-item facts.

`project export` checks its finished folder against the exact six-file allowlist in the project license module. The source scanner parses Python imports and rejects names registered as `never_imports`, including constant-name dynamic imports. These checks do not replace review of data lineage or rendered artefact provenance.

Measured RIRs marked `internal` are for read-only validation only. BRAS is ShareAlike and must not be used in renders or anything that could ship. Never-use rows include noncommercial or ShareAlike corpora and records whose audio or license scope is unclear.

## WP9 addendum: removed sources and lineage controls

The original WP9 acceptance result remains **INCONCLUSIVE**: source completeness was not established. Passing executable gates does not clear unavailable or ambiguous sources. The legacy dEchorate SOFA record remains `never` (HTTP 410); its loaders and historical figures are withdrawn. HDF5 editions are for internal development validation from the user's own CC BY 4.0 download, with [dataset attribution](../THIRD_PARTY.md#audio-data-sources). No dataset ships here.

The unused Appendix A (from the parent project's drone-node brief) microphone coordinate array and its raised-microphone index have been deleted. The legacy schema no longer advertises that unsupported preset; provide custom positions or use a single microphone.

Band labels now come from a project function reproducing the Renard R10 values and powers of ten. The historical reference is Hans J. Milton, *The Selection of Preferred Metric Values for Design and Construction*, NBS Technical Note 990 (1978), §2.5.5 p. 29 (Renard's R5); §2.5.6 Table 7 p. 31 and §2.5.12 p. 37 (R10): <https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote990.pdf>. It dates Renard's work to 1877–1879. NIST's public-information policy is <https://www.nist.gov/copyrights-disclaimers>. The source document is not distributed. The labels are computed from the geometric series, using twentieth-unit rounding at decade indices 1 and 5, whole-unit rounding at 9, and tenths elsewhere. There are no further acoustics exceptions in the shipped range. Exact base-10 centres follow `1000 * 10**(n/10)`; nominal labels are not exact centres. Regression digests preserve all five prior nominal tuples without embedding another table. The ISO source remains excluded as a historical record; no ISO table is copied into the replacement.

The `adpcm8k` stress augmentation uses a **project ADPCM-like stand-in, not IMA bit-exact**. Steps are generated from `round(7 * 1.1**i)`, made strictly increasing and clipped at 32767. The predictor is project code; its index-adaptation increments equal IMA/DVI's (written as a formula, no table copied). `ima_adpcm` is retained only as a compatibility function name and explicitly documents the changed behavior. Old IMA/audioop equivalence and recognition results using that codec need rerunning or historical labels.

`load_bras_genelec` marks its provenance `bras_derived: true`. Generic CSV and SOFA loaders retain declared source fields. The lineage guard matches register names and aliases only in declared provenance fields (`licence`/`license`/`rights`, `source`, `dataset` (including SOFA `DatabaseName`), `url`, `doi`, `citation`, `attribution`, `corpus`, `collection`, `title`, `origin`, `references`, their compound field names and nested declarations, and explicit `registry_id`/`restricted_assets`). Matching is identifier-based: common words such as `demand`, `arni`, `sins`, `air` and `bras` in paths, labels or uploader names do not imply ancestry. Those fields match only distinctive dataset identifiers, DOI/Zenodo record identities, source URLs (host plus path), and explicit licence identifiers/phrases. Descriptive children such as paths, uploaders, labels, source groups and notes retain the narrower rule even within a provenance object. Generic hosts such as GitHub or Zenodo alone do not identify a dataset.

The distinctive name list is deliberately curated: `fsd50k`, `esc50`/`esc 50`, `esc10`, `musan`, `meshrir`, `dechorate`, `reverbdb`, `tau srir`, `tau nigens`, `starss22`, and BRAS qualified by `v3`, `cr`/`cr2`, `rs`/`rs1` or `genelec`. These identify corpora or dataset-specific scenes/products; no common word alone is matched outside declared provenance. BRAS archive paths beginning `bras/1_scene_descriptions-` or `bras/2_source_and_receiver_descriptions-` also identify that dataset. Versioned register DOIs retain their base record identity (for example depositonce-6726.2 and .3); a Zenodo record number itself is not a version suffix. `chime home`, `esc 10` and `desed` match only declared provenance fields. Keys outside the provenance list use only distinctive identifiers. Case, separators and URL encoding are normalized. Refusal messages show the matched token and metadata field. Document, standard-table, package and tool register rows are excluded from data-ancestry matching; the separate package import gate remains in force. Algorithm validity/evidence notes are citations, not data ancestry.

An explicit `simulate_session(..., purpose="internal-validation",
external_rir_provenance=...)` allows internal benches and stamps restricted lineage into the session manifest,
RIR metadata and truth. Ingest, fitting and dry extraction preserve that mark. Commercial sessions, dataset
generation, the shared recognizer training reader and device export refuse declared restricted lineage.

The guard depends on honest retained provenance. It cannot identify renamed raw arrays/WAVs after their provenance is discarded or recorded in an unrecognised form, nor reconstruct ancestry in an old model that never recorded it. An empty matcher result is not licence clearance. Such models require an input audit before release. Scientific citations to validation datasets are not themselves derivative data.

## Where the register lives

The register and the manifest schema are package data of `sensorsim` (`src/sensorsim/licences/`), declared in
`pyproject.toml` under `[tool.setuptools.package-data]` and read with `importlib.resources`, so a source checkout,
an installed wheel and an sdist build all use the same lookup. A build that omitted them would make `project export`
fail closed. This folder keeps only the human-facing notes (this file).

## Documents and dependency distribution

Datasheet sensitivity/noise facts and document citations do not grant permission to copy tables, figures, overlays or prose. This audit found no such copied material from the Raspberry Pi I2S white paper or the three MEMS datasheets in the current tree. Keep them citation-only.

Pyroomacoustics is a required runtime dependency in `pyproject.toml`. This source/wheel does not bundle its binary. An upstream wheel distributor is responsible for that wheel's notices; merely declaring a pip dependency is not redistribution of the dependency binary. Anyone redistributing an environment or compiled extension must satisfy the relevant licences. The inspected pyroomacoustics wheel contains Eigen use but no Eigen/MPL notice in its licence-file inventory. Resolve those obligations before bundling it.

## Current review limits

The dEchorate v1.0 HDF5 record (Zenodo 5562386) is recorded separately as `dechorate-hdf5-v1`, using the historical CC BY 4.0 attribution and explicitly identified local evidence. It is user-supplied and internal validation only; no fresh publisher verification is claimed. Keep it distinct from v2 and the excluded SOFA edition. MeshRIR v1.2 and BRAS remain internal validation only; no dataset, BRAS render or BRAS-derived training material ships.

w9d's scoped matcher review does not authenticate metadata. Distinctive names in
filenames can still false-positive. Polish-1 adds BRAS version/archive identities
and the declared fields listed above; its scoped software review passed. Later
matcher repairs distinguish incidental folder names from explicit versioned
ancestry such as `dEchorate/v1`; slash/encoded declarations retain their restrictions.
The HDF5 lineage identifier protects these declarations; the v1-specific row records its separate edition and evidence limits. Keep complete provenance and audit inputs; stripped ancestry and forged receipts are not cleared.
An independent licensing review on 2026-10-11 covered licence texts, code provenance, data and identity, with planted-defect controls in each area. Its findings were addressed in this release; that review is not itself a clearance of the published tree. This is diligence, not legal advice or a guarantee; patents and trademarks were not reviewed.
