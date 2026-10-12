# Third-party material in this distribution

This file records the third-party rights attached to the code and data here,
what was deliberately left out, and how this distribution diverges from the
project it was extracted from. It is an engineering record, not legal advice.

**Summary for integrators.** Runtime dependencies are permissively licensed packages; bundled native libraries have their own terms (see [Runtime dependencies](#runtime-dependencies)). No
third-party data ships: no audio, no measured data, no data tables and no model
weights. Nominal band labels are generated from the Renard R10 series (NBS Technical
Note 990, 1978, p. 37, a U.S. Government work). The `adpcm8k` stress codec uses
IMA-style adaptation, project-generated step sizes, and is not bit-exact. Third-party data the code can use is a plug-in you install yourself
([Plug-in parts](#plug-in-parts-how-to-install)), and every audio source is
audited under [Audio data sources](#audio-data-sources). No pretrained weights or external dataset is bundled. Training can use user
recordings, project-generated confusers and explicitly licensed manifests.
`project export` writes six files: `edge.py`, `model.npz`, `LICENSE`, `NOTICE`,
`SHA256SUMS` and `DEPLOY.md`; declared restricted ancestry is refused.

## Runtime dependencies

Version inventory originally inspected 2026-10-06; native-library notice findings
were reviewed in w9c-licence-final (2026-10-08). These are attributed offline
records, not a fresh online licence verification by this documentation pass.

| Package | Version tested | Role | Licence |
| --- | --- | --- | --- |
| numpy | 2.5.3 | Arrays, FFT | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) |
| scipy | 1.18.1 | Signal processing, optimisation, WAV I/O | BSD-3-Clause |
| pydantic (+ pydantic-core, annotated-types, typing-inspection) | 2.13.5 | Config and schema validation | MIT |
| typing-extensions | 4.16.0 | Typing backports | PSF-2.0 |
| PyYAML | 6.0.3 | Manifest parsing | MIT |
| pyroomacoustics | 0.10.1 | RT60 measurement in `room.dataset` | MIT; see note |
| Cython | 3.3.0 | Runtime dependency of pyroomacoustics | Apache-2.0 |
| threadpoolctl | 3.7.0 | Caps native BLAS/OpenMP thread pools during evaluation | BSD-3-Clause |

The packages above are permissively licensed. Their binary wheels also contain native libraries under their own
terms. The numpy and scipy wheels bundle OpenBLAS (BSD-3-Clause), libgfortran (GPL-3.0-or-later WITH
GCC-exception-3.1) and libquadmath (LGPL-2.1-or-later), as each wheel's LICENSE.txt states.

The pyroomacoustics 0.10.1 extension `libroom` is compiled from Eigen 5.0.1 (MPL-2.0, built with `EIGEN_MPL2_ONLY`),
nanoflann 1.9.0 (BSD-2-Clause) and pybind11 3.0.1 (BSD-3-Clause). Its wheel carries only the pyroomacoustics MIT
licence. The `[sofa]` extra's netCDF4 1.7.4 wheel bundles HDF5 1.14.6, netcdf-c, libaec 1.1.3, blosc, zstd,
bzip2, libcurl 8.14.1 and OpenSSL 3.5.1, and pulls in certifi (MPL-2.0).

This project distributes none of these binaries: `pip` fetches them from their publishers. Anyone who
redistributes an environment, wheelhouse, container or device image must include these notices, and must provide
the MPL-2.0 (Eigen) and LGPL-2.1 (libquadmath) source or source offers. No such bundle has been cleared.

Optional extras, for the installed versions inspected here: `recognizer` adds scikit-learn 1.9.1 (BSD-3-Clause), joblib 1.6.0 (BSD-3-Clause), narwhals 2.26.0 (MIT), and joblib's cloudpickle 3.1.2 dependency (BSD-3-Clause); threadpoolctl 3.7.0 (BSD-3-Clause) is a base runtime dependency listed above. `sofa` adds netCDF4 1.7.4 and cftime 1.6.6 (MIT), plus certifi 2026.7.22 (MPL-2.0); netCDF4's bundled native libraries have separate terms described above and in the register. `nn` uses torch; the tested CPU wheel's full licence expression and distribution limits are recorded below. None of these dependency distributions is bundled here.
The CNN training path imports torch; room acoustics does not require it.

`nn` adds torch for CNN training only. Tested: torch 2.14.1+cpu from
`https://download.pytorch.org/whl/cpu`, whose metadata declares
`Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`;
the wheel carries its third-party licence files under `torch-*.dist-info/licenses/`.
The CPU build pulls in no NVIDIA libraries (the default PyPI build on Linux
installs CUDA packages under NVIDIA's own licence terms). PyTorch is a training
tool here: nothing from it ships in the exported device folder, and the trained
weights are plain numpy arrays.

**On the device:** `edge.py` needs Python 3.9+ (PSF) and numpy (BSD-3-Clause,
from the distribution's own package, e.g. Raspberry Pi OS `python3-numpy`).

**Host programs, optional.** ffmpeg decodes FLAC, ALAC, MP3 and Ogg recordings
(`room/ingest.py`, `realdata/audio_io.py`); WAV is parsed in Python and needs
nothing. FFmpeg is LGPL-2.1-or-later, and GPL-2.0-or-later when built with its
optional GPL parts ([ffmpeg.org/legal](https://ffmpeg.org/legal.html)). This code
runs it as a separate program and never bundles it. If you ship a device image
that includes ffmpeg, its licence applies to that copy. `h5dump` from the HDF5
tools is used only by the validation benches. HDF5 has its own BSD-style licence
with additional enhancement wording; do not label it simply BSD-3-Clause. h5py
(BSD-3-Clause) is an optional internal HDF5 validation tool, not a core dependency.

**pyroomacoustics note.** The installed wrapper is MIT; its compiled libroom extension has separately recorded Eigen/MPL-2.0 obligations, while its installed licence inventory contains only the MIT file. A dependency declaration does not bundle that extension. Any distributor of the extension or an environment containing it must preserve applicable notices and satisfy the covered-source availability requirements; adding an MPL label alone is insufficient.

## Third-party data in the shipped code

A scan of every numeric literal table in the shipped source (six or more
values), plus every comment marked FITTED, DIGITISED, Table, ISO/IEC or licence,
finds **no third-party data table**. The last one became a plug-in in this release:

| Where | What | Source | Status now |
| --- | --- | --- | --- |
| `acoustics/materials.py` (module since removed) | Random-incidence absorption coefficients of seven generic surfaces, 125–4000 Hz | Engineering ToolBox (2003), "Sound absorption coefficients". The site states no licence or permission to reproduce and cites no source for the values (checked 2026-10-07) | **Removed**, and later the whole building-scene material module with it |

The building-scene acoustics inherited from the parent project (the outdoor moving-source
scene simulator, its transmission-loss material presets, ground reflection and interior reverb)
are not part of the audio training system and are not included. With them went the glazing presets
whose loss factors had been calibrated against withheld measurements (Viridian, Balastegui); no
measurement-derived material value ships. The room renderer keeps only the two obstacle diffraction
routes it uses (`acoustics/scene_paths.py`).
This distribution contains no radio, radar or optical simulation code or data. The parent
project's material-catalogue library and its RF, optical and thermal catalogues are not included.

Checked and **not** third-party data: `acoustics/weighting.py` (IEC 61672
A/C-weighting computed from its formulas), `acoustics/absorption.py` (ISO 9613-1
air absorption, formula), `room/metrics.py` (ISO 3382 decay metrics, method),
and `acoustics/bands.py` (nominal labels generated from the Renard R10 series;
exact centres computed as 1000·10^(n/10); no standards table is included).

The recognizer modules added in this distribution contain:

| Where | What | Source | Status |
| --- | --- | --- | --- |
| `iot_mic.py` `adpcm_like` (`ima_adpcm` is a compatibility alias) | 4-bit ADPCM-like stress quantizer: geometric step scale round(7·1.1^i), clipped at 32767; IMA-style adaptation | The step scale follows IMA/DVI's design principle; the adaptation increments equal IMA/DVI's (written as a formula, not a copied table) | Project-generated step sizes; not bit-exact and not G.726-interoperable. The increments are functional parameters of the method. |
| `iot_mic.py` `mulaw`, `peaking`, `high_shelf` | Continuous μ-law companding (μ = 255); peaking and high-shelf biquads | ITU-T G.711's μ-law curve, in its continuous textbook form, not the standard's segment tables; R. Bristow-Johnson's *Audio EQ Cookbook* formulas | Formulas, written here from the equations |
| `iot_mic.py` `PROFILES`, `random_chain` | Device archetypes and training ranges | Chosen for this project from public data sheets and codec descriptions (TDK InvenSense INMP441; ST gasket application note; codec guides cited in USAGE.md) | Own parameter choices; no measured device responses are included |
| `edge.py`, `nn.py` | Mel filterbank (HTK mel formula), DS-CNN architecture, SpecAugment, Freq-MixStyle | Published methods (Zhang et al. 2017; Park et al. 2019; Kim et al. / Schmid et al.) | Ideas implemented from the papers; no code copied from their repositories |
| `acoustics/room/visualize.py` | The 3D page: canvas renderer, layout, palette | Written for this project | No third-party JavaScript, fonts or colour maps; system fonts are named, not shipped |

Deliberately not used: pretrained audio models (YAMNet and similar are trained on
YouTube-derived AudioSet, and some redistributions carry CC BY-NC-ND terms) and
measured microphone impulse-response collections (MicIRP), whose redistribution
terms were unclear.

Source-document rights are distinct from the project's implementation licence. The cited standards, papers and technical documents are references to methods; their text, code, tables and figures are not licensed for redistribution by this project's Apache-2.0 grant. No document-reuse grant is established here for the Audio EQ Cookbook, ISO/IEC/ITU standards, Farina's paper, the Lagrange-interpolation reference, or the cited learning and room-acoustics papers. Keep them citation-only unless their exact source terms and required notices are independently recorded. The implementations described above are project code under Apache-2.0; this statement does not clear copying from those sources.
taxonomy.yaml cites Neitzel et al. (Table 4) as a bibliographic reference only; no values from it are distributed.

`core/schema.py` `DirectionalAmbientSpec` used to default to four `beta` values
fitted to the DEMAND recordings (Thiemann, Ito & Vincent, 2013, CC BY-SA 3.0,
https://zenodo.org/records/1227121). Those defaults were removed in this release:
the spec, which the room workflow never uses, now requires your own fitted values.
The schema's `ambient_spectrum` names (ISO 717-1 traffic, DEMAND park, Illinois
residual) are only labels: the spectra and the model that used them were removed
earlier.

In-code citation strings and module READMEs name documents in the parent project
(audits, protocols, benchmark notes). Those documents are not included. The
names are kept as provenance, not as links.

## Audio data sources

Every source of audio, or of numbers derived from audio, that this code uses, used
to use, or can read. Source records were checked in the 2026-10-07/08 programme licensing reviews;
no datasets or source licences were downloaded again for this doc pass. Internal
validation permission below is narrower than the public licence alone.

| Source | How this code meets it | Ships here? | Licence | Commercial use |
| --- | --- | --- | --- | --- |
| Your recordings: sweeps, target sounds, background, `evaluate/`, device calibration | Project folders you record (USAGE.md) | No | Yours | Yes. Recordings of people (voices) may need their consent, which no licence covers |
| Generated audio: capture-kit sweeps and pink noise, test fixtures, procedural confusers, simulated sessions, simulated IoT microphones | Generated at run time by this code | No (generated) | Apache-2.0 (this project) | Yes |
| Licensed external clips (CC0-1.0 or CC BY 4.0 items) listed in a negatives manifest | `negatives.clips`, from your download; each item is checked by `licensing.validate_commercial_manifest` | No | CC0-1.0 or CC BY 4.0, per item | Yes; keep each CC BY item's attribution with the training run and model |
| [dEchorate](https://doi.org/10.5281/zenodo.5562386) v1.0 HDF5 | WP4 internal measured-RIR proxy checks; separate from v2 and SOFA | No | CC BY 4.0, historically attributed | User-supplied, read-only internal validation only; see register id `dechorate-hdf5-v1` and the dataset attribution below. No fresh publisher verification is claimed. |
| [dEchorate](https://doi.org/10.5281/zenodo.6576203) v2.0 HDF5 | Read only by the validation benches, from your download | No | CC BY 4.0 | CC BY allows attributed reuse; this programme restricts it to internal validation |
| [MeshRIR](https://doi.org/10.5281/zenodo.10852693) v1.2: measured impulse responses on a 3D grid | Validation benches only | No | CC BY 4.0 | CC BY allows attributed reuse; this programme restricts it to internal validation |
| [BRAS](https://doi.org/10.14279/depositonce-6726.3) v3: RS1 and CR2 scenes; Genelec 8020c directivity | Validation benches; `directivity.load_bras_genelec()` | No | CC BY-SA 4.0 | Internal validation only; not permissive. Declared BRAS/ShareAlike ancestry propagates through loading, synthesis, session ingestion, fitting and dry extraction. Explicit internal-validation sessions retain restricted flags; commercial dataset generation, training and export refuse them. The guard also recognises declared NC/ND terms and internal/never register identifiers. Stripped or unknown ancestry is not cleared. |
| SOFA directivity files | `directivity.load_sofa()` reads a file you supply | No | Per file | Check each file |
| [DEMAND](https://zenodo.org/records/1227121) noise recordings | Formerly the parent's ambient-noise model and four fitted defaults in `core/schema.py`; both removed | No | CC BY-SA 3.0 | Not permissive (ShareAlike). Nothing derived from it ships |
| Freesound item 863491; DataSEC | Never used by this code | No | Non-commercial terms | No; not used |
| Pretrained audio models (YAMNet, VGGish, PANNs) and AudioSet | Not used: training starts without pretrained weights | No | — | — |

The legacy dEchorate SOFA edition remains excluded (`never`); its source record
was reported unavailable and its licence scope is unconfirmed. Its validation
loaders have been removed. The generic AES69 directivity reader is unrelated.

For HDF5 dataset attribution: Diego Di Carlo, Pinchas Tandeitnik, Cedric Foy,
Nancy Bertin, Antoine Deleforge and Sharon Gannot, *The dEchorate dataset*,
[v1.0, DOI 10.5281/zenodo.5562386](https://doi.org/10.5281/zenodo.5562386) and
[v2.0, DOI 10.5281/zenodo.6576203](https://doi.org/10.5281/zenodo.6576203),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Related publication: *dEchorate: a calibrated room impulse response dataset for
echo-aware signal processing* (2021). Changes: the internal v1 adapter selects
hyperslabs and interprets geometry from the publisher's script; downstream
benches may resample and align them, as recorded in each run's provenance.
No archive data or derived acoustic result table is included here.

No CC BY dataset or audio pack is bundled or approved for redistribution by this source release. Before using an external CC BY item, retain the creator and any designated attribution parties, the source title when supplied, source URL/DOI and edition, licence name and link, supplied copyright and disclaimer notices, and a statement of changes in a separate attribution record that travels with the item and any required downstream attribution. The uploader field alone is not a complete attribution record. The current MeshRIR, Arni, ReverbDB and FUSS register entries are licence/disposition references, not complete attribution records; complete them from the publisher's records before reuse. Do not invent missing creators or source notices.
The current manifest schema has no full attribution field and forbids extra item properties, so use a companion record until the schema and its validator are amended together. This does not authorize new external data in the repository.

Historical internal figures from the legacy dEchorate edition are withdrawn; dEchorate HDF5 is used only for internal validation from the user's own CC BY 4.0 download.

## Plug-in parts: how to install

None of these is needed for the room workflow and recognizer. Each one adds a
capability, and none ships, so you choose the source and its terms. The README's
[Install section](https://github.com/Pbjis2op4me/sensorsim-room/blob/main/README.md#install) and [Python extras and programs](#python-extras-and-programs) summarize them.

### Absorption catalogue

Per-material absorption priors for the room fit. Without it, every surface starts
from a uniform 0.2 prior (the fitter warns once), then fits to your measurements.
A folder holds two YAML files; these are the fields the fitter reads:

```yaml
# materials.yaml
materials:
  concrete:                                  # IDs: carpet_with_underlay, plasterboard, concrete, glass_float, wood_generic
    absorption_profile_refs: [my_concrete]
```

```yaml
# assemblies_measured.yaml
absorption_profiles:
  my_concrete:
    frequency_hz: {nominal: [125, 250, 500, 1000, 2000, 4000]}
    alpha: {nominal: [a125, a250, a500, a1000, a2000, a4000], source: "where the values come from", validity: "conditions"}
```

Manifest surface descriptions map to IDs by keyword: carpet, drywall or
plasterboard, concrete, glass or window, wood. A description that equals an ID or
a profile name is used directly. Install with `export
SENSORSIM_ACOUSTIC_CATALOGUE=/path/to/folder`, with
`sensorsim.acoustics.room.fit.use_catalogue(folder)` in code, or by copying the
files into `src/sensorsim/library/data/acoustic/` of a source checkout. The
project pipeline refits when the catalogue changes, because its files are part of
the fit stage's inputs. A chosen folder with a file missing is an error, not a
silent fallback. Values can come from your own measurements (ISO 354
reverberation room), manufacturers' data sheets for the products in the room, or a
reference you are licensed to use. Handbook tables are often copyrighted
compilations.

### Measured directivity

- SOFA files (AES69 `FreeFieldDirectivityTF` 1.0 or 1.1): `pip install '.[sofa]'`,
  then `sensorsim.acoustics.room.directivity.load_sofa(path, measurement_index=...)`.
- BRAS Genelec 8020c: download `2_source_and_receiver_descriptions-Genelec_8020c.zip`
  (1.2 GB) from [BRAS v3](https://doi.org/10.14279/depositonce-6726.3), then
  `load_bras_genelec(zip_path)`. It streams the one `_MPS_` CSV member without
  unpacking. CC BY-SA 4.0: see [Audio data sources](#audio-data-sources).

### Measured-room datasets for the validation benches

Only for internal development with `tests/validation/bench_room_closedloop` and
`bench_room_recheck` (v1 HDF5), or `bench_room_fit` (MeshRIR/BRAS). The HDF5
selection remains inconclusive; run its feasibility check before any computation.
Arrange your own permitted downloads as:

```
$SENSORSIM_ROOM_DATA/
  MANIFEST.json                                       written by tests/validation/pin_data.py
  dechorate/v1/dechorate_rir_processed.hdf5          dEchorate v1.0 (10.5281/zenodo.5562386)
  dechorate/v1/load_positions_and_plot.py           publisher geometry; parsed, never executed
  meshrir/S1-M3969_npy.zip                            MeshRIR v1.2 (10.5281/zenodo.10852693)
  meshrir/room_context.json                           you write: {"dimensions_m": [x, y, z], "temperature_c": t}
  bras/1_scene_descriptions-RS1.zip                   BRAS v3 (10.14279/depositonce-6726.3)
  bras/RS1_context.json                               you write: {"temperature_c": t, "humidity_pct": h}
  bras/1_scene_descriptions-CR2.zip                   BRAS v3
```

The two context files hold the room conditions from each publisher's own documentation.
This project does not ship those values; the `bench_room_fit` cases stop without them.

Then install the HDF5 tools (`sudo apt install hdf5-tools` or `brew install
hdf5`) and run `python tests/validation/pin_data.py "$SENSORSIM_ROOM_DATA"`. The
benches refuse any input whose sha256 or size changed since pinning. Pinning
records your download; it does not prove the files match the ones the parent
project used, whose hashes are not included. Example:
`python tests/validation/bench_room_fit/run.py --case meshrir --data "$SENSORSIM_ROOM_DATA" --out /tmp/bench`.

### Python extras and programs

PyTorch (CPU build), scikit-learn and netCDF4 have install commands in [README.md](https://github.com/Pbjis2op4me/sensorsim-room/blob/main/README.md#install). ffmpeg is a host program (`sudo apt install ffmpeg` or `brew install ffmpeg`); numpy on the device comes from the OS package (`sudo apt install python3-numpy`, also listed in the exported DEPLOY.md). Licences are above under [Runtime dependencies](#runtime-dependencies).

## Removed from this distribution

| Removed | Licence problem | Effect |
| --- | --- | --- |
| `library/data/acoustic/materials.yaml`, `assemblies_measured.yaml` (absorption catalogue) | 15 entries marked "Copyright retained; extracted factual values only; no source files redistributed", including Long, *Architectural Acoustics* (Elsevier) Table 7.1 | Absorption priors are uniform: the fitter falls back to the broad 0.2 prior it already defined and warns once. Install a catalogue you have the rights to as a [plug-in](#absorption-catalogue) to restore per-material priors. |
| `acoustics/noise.py` (ambient noise model) | ISO 717-1:2013 Table 4 spectrum No. 2 (standards table); band levels, event statistics and directional fits derived from the DEMAND dataset (CC BY-SA 3.0, share-alike) | `AugmentationConfig.noise_spectrum` accepts `white` or `pink` only; `traffic`/`park` are rejected with an explanation. Record your own room's noise instead. |
| ISO 717-1 reference curve and `rw_iso717()` in `materials.py` | ISO standards table | None: nothing called it. |
| `BALASTEGUI_2013_NR` in `materials.py` | Car noise-reduction curves digitised from a figure in Balastegui et al., *Applied Acoustics* 74 (2013); the open proof is marked **All Rights Reserved** | None: never referenced (validation data only). |
| `VIRIDIAN`, `VIRIDIAN_F_HZ` in `materials.py` | Viridian Glass TechDirect (2018) manufacturer measurements | None: never referenced. |
| `core/configuration.py` | Not a licence issue: unfixed HIGH security finding, see [KNOWN_ISSUES.md](https://github.com/Pbjis2op4me/sensorsim-room/blob/main/KNOWN_ISSUES.md) | Not shipped; neither is the scenario schema it composed. |
| `listening_kit/`, generated WAVs | Rendered audio tied to the parent's reference datasets | Generate the capture kit with `scripts/room_sweep_kit.py`. |
| DataSEC, Freesound item 863491 | Non-commercial licences | Never used by this code. |

The test suite was ported from the parent. It generates every fixture
synthetically at test time and contains no recorded audio or external data.
Loaders for public datasets (dEchorate, BRAS) read paths you supply; no data
from those sets ships.

## Divergence from the parent project

The original export retained many parent blobs; later ports and programme modules
diverge as recorded below. Matching unchanged blobs identifies compatible parent
snapshots, not a unique extraction revision. Before the corrections in the second table, a fixed-seed run (synthetic
session → fit → dataset, catalogue present) produced byte-identical audio and
identical fitted parameters in both. The corrections change outputs on purpose
(sound speed, late tail, band metrics, device features), so this package no
longer reproduces the parent's numbers; historical dEchorate accuracy figures
are withdrawn. Datasets
where a class has two or more recordings also split differently, by design (ML02).

Removals for licensing are listed above. Code changes:

| File | Change | Why |
| --- | --- | --- |
| `room/fit.py` | `_material_tables()` treats the catalogue as optional | Run without the withheld catalogue |
| `room/fit.py` | `estimate_geometry` / `_first_echoes` use float64 | cr-room-model-04: integer coordinates broke geometry |
| `room/ingest.py` | FLAC/ALAC use the integer-PCM clipping rule | cr-room-capture-01 |
| `room/dry.py` | Output-buffer alias check covers noise floors and RIRs | cr-room-dsp-02 |
| `core/fingerprint.py` + `room/{session_sim,dataset,fit,dry}.py` | Provenance via `source_revision()`: package version, plus a commit only from this package's own checkout | cr-room-capture-04: crashes, and the caller's repository HEAD recorded as the simulator revision |
| `room/dataset.py` | Validation recordings stratified by label; single-recording classes always train; `--all-sources` CLI flag | ML02: a class could be missing from validation, or present only there |
| `room/ingest.py`, `room/pipeline.py` | Per-take `event_merge_gap_s` option (default 0.15 s = unchanged) | Multi-part sounds (knocks) could not be selected as one event |
| `room/augment.py` | `noise_spectrum` limited to `white`/`pink` | `noise.py` removed |
| `acoustics/scene_paths.py` | `propagation` imported lazily (later removed with the outdoor scene engine) | The base install no longer needs torch (the parent's own patch, verified identical on 13/13 cases) |
| `core/schema.py` | Reduced to the obstacle-box geometry the room renderer uses | The drone scenario schema and recording-clock simulator are not part of the audio training system |
| `room/pipeline.py` | `--help` points to USAGE.md | The referenced document was not shipped |
| `src/sensorsim/recognizer.py` | **New**: baseline recognizer. Fixed windows, targets scaled into real-level background, gain-normalized features, JSON model | Train/recognize step for the intended use |
| `src/sensorsim/project.py` | **New**: one-command pipeline with hash-keyed stages and gates; the CNN as the default recognizer; `export` (device folder) and `visualize` (3D page) | The training sequence as one reproducible run |
| `src/sensorsim/iot_mic.py` | **New**: simulated IoT microphone capture chains (training draws and evaluation archetypes) | Recognition through cheap microphones |
| `src/sensorsim/nn.py`, `src/sensorsim/edge.py` | **New**: CNN trainer (PyTorch) and its numpy-only device runtime, with per-class event tracking and calibration | A recognizer that runs on a Raspberry Pi |
| `src/sensorsim/acoustics/room/visualize.py` | **New**: self-contained 3D page of the renderer's sound paths | Seeing how sound reaches the microphone |
| `src/sensorsim/recognizer.py` | `train(..., mic_augment=)` option | Baseline comparison with the same microphone augmentation |
| `acoustics/materials.py` | Absorption rows removed (the module was later removed with the building-scene acoustics) | Engineering ToolBox values: rights unverified |
| `room/fit.py` | Catalogue location: `use_catalogue()`, `SENSORSIM_ACOUSTIC_CATALOGUE`; `catalogue_fingerprint()` keys the project's fit stage | Install a catalogue without editing the installed package; refit when it changes |
| `core/schema.py` | `DirectionalAmbientSpec` fitted defaults removed | Fitted to the CC BY-SA DEMAND recordings |
| `realdata/__init__.py` | Docstring describes only what ships | It described a downloader and dataset registry that are not included |
| `tests/validation/bench_room_closedloop/measured.py` | HOMULA scenarios removed | Their engine module (`measured_homula.py`) is not included |
| `tests/validation/pin_data.py` | **New**: pins downloaded datasets in `MANIFEST.json` | The parent's downloader, which wrote that file, is not included |
| dEchorate documentation and validation benches | Historical figures withdrawn; legacy SOFA loaders and cost registration removed | Export-native release cleanup; HDF5 remains internal development only, with no replacement accuracy table |
| `tests/conftest.py` | Ignores the plug-in environment variables | Goldens assume the shipped defaults |
| `tests/validation/math_compare/` | **New**: scripts that run each core formula next to an open-source equivalent | MATH.md. They import pyfar, pyrato, librosa, torchaudio and python-acoustics when installed; none is a dependency and no code is copied |
| `src/sensorsim/edge.py` `resample` | Now the filter of scipy's `resample_poly` defaults | The training data is resampled with it (MATH.md) |

Corrections to the physics and metrics, made after the first export (each changes outputs):

| File | Change | Effect |
| --- | --- | --- |
| `core/geometry.py` | Cramer's humid-air speed of sound (`speed_of_sound_cramer`, `sound_speed`); the dry formula stays as `dry-air-core/1` | 343.98 instead of 343.21 m/s at 20 °C, 50 % RH: arrival anchors move by about one sample per 3 m at 48 kHz |
| `room/model.py`, `room/pipeline.py` | `RoomModel.sound_speed_model` and `timing_policy` default to `humid-air-cramer/1`, fed by `fit: {temperature_c, humidity_pct}`; refused outside 0–30 °C without a fixed reference speed | Timing and the fitted room use the room's air. Room models saved without the field (the parent's, earlier exports) now load as humid air: refit, or set `sound_speed_model: dry-air-core/1` to reproduce them |
| `room/synth.py`, `room/fit.py` | Late tail tiles DC to Nyquist (low-pass, octave band-passes, high-pass or an air-absorbing extension band); the fit mirrors the same bands and keeps the renderer settings it used | Training audio has reverberation below 88 Hz and up to Nyquist; it had none there |
| `room/metrics.py` | `room-measured-metrics/2`: per-band decay times from a time-reversed filter, energy metrics from a zero-phase filter; `/1` stays selectable | Low-band EDT, T30 and clarity agree with the closed form (MATH.md); the fit's internal objective still compares its own measured and predicted values under `/1` |
| `room/model.py`, `room/synth.py`, `room/pipeline.py` | Schroeder frequency 2000 √(T/V); bands below it reported as the modal region | Reports and render metadata say where the diffuse model stops applying |
| `edge.py` | Each window's mean removed before framing | A raw MEMS microphone's DC offset no longer changes the features |
| `iot_mic.py` | Historical random stress chains and Pi alias; WP5 below replaces the in-distribution default with the versioned absolute-level profile | Old alias-based results are historical, not current-device validation |
| `project.py` | DEPLOY.md explains I2S, USB and I2C level-sensor microphones, generically | A level sensor cannot feed the recognizer |
| `tests/validation/bench_room_closedloop/{measured,runner}.py`, `bench_room_kit_e2e/inverse_run.py` | Measured-RIR time origin uses the measurement's logged air; provenance hashes of undistributed documents recorded as absent | The benches ran to completion only in the parent repository |

Fixes from the second port round (2026-10-08), each with a regression test (see
[KNOWN_ISSUES.md](https://github.com/Pbjis2op4me/sensorsim-room/blob/main/KNOWN_ISSUES.md#fixed-in-this-distribution)). "Parent" names the parent commit it was ported
from; "export" marks a fix made here first:

| File | Change | Why |
| --- | --- | --- |
| `room/capture_qa.py` | `capture-qa/2`: two-level gating includes each level's own consistency and planned-repeat status; harmonic and fundamental packets use matching windows; `/1` reports still attach | cr-room-capture-02/03 (parent 7efad42, 25a538a, 716ef5c) |
| `room/rir.py` | Each channel searched near its own arrival; every channel must pass the clipping rule; opt-in `packet-medoid/1` refuses arrivals outside its early window | cr-room-dsp-01/04 (parent 55d4f9f, c479f31, 4c2d78b) |
| `room/ingest.py` | `SweepKit.period_s` must match the generated playback's whole-sample period | cr-room-dsp-03 (parent a205993) |
| `acoustics/materials.py` | Each constructed or replaced material has its own transmission cache | cr-acoustics-2-01 (parent e291b9a) |
| `acoustics/audio_boundary.py` | Strict PCM clipping compares signed bounds directly | cr-acoustics-3-03 (parent b4c8c38) |
| `core/geometry.py` | Integer angle arrays become float64 before wrapping | cr-core-run-05 (parent a09734c) |
| `core/serialization.py` | `atomic_json` follows the umask for new files and keeps an existing file's mode | Files were published as 0600 (parent 851966f) |
| `scene/raycore.py` | `TriangleMesh` and exclusion IDs must fit non-negative signed `int64` | cr-scene-gpu-03 (export) |
| `scene/raycore.py`, `scene/raycore_vulkan.py` (file since removed) | Removed the EO ray adapters (`_ray_box_intervals`, `ray_closest`, `ray_shadowed`), the EO and optical query contracts (`eo-entering-ray/1`, `eo-shadow-ray/1`, `optical-patch-ray/1`), the backend manifest's `eo` note and Vulkan query modes 3/4; finite-segment queries (modes 0/1/2 and recheck mode 5) unchanged | cr-scene-gpu-04: camera-only code with no caller here (export) |
| `room/fit.py` | `_cache` resolves both devices; `_expected_all` applies their per-octave diffuse-power product to the late tail | cr-room-model-01 (parent a9a5b51) |
| `room/fit.py` | Floor-dependent obstacle path groups follow the fitted floor absorption per octave | cr-room-model-02 (parent 5ad1f90, 9565122) |
| `room/fit.py` | `run_manifest` forms device references from `REF` sweeps only | Real and noise takes at `REF` were treated as reference sweeps (parent 71de86b) |
| `room/synth.py` | `SynthConfig` dumps keep omitted, inherited, overridden and disabled fields apart | cr-room-model-03 (parent da4f079) |
| `room/synth.py`, `room/fit.py` | A too-early **default** handoff is raised continuously to direct delay + crossfade/2 + 1/c, in synthesis and the fit cache alike; RIR metadata records both times; an explicit too-early `mixing_time_s` raises | ML10 (export) |
| `room/synth.py` | Measured-directivity late power indexed by each late band's octave, including the Nyquist extension band | D2: renders at 8/32/44.1/48 kHz crashed (export) |
| `room/dataset.py` | Identity uses dtype-tagged bytes and refuses non-numeric dtypes; identical audio under two labels is refused, naming both; rows whose final PCM16 is all zero are refused | ML06, ML03, ML04 (export) |
| `room/dataset.py`, `room/README_SYNTH.md`, `room/README_DRY.md` | `--clips` help states the `source_reference`/`dry_report` requirement; clip rows and the versioned RIR index documented | Parent 861d4ab, a266654 |
| `tests/validation/bench_room_closedloop/runner.py`, `measured.py` | Dry recovery bound to a fresh attempt, the selected session file, its input digest and sample rate | cr-tests-harness-05 (parent c561229, review fix ce62c28) |

Tests: `tests/conftest.py` is new (no torch). `test_pipeline.py` now asserts *a*
source identity, since an installed copy has no git commit.
`test_fit.py::test_material_priors_keep_provenance_and_unmeasured_band` now runs
on a synthetic catalogue installed as a plug-in; the parent's version asserted a
value from the withheld handbook catalogue. `measured.py` no longer hard-codes a local data
path. New: `test_export_regressions.py`, `tests/unit/recognizer/`,
`tests/unit/project/`, `tests/unit/iot_mic/`, `tests/unit/edge/`, `tests/unit/nn/`,
`tests/unit/visualize/`, `tests/unit/plugins/` and `tests/unit/physics/` (sound speed, band coverage,
band metrics, modal region). Three parent tests were updated for the new validation-split rule, each
with a comment: `test_dataset_audit.py::test_source_group_connections_are_transitive_even_for_renamed_copies`
(clips given one label, so it still tests transitive grouping), and
`test_pipeline.py::test_fast_fake_user_session` / `::test_full_canonical_kit_cli`
(two single-recording labels now both train, with an empty validation split and a warning).

### Programme additions through release-doc base 506315b

All rows below are **export-native** (parent source commit: N/A). Commit IDs name
implementation sources, not current scientific acceptance. Lane result/review
records are internal programme citations; the failure history is retained in
KNOWN_ISSUES.md. No third-party recordings, weights or measured tables were added.

| Lane / source commit | Files / change | Scope |
| --- | --- | --- |
| w1-eval-fix/final `7b65b91`, `c1b0986`, `a1c4a44` (merge `94a3b02`) | `evaluation/{realism,calibration,c2st,stream,sensitivity,reporting}.py`, `project.py`: matched training stream, grouped null, power checks, refit permutation and schema /2 | Weak-evidence qualifiers and original framing FAIL retained |
| w2-fa-fix/polish/final `944e005`, `0fd0844`, `be44ce5`, `f3d95e8`, `705edd3` | `evaluation/{fa_rate,intervals}.py`: whole-context counts, separate starts/ends, sparse floor and constant-stratum bounds | Approximate coverage and original stress FAIL retained |
| w3-negatives-final `6605270` (from `2cc47b6`, `dbad7b7`) | `negatives/`, `training_device.py`: procedural/manifest admission, receipts, locked mining, shared composition and floating-point bed preservation | L4 outcome-informed equivalence and unvalidated real-audio benefit remain |
| w4-ensemble-final `19225ff` (from `fcf3a81`, `5388c44`) | `room/{prior,render_prior}.py` and no-sweep pipeline: surveyed/calibration/domestic ensembles, clock alignment and conflict states | WP4(a) not accepted; prior ranges are assumptions |
| w5-device-fix `05340b4` | `iot_mic.py`, `training_device.py`: sine-referenced absolute floor, filters and event-only capture | Original floor FAIL plus fresh proxy correction PASS |
| w6-targets-final `85f5bc0`, `647b740`; integration `945c6cd`, `349def1` | `sources/`, dataset/project handoff: chamber QA, clap, grouping, playback exclusion, frame approvals and adapted fixture | Proxy-developed source QA; no real learning curve |
| w7-dynamics-final `4b1451b` (from `a1015b1`, `675a470`, `7ad9684`) | `training_dynamics.py`, observer and trainer: Pi trace, opt-in knobs, paired crops/gain, validation isolation | New knobs off; partial/hard project modes refused |
| w8-recheck-fix `cce1547` (from `da8f9c6`) | `tests/validation/bench_room_recheck/`: explicit zero noise for synthetic decay, qualified diagnostics | RR-02 not a defect; raw join FAIL cannot diagnose a step |
| w8c-renderer `a96574c` | `room/{synth,fit,late}.py`: stationary warm-up, output envelope, shared covariance | Changes waveforms/fits; join FAIL retained |
| w8d-renderer-speed `2c73460`, `c421ced` | Exact fade/geometry reuse and conservative rounding shortcut; post-direct join/cost tools | Merged R1 with cost FAIL; low-band join diagnostic inconclusive |
| w8e-renderer-speed2 `81289b8`; oracle `ddf1dea` | Separable evaluator prototype and independent real-SOS controls | Prototype not directly merged as a passing fix; evaluator was subsequently ported by w8f. WP8E-NUM remains open |
| w8f-renderer-speed3 `ea09e71`, merge `159518b` | `room/late.py`: byte-bounded geometry reuse | Legacy dEchorate cost/numerical figures withdrawn; historical default-cost FAIL and invalid parameter gate remain recorded |
| w9c-licence-final `d0455e1` (from `3fb1667`, `643ef2e`) | R10 generation, procedural ADPCM-like steps, Appendix A/Vulkan removal, packaged register/schema, internal-validation lineage | No source-completeness or stripped-ancestry clearance |
| w9d-licence-matcher `e728775` | `licensing.py`: field-sensitive declared-source matching and benign training controls | Honest provenance required; remaining distinctive-name false positives/misses |
| int-eval `6b9cf7e` | `edge.py`, evaluation overlays/project gates: combined calibration budget and own-audio composition | OVERLAY receipt rejected; other engineering claims scoped separately |
| int-eval-2 `179ab2f` | Saved calibration digests, original sample ranges and exact block fingerprints in all scoring routes | Later independent receipt accepts scoped disjointness checks; transformed/short crops remain limits |
| int-train `aa420df` | Shared negative-training draw and tracker-panel selection | Acceptance not run: unreachable gate/default-manifest defect |
| int-train-2 `c5db9b2`, merge `242cf44` | Automatic confusers/panel, receipt hashes, terminal fallback, screen tools | Software integration accepted; pre-R1 screen does not show a production benefit |
| polish-1 `3678cb8`, merge `d941e45` | Intake/cache exclusions, portable receipts, PCM/date errors, calibration recipe/warnings and declared-lineage refinements | Scoped software review passed; no new physical evidence |
| window-fix `f24a831`, merge `54ad095` | Opt-in deployment-grid placement, masked unknown labels, full-take history/gain and wrong-class/confusion reporting | Legacy stays default; 2026-10-10 synthetic-proxy screen registered FAIL, audit pending |
| t3-negative-training-fix2 `72b05f3`, handoff `007f381` | `training_negatives.py` and WP2 hooks: matched contexts, bounded reviewed mining, checkpoint/policy/split binding and original-parent grouping | Software repairs integrated; no accepted nuisance/recall benefit |
| t3-operational-selection-fix2 `477071f`, integration `ac3478f` | `evaluation/selection.py`, `nn.py`, `edge.py`: operational-bank selection, tied-score AP, checkpoint/policy binding and invalidation | Integration reviewed; population efficacy remains unqualified |
| t3-room-discrepancy-fix2 `fcb72b4`, integration `58024c9` | Modular discrepancy/pooling, research-only ensemble type, training refusal and complete-scene geometry draws | Original WP4 failure retained; no new predictive-coverage acceptance |
| t3-cue-valid-augmentation-fix2 `edfb99b`, integration `e5e8016` | `training_policy.py` and paired trainer guards: cue validity, parent identity and export retention checks | Defaults off; confuser pairing refused without its identity hook |
| t3-parent-event-levels-fix3 `c27279b`, integration `98951c6`, `f773f05` | `event_levels.py`, mixer, dataset metadata and trainer/project guards: whole-parent gains, relative sidecars and JSON-stable diagnostics | Integration reviewed; full-parent training remains refused |
| dechorate-recompute-fix3 `6226917`, integration `48cfca2` | `licensing.py` and controls: preserve versioned declared ancestry while allowing incidental folder names | Matcher repair only; HDF5 acoustic recompute remains inconclusive |
| opt-tests `09c8ac2`, merge `ce540cf` | Session-local copies of generated fixtures; separate slow regressions and frozen acceptance markers | Fit/train calls and assertions retained; no third-party fixtures |
| opt2-device-chain `9426437`, merge `1addd4f` | `iot_mic.py`: bounded exact SOS/FIR-response caches keyed by parameters, hooks and tap bytes | CPU route equivalence; no physical microphone validation |
| opt-eval `17dcc68`, `6c51b85`, merge `82f3267` | Grouped C2ST bootstrap bookkeeping and ordered MMD complements | Edge-padding commit excluded; later exact-MMD amendment supersedes old MMD results |
| opt-train-stream `ebd8af1`, `655639e`, merge `7f1ecab` | Recording-QA frame powers, silent-event bypass and bounded whole-take power cache | Exact CPU route changes; no per-lane speed claim |
| opt2-matched-context `25e0993`, `3ba274e`, merge `c4c7e1e` | Parsed split plans and bounded exact RIR/parent-render caches, with current-content authorization checks | Seeds, audio and admission checks retained |
| opt2-parallel-gen `f39c6dc`, merge `31e3a14` | `training_stream.py`, `nn.py`: opt-in per-example stream/2 seeds, private loader RNG and affinity-aware workers | Legacy stream remains default; streams differ in samples and weights |
| opt-render `38b466c`, `5d0a0fd`, merge `30c2d1e` | Reuse late-envelope work and bounded immutable image-lattice arrays | Original path/RNG order retained; review's surviving rounding mutant remains a harness limit |
| opt-fit `2395343`, `a2f5f15`, merge `4d57293` | Exact forward-table reuse, ordered spawned bootstrap refits and batched likelihood means | Main fit stays serial; optimizer starts and acoustic operator retained |
| opt-train-loop `fd807bc`, `5953f05`, integration `7b90078` | Reuse validation tensors/checkpoint export; explicit CPU/CUDA/ROCm Python API | CPU route equivalence reviewed; GPU opt-in and non-bit-identical |
| t3-realism-power-fix2 `8d08a0f`; opt-accept-realism `209d34f` | Scoped physical-feature/complete-family audit and exact per-call C2ST reuse | This work record's complete-family P5 qualification was not run; earlier runtime projections are historical. The later registered synthetic-proxy power result is summarized in README.md and does not establish physical/deployment qualification. |
| opt2-realism-design `33d24fa`, alpha fix `c2d790f` | Fixed-budget decision-only stopping bounds; native Boolean for NumPy alpha | Full diagnostics remain default; bounds preserve the complete-family decision on valid executions |
| realism-mmd-fix `81cedf0`, `827b14c`, merge `921dadd` | Exact dyadic symmetrized block sums and inclusive rational permutation ranks | Arithmetic repair reviewed; old MMD p-values/decisions historical; audio/features unchanged |
| p5-launch-prep `2b614c9`, `bd94e54`, `c01adfb`, merge `5b788c1` | Explicit custodian cases, ordered 1/2/4/8-worker execution, deadlines and detached reviewer binding | This launch-preparation record reviewed code integration but did not itself qualify physical or deployment performance; the later registered synthetic-proxy result is summarized in README.md. |

### Release and lineage limits

The package register and schema under `src/sensorsim/licences/` are loaded through
`importlib.resources` and included by packaging rules. A missing register blocks
export. BRAS/ShareAlike, NC/ND and internal/never declarations propagate into
restricted lineage; an explicit internal-validation session does not authorize
commercial training or shipping renders. The matcher scopes common names to
provenance fields and uses only distinctive identifiers in descriptive paths.
It cannot authenticate receipts or recognize stripped ancestry. Remaining reviewer
counterexamples include distinctive-name false positives and alternate DOI/field
spellings. Keep the input audit even if the guard returns no match.

No fresh source-clearance decision is made here. The original WP9 result remains
inconclusive on completeness. The development Git history contains removed tables and
is not published; this repository is a single snapshot of the reviewed tree.

## Patents

The parent project's survey noted published patent grants covering exponential
sine sweep (ESS) measurement improvements and WPE dereverberation. Their status
and applicability were not verified. This is not a freedom-to-operate opinion.
