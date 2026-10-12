"""Generate the user-facing room-measurement playback kit (exponential sine sweeps + level check).

Farina (2000, AES 108th Conv., preprint 5093; 2007 AES 122nd Conv.) exponential sine sweep:
    x(t) = sin( 2*pi*f1*T/ln(f2/f1) * (exp(t*ln(f2/f1)/T) - 1) ),  0 <= t < T
Every parameter needed to regenerate the exact excitation (and its inverse filter) is written to
sweep_kit.json; the playback WAV itself can also be used directly as the known excitation.

Usage: python scripts/room_sweep_kit.py --out OUTPUT [--kit-version 2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile

from sensorsim.acoustics.room.ingest import ESSExcitation
from sensorsim.acoustics.room.rir import generate_ess
from sensorsim.core.fingerprint import sha256_file
from sensorsim.core.serialization import atomic_json, load_yaml_unique

FS = 48_000
F1, F2 = 40.0, 20_000.0          # small Bluetooth speakers cannot reproduce < ~40 Hz
T_SWEEP = 10.0                   # s; long sweep = high SNR and clean harmonic separation
FADE_IN, FADE_OUT = 0.10, 0.01   # s; half-Hann edges (fade-out short to keep the top band)
PEAK_DBFS = -6.0                 # digital headroom; speaker/phone linearity still needs QA
LEAD_SILENCE, TAIL_SILENCE, END_SILENCE = 2.0, 4.0, 2.0   # s
REPEATS = 4                      # averaged after deconvolution -> +6 dB SNR
SEED = 20260930


def ess(fs: int = FS, f1: float = F1, f2: float = F2, T: float = T_SWEEP) -> np.ndarray:
    # Keep unit-amplitude synthesis then playback gain: shipped PCM bytes depend
    # on that operation order. Noninteger durations use the same n/fs phase as RIR.
    return generate_ess(fs, ESSExcitation(duration_s=T, f_start_hz=f1, f_end_hz=f2,
        amplitude=1., fade_in_s=FADE_IN, fade_out_s=FADE_OUT))


def to_pcm16(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    tpdf = (rng.random(x.size) - rng.random(x.size)) / 32768.0   # +-1 LSB triangular dither
    return np.clip(np.round((x + tpdf) * 32767.0), -32768, 32767).astype(np.int16)


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    spec[1:] /= np.sqrt(f[1:]); spec[0] = 0
    band = (f >= F1) & (f <= F2); spec[~band] = 0
    y = np.fft.irfft(spec, n)
    return y / np.max(np.abs(y))


def write_templates(out: Path, placements: int = 3) -> None:
    """Write K-placement/three-fit-position plus one-check guide; no audio mutation."""
    if placements < 1:
        raise ValueError("placements must be positive")
    templates = Path(__file__).with_name("room_sweep_kit_templates")
    (out / "README.md").write_text((templates / "README.md").read_text())
    template = (templates / "manifest_template.yaml").read_text()
    if placements != 3:
        value = load_yaml_unique(template)
        value['positions']['speakers'] = {f'S{k}':[0.,0.,0.] for k in range(1,placements+1)}
        value['takes'] = ([dict(file='noise_floor.wav',type='noise',mic='P1')] +
            [dict(file=f'S{k}_P{j}_sweep.wav',type='sweep',speaker=f'S{k}',mic=f'P{j}')
             for k in range(1,placements+1) for j in range(1,4)] +
            [dict(file='S1_C1_sweep.wav',type='sweep',speaker='S1',mic='C1',role='check')] +
            [dict(file=f'S1_P{j}_real_example.wav',type='real',speaker='S1',mic=f'P{j}',label='example')
             for j in range(1,4)])
        template = '# Sequential one-phone takes; see README for orientation, C1 check, REF and event selection.\n'+yaml.safe_dump(value,sort_keys=False)
    (out / "manifest_template.yaml").write_text(template)
    (out / "capture_qa_template.yaml").write_text((templates / "capture_qa_template.yaml").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("room_recording_kit"))
    ap.add_argument("--placements", type=int, default=3)
    ap.add_argument("--templates-only", action="store_true")
    ap.add_argument("--kit-version", type=int, choices=(1,2), default=1,
                    help="1: unchanged 60 s kit; 2: 76 s kit with an 8 s quiet tail")
    args = ap.parse_args()
    if args.placements < 1:
        ap.error("--placements must be positive")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.templates_only:
        write_templates(out,args.placements)
        return
    rng = np.random.default_rng(SEED)
    peak = 10 ** (PEAK_DBFS / 20)
    s = ess() * peak
    tail = TAIL_SILENCE if args.kit_version==1 else 8.
    period = np.concatenate([s, np.zeros(int(tail * FS))])
    play = np.concatenate([np.zeros(int(LEAD_SILENCE * FS)), np.tile(period, REPEATS),
                           np.zeros(int(END_SILENCE * FS))])
    playback = to_pcm16(play,rng)
    if args.kit_version==2:
        # Keep the single-sweep and level-check PCM byte-identical to v1.
        # Resume the original deterministic dither stream after its 60 s file.
        rng=np.random.default_rng(SEED)
        canonical_count=round((LEAD_SILENCE+REPEATS*(T_SWEEP+TAIL_SILENCE)+END_SILENCE)*FS)
        rng.random(2*canonical_count)
    files = {
        "sweep_playback_48k.wav": playback,
        "sweep_single_48k.wav": to_pcm16(s, rng),                       # reference excitation only
        "level_check_pink_noise_48k.wav": to_pcm16(
            pink(10 * FS, rng) * peak * 10 ** (-6 / 20), rng),            # ~ same loudness, 10 s
    }
    meta = {"fs_hz": FS, "f1_hz": F1, "f2_hz": F2, "sweep_s": T_SWEEP, "fade_in_s": FADE_IN,
            "fade_out_s": FADE_OUT, "peak_dbfs": PEAK_DBFS, "lead_silence_s": LEAD_SILENCE,
            "tail_silence_s": tail, "end_silence_s": END_SILENCE, "repeats": REPEATS,
            "sweep_samples": len(s), "resolved_sweep_s": len(s)/FS,
            "duration_policy": "round(duration_s*fs)/fs",
            "period_s": T_SWEEP + tail, "first_sweep_start_s": LEAD_SILENCE,
            "formula": "x(t)=sin(2*pi*f1*L*(exp(t/L)-1)), L=T/ln(f2/f1); half-Hann fades",
            "reference": "Farina 2000 AES preprint 5093; Farina 2007 AES 122nd Conv.",
            "dither": "TPDF +-1 LSB, seed %d" % SEED, "files": {}}
    if args.kit_version==2:
        meta['kit_version']='2'
    for name, data in files.items():
        wavfile.write(out / name, FS, data)
        meta["files"][name] = {"sha256": sha256_file(out / name),
                               "seconds": round(data.size / FS, 3)}
    atomic_json(out / "sweep_kit.json", meta)
    # Scale the actual playback PCM, independently of the legacy dither stream.
    # Keep both this asset and its level declaration outside the production kit.
    quiet_name = "QA_sweep_minus6_48k.wav"
    wavfile.write(out / quiet_name, FS, np.rint(playback.astype(float)*10**(-6/20)).astype(np.int16))
    def asset(name):
        return {"file": name, "sha256": sha256_file(out / name)}
    atomic_json(out / "capture_qa_assets.json", dict(schema_version="1",
        kit=asset("sweep_kit.json"), nominal=asset("sweep_playback_48k.wav"),
        quiet=asset(quiet_name), digital_difference_db=-6.0))
    write_templates(out,args.placements)
    print(json.dumps(meta["files"], indent=2))


if __name__ == "__main__":
    main()
