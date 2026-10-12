"""Room-chain math against open-source equivalents, each judged by a closed-form case.

Not part of the test suite: it needs comparison libraries that are not dependencies of this
package (pyfar, pyrato, and optionally the unmaintained python-acoustics). Install them in a
separate environment, then run one section or all:

    pip install pyfar pyrato            # plus this package and its dependencies
    python tests/validation/math_compare/room_chain.py [images|delay|decay|clarity|bands|sweep|air|speed|reverb]

Results and verdicts are summarized in MATH.md. Every comparison runs the other library's code
locally; none of it is copied here.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)
SURFACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _table(rows, header):
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


# ------------------------------------------------------------------------------------------ images
def images():
    """Image positions, orders and per-band pressure damping vs pyroomacoustics' image-source model."""
    import pyroomacoustics as pra
    from sensorsim.acoustics.room.model import RoomModel, uniform_bands
    from sensorsim.acoustics.room.synth import image_paths
    dims, src, mic = (5.4, 4.2, 2.7), (1.13, 0.71, 0.93), (3.87, 2.96, 1.41)
    # A different absorption for every wall and band, so a swapped wall or band cannot hide.
    alpha = {s: tuple(round(0.03 + 0.04*w + 0.11*b, 3) for b in range(7)) for w, s in enumerate(SURFACES)}
    ours = image_paths(RoomModel(shoebox_dims_m=dims, absorption=alpha, scattering=uniform_bands(0.)),
                       src, mic, max_order=3, air_absorption=False)
    room = pra.ShoeBox(dims, fs=48000, max_order=3, air_absorption=False, ray_tracing=False,
                       materials=pra.Material(0.1))
    # Map pyroomacoustics wall names to our surfaces from their geometry, not from documentation.
    names = {}
    for wall in room.walls:
        corners = np.asarray(wall.corners)
        for axis in range(3):
            for side, value in ((0, 0.), (1, dims[axis])):
                if np.allclose(corners[axis], value):
                    names[wall.name] = SURFACES[2*axis + side]
    materials = {name: pra.Material(energy_absorption={"coeffs": list(alpha[names[name]]), "center_freqs": list(BANDS)})
                 for name in names}
    room = pra.ShoeBox(dims, fs=48000, max_order=3, air_absorption=False, ray_tracing=False, materials=materials)
    room.add_source(src)
    room.add_microphone(mic)
    room.image_source_model()
    s = room.sources[0]
    theirs_pos, theirs_order, theirs_damp = np.asarray(s.images).T, np.asarray(s.orders), np.asarray(s.damping).T
    centers = np.asarray(room.octave_bands.centers)
    keep = [int(np.argmin(abs(centers - b))) for b in BANDS]
    matched, worst_pos, worst_damp, order_mismatch = 0, 0., 0., 0
    for p, hits, wall in zip(ours.images_m, ours.wall_hits, ours.wall_pressure):
        d = np.linalg.norm(theirs_pos - p, axis=1)
        j = int(np.argmin(d))
        worst_pos = max(worst_pos, float(d[j]))
        order_mismatch += int(theirs_order[j] != hits.sum())
        worst_damp = max(worst_damp, float(np.max(np.abs(theirs_damp[j, keep] - wall))))
        matched += 1
    print(f"pyroomacoustics {pra.__version__}: wall names -> {names}")
    _table([["images (order <= 3)", len(ours.images_m), len(theirs_pos)],
            ["largest position difference (m)", f"{worst_pos:.1e}", ""],
            ["reflection-order mismatches", order_mismatch, ""],
            ["largest per-band damping difference", f"{worst_damp:.1e}", ""]],
           ["", "ours", "pyroomacoustics"])
    # Closed form: the first-order floor image of a source at height z is at -z, damped by sqrt(1 - alpha_floor).
    floor = np.flatnonzero((ours.wall_hits.sum(1) == 1) & (ours.wall_hits[:, 4] == 1))[0]
    print("  floor image:", ours.images_m[floor], "expected z =", -src[2],
          "| damping == sqrt(1-alpha):", np.allclose(ours.wall_pressure[floor], np.sqrt(1 - np.array(alpha["z_min"]))))


# ------------------------------------------------------------------------------------------- delay
def _response(taps, centre, w):
    k = np.arange(len(taps)) - centre
    return np.exp(-1j*np.outer(w, k)) @ taps


def delay():
    """Fractional-delay kernels: DC gain, delay error and ripple against the ideal e^{-jwd}."""
    import pyroomacoustics as pra
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.synth import ImagePaths, SynthConfig, path_kernels
    fs, half = 48000, SynthConfig().filter_half_length
    w = np.linspace(1e-4, np.pi, 4000)
    rows = []
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        paths = ImagePaths(np.zeros((1, 3)), np.zeros((1, 6), int), np.array([1.]), np.array([(200 + frac)/fs]),
                           np.ones((1, 7)), np.ones((1, 7)), np.ones((1, 7)), np.array([1.]))
        _, _, kernel = next(path_kernels(RoomModel(shoebox_dims_m=(5, 4, 3)), paths, fs, SynthConfig(air_absorption=False)))
        ours = kernel[0]
        theirs = np.asarray(pra.utilities.fractional_delay(frac))
        for name, taps, centre in (("ours", ours, half), ("pyroomacoustics", theirs, (len(theirs) - 1)/2)):
            h = _response(taps, centre, w)
            ideal = np.exp(-1j*w*frac)
            band = w <= 0.9*np.pi            # up to 0.45 fs
            rows.append([f"{frac:.2f}", name, len(taps), f"{taps.sum():.6f}",
                         f"{np.max(np.abs(20*np.log10(np.abs(h[band])))):.3f}",
                         f"{np.max(np.abs(h[band] - ideal[band])):.1e}",
                         f"{np.max(np.abs(-np.unwrap(np.angle(h[w <= .5*np.pi] * np.exp(1j*w[w <= .5*np.pi]*frac))) / w[w <= .5*np.pi])):.1e}"])
    _table(rows, ["delay", "kernel", "taps", "DC gain", "max |ripple| dB to 0.45 fs", "max |H - ideal| to 0.45 fs",
                  "max delay error (samples) to 0.25 fs"])


# ------------------------------------------------------------------------------------------- decay
def _decay_rir(fs, t60, seconds, seed, noise_db=None, lead_s=0.):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds*fs))/fs
    h = rng.standard_normal(t.size)*np.exp(-3*np.log(10)*t/t60)          # energy falls 60 dB in t60
    if noise_db is not None:
        h = h + rng.standard_normal(t.size)*10**(noise_db/20)
    return np.r_[np.zeros(int(lead_s*fs)), h]


def decay():
    """T20/T30/EDT on exponential decays with a known T60, without and with a noise floor."""
    import pyfar as pf
    import pyrato
    import pyroomacoustics as pra
    from sensorsim.acoustics.room.metrics import fit_decay, room_metrics, schroeder_decay
    fs, truth = 48000, 0.6
    rows = []
    for label, noise_db in (("no noise", None), ("noise at -70 dB", -70.)):
        est = {k: [] for k in ("ours T30", "ours T20", "ours EDT", "pra T30 (decay_db=30)", "pyrato T30", "pyrato EDT")}
        for seed in range(8):
            h = _decay_rir(fs, truth, 2.5, seed, noise_db)
            curve = schroeder_decay(h, fs, noise_power=0. if noise_db is None else None)
            est["ours T30"].append(fit_decay(curve, -5, -35).rt60_s)
            est["ours T20"].append(fit_decay(curve, -5, -25).rt60_s)
            est["ours EDT"].append(fit_decay(curve, 0, -10).rt60_s)
            est["pra T30 (decay_db=30)"].append(pra.experimental.measure_rt60(h, fs, decay_db=30))
            sig = pf.Signal(h, fs)
            edc = (pyrato.edc.schroeder_integration(sig) if noise_db is None
                   else pyrato.edc.energy_decay_curve_chu_lundeby(sig, freq="broadband", normalize=True, is_energy=False))
            est["pyrato T30"].append(float(np.squeeze(pyrato.parameters.reverberation_time_linear_regression(edc, "T30"))))
            est["pyrato EDT"].append(float(np.squeeze(pyrato.parameters.reverberation_time_linear_regression(edc, "EDT"))))
        for k, v in est.items():
            v = np.array([np.nan if x is None else x for x in v], float)
            rows.append([label, k, f"{np.nanmean(v):.4f}", f"{100*(np.nanmean(v)/truth - 1):+.2f}%", f"{np.nanstd(v):.4f}"])
    _table(rows, ["case", "estimator", "mean T (s)", "bias vs 0.600 s", "spread"])
    # ISO 3382: the decay starts at the direct sound. 20 ms of silence before it must not change EDT.
    h = _decay_rir(fs, truth, 2.5, 0, None, lead_s=0.02)
    m = room_metrics(h, fs, noise_power=0.)
    print(f"  onset found at {m.onset_sample} (true {int(0.02*fs)}); EDT {m.edt.rt60_s:.4f} s with leading silence")


# ----------------------------------------------------------------------------------------- clarity
def _expected_band_energy(envelope_energy, ir):
    """Expected energy of white noise shaped by an envelope, then filtered: conv(envelope^2, h^2)."""
    from scipy.signal import fftconvolve
    return fftconvolve(envelope_energy, ir**2)[:len(envelope_energy)]


def _clarity(energy, onset, fs, ms, start=None):
    """Early/late energy split `ms` after the onset; integration starts at `start` (default: the onset)."""
    split = onset + int(np.ceil(ms/1000*fs))
    return 10*np.log10(energy[onset if start is None else start:split].sum()/energy[split:].sum())


def clarity():
    """C50/C80 per octave from exact expected energies: /2 (ours), /1 (causal), pyfar's filters."""
    from scipy.signal import fftconvolve, sosfilt
    import pyfar as pf
    import pyrato
    from sensorsim.acoustics.room.metrics import filter_span, octave_sos
    fs, t60 = 48000, 0.5
    onset = max(filter_span(fs, b) for b in BANDS) + 480                  # room for /2's integration start
    k = 6*np.log(10)/t60                                                  # energy decay rate, 13.8/T60
    truth = {ms: 10*np.log10(np.exp(k*ms/1000) - 1) for ms in (50, 80)}
    t = np.arange(int(2.0*fs) + onset)/fs
    envelope = np.r_[np.zeros(onset), np.exp(-k*t)][:len(t)]
    impulse = np.zeros(int(1.0*fs)); impulse[0] = 1.
    rows = []
    for band in BANDS:
        causal = sosfilt(octave_sos(fs, band), impulse)
        zero_phase = fftconvolve(causal, causal[::-1])                  # filtfilt = h * h(-t), centred on 0
        centre = len(causal) - 1
        e_zp = fftconvolve(envelope, zero_phase**2)[centre:centre + len(envelope)]
        pyfar_ir = np.squeeze(pf.dsp.filter.fractional_octave_bands(pf.signals.impulse(len(impulse), sampling_rate=fs),
                                                                     1, frequency_range=(band, band)).time)
        span = filter_span(fs, band)
        e = {"/2 (ours): zero-phase, from onset - filter span": (e_zp, onset - span),
             "/1: causal, from the onset": (_expected_band_energy(envelope, causal), onset),
             "zero-phase, from the onset": (e_zp, onset),
             "pyfar octave filter, causal": (_expected_band_energy(envelope, pyfar_ir), onset)}
        for name, (energy, start) in e.items():
            rows.append([band, name, f"{_clarity(energy, onset, fs, 50, start) - truth[50]:+.2f}",
                         f"{_clarity(energy, onset, fs, 80, start) - truth[80]:+.2f}"])
        group = np.angle(np.fft.rfft(causal, 1 << 20))
        f = np.fft.rfftfreq(1 << 20, 1/fs); i = int(np.argmin(abs(f - band)))
        rows.append([band, f"(our filter's group delay at {band} Hz: "
                     f"{-np.gradient(np.unwrap(group), 2*np.pi*f[1])[i]*1000:.1f} ms)", "", ""])
    print(f"  closed form for T60 {t60} s: C50 {truth[50]:.2f} dB, C80 {truth[80]:.2f} dB (exact expected energies, no noise draws)")
    _table(rows, ["band Hz", "method", "C50 error dB", "C80 error dB"])
    # How each convention's C50 error depends on the reverberation time (dB).
    rows = []
    for t60b in (0.3, 0.5, 1.0, 2.0):
        kb = 6*np.log(10)/t60b
        envb = np.r_[np.zeros(onset), np.exp(-kb*t)][:len(t)]
        true50 = 10*np.log10(np.exp(kb*0.05) - 1)
        for name in ("/2", "/1"):
            cells = []
            for b in (125, 250, 500, 1000, 2000):
                h = sosfilt(octave_sos(fs, b), impulse)
                if name == "/1":
                    cells.append(_clarity(_expected_band_energy(envb, h), onset, fs, 50) - true50)
                else:
                    g = fftconvolve(h, h[::-1])
                    e_zp = fftconvolve(envb, g**2)[len(h) - 1:len(h) - 1 + len(envb)]
                    cells.append(_clarity(e_zp, onset, fs, 50, onset - filter_span(fs, b)) - true50)
            rows.append([f"{t60b:g} s", name] + [f"{c:+.2f}" for c in cells])
    _table(rows, ["T60", "policy", "125 Hz", "250 Hz", "500 Hz", "1 kHz", "2 kHz"])
    # pyrato computes clarity from an energy decay curve: check its formula on the broadband closed form.
    edc = np.cumsum(envelope[onset:][::-1])[::-1]
    print(f"  pyrato.parameters.clarity on the exact broadband EDC: C50 "
          f"{float(np.squeeze(pyrato.parameters.clarity(pf.TimeData(edc/edc[0], t[:len(edc)]), 50))):.3f} dB"
          f" (closed form {truth[50]:.3f})")


# ------------------------------------------------------------------------------------------- sweep
def bands():
    """Per-octave decay and clarity: room-measured-metrics/1 (causal filter, from the onset) against /2
    (decay: time-reversed filter; energy: zero-phase, from onset - filter span), on exact expected energies."""
    from scipy.signal import fftconvolve, sosfilt
    from sensorsim.acoustics.room.metrics import DecayCurve, filter_span, fit_decay, octave_sos
    fs, onset, n = 48000, 24000, int(3.8*48000)
    impulse = np.zeros(fs); impulse[0] = 1.

    def measure(e, start, origin):
        """EDC and integration from `start`; time origin and clarity split at `origin` (the direct sound)."""
        x = e[start:]
        edc = np.cumsum(x[::-1])[::-1]
        curve = DecayCurve(np.arange(len(x))/fs, edc, 10*np.log10(np.maximum(edc/edc[0], 1e-300)), 0., len(x),
                           0., 0., None, True, None, 0)
        out = {"EDT %": fit_decay(curve, 0, -10).rt60_s, "T30 %": fit_decay(curve, -5, -35).rt60_s}
        for ms in (50, 80):
            early = e[start:origin + int(np.ceil(ms/1000*fs))].sum()
            out[f"C{ms} dB"] = 10*np.log10(early/(edc[0] - early))
        out["Ts ms"] = 1000*float(np.dot((np.arange(len(x)) + start - origin)/fs, x)/edc[0])
        return out

    def responses(t60, drr_db, band):
        k = 6*np.log(10)/t60
        t = (np.arange(n) - onset)/fs
        r = np.where(t >= 0, k*np.exp(-k*np.maximum(t, 0))/fs, 0.)          # diffuse energy per sample, total 1
        d = 0. if drr_db is None else 10**(drr_db/10)                         # direct energy
        h = sosfilt(octave_sos(fs, band), impulse)
        h = h[:np.searchsorted(np.cumsum(h*h), (1 - 1e-10)*np.sum(h*h)) + 1]
        h2, L = h*h, len(h)
        truth = r.copy(); truth[onset] += d
        forward = fftconvolve(r, h2)[:n]; forward[onset:onset + L] += d*h2
        reverse = fftconvolve(r, h2[::-1])[L - 1:L - 1 + n]; reverse[onset - L + 1:onset + 1] += d*h2[::-1]
        g = fftconvolve(h, h[::-1]); g2 = g*g
        zero_phase = fftconvolve(r, g2)[L - 1:L - 1 + n]; zero_phase[onset - L + 1:onset + L] += d*g2
        span = filter_span(fs, band)
        v1 = measure(forward, onset, onset)
        decay2, energy2 = measure(reverse, onset, onset), measure(zero_phase, onset - span, onset)
        v2 = {m: (decay2 if m in ("EDT %", "T30 %") else energy2)[m] for m in v1}
        return measure(truth, onset, onset), v1, v2

    columns = (125, 250, 500, 1000, 2000, 4000)
    worst = {}
    for t60 in (0.15, 0.3, 0.6, 1.2):
        for drr_db in (None, -6., 0., 6.):
            for band in columns:
                truth, v1, v2 = responses(t60, drr_db, band)
                for policy, values in (("/1", v1), ("/2", v2)):
                    for metric, value in values.items():
                        err = 100*(value/truth[metric] - 1) if "%" in metric else value - truth[metric]
                        key = (metric, policy, band)
                        worst[key] = max(worst.get(key, 0.), abs(err))
    print("  worst |error| against the closed form over T60 0.15-1.2 s and direct-to-reverberant ratio none/-6/0/+6 dB")
    rows = [[metric, policy] + [f"{worst[(metric, policy, b)]:.2f}" for b in columns]
            for metric in ("EDT %", "T30 %", "C50 dB", "C80 dB", "Ts ms") for policy in ("/1", "/2")]
    _table(rows, ["metric", "policy", *[f"{b} Hz" for b in columns]])


def sweep():
    """Exponential sweep law vs pyfar, harmonic advance L ln k, and the analytic inverse's in-band gain."""
    import pyfar as pf
    from sensorsim.acoustics.room.rir import ESSExcitation, deconvolve_ess, ess_inverse, generate_ess
    fs, f1, f2, dur = 48000, 50., 20000., 2.0
    spec = ESSExcitation(f_start_hz=f1, f_end_hz=f2, duration_s=dur, amplitude=1., fade_s=0.)
    ours = generate_ess(fs, spec)
    theirs = np.squeeze(pf.signals.exponential_sweep_time(len(ours), (f1, f2), n_fade_out=0, sampling_rate=fs).time)
    print(f"  pyfar {pf.__version__}: max |ours - pyfar| = {np.max(np.abs(ours - theirs)):.2e} over {len(ours)} samples")
    L = dur/np.log(f2/f1)
    for k in (2, 3):                    # harmonic k of instantaneous frequency f1 e^{t/L} appears L ln k earlier
        print(f"  harmonic {k}: advance L ln k = {1000*L*np.log(k):.3f} ms")
    inverse = ess_inverse(fs, spec)
    n = 1 << int(np.ceil(np.log2(2*len(ours))))
    H = np.fft.rfft(ours, n)*np.fft.rfft(inverse, n)
    f = np.fft.rfftfreq(n, 1/fs)
    inband = (f > 2*f1) & (f < f2/1.5)
    mag = 20*np.log10(np.abs(H[inband]))
    print(f"  |X*F| in band {2*f1:.0f} Hz..{f2/1.5:.0f} Hz: mean {mag.mean():+.4f} dB, ripple {mag.max() - mag.min():.4f} dB")
    edge = (f > f1) & (f < 1.2*f1)
    print(f"  near f1 ({f1:.0f}..{1.2*f1:.0f} Hz) the gain is {20*np.log10(np.abs(H[edge])).mean():+.2f} dB (sweep band edge)")
    # Closed form: a direct path and one reflection (0.5 at +3.2 ms) are recovered at their amplitudes.
    ir = np.zeros(int(0.1*fs)); ir[480] = 1.; ir[480 + int(0.0032*fs)] = 0.5
    rec = np.convolve(ours, ir)
    lin = deconvolve_ess(np.r_[rec, np.zeros(fs)], fs, spec, rir_duration_s=0.05).linear
    print(f"  recovered direct {lin[480]:.4f} (1), reflection {lin[480 + int(0.0032*fs)]:.4f} (0.5)")
    # pyfar's regularized spectral division on the same recording
    print(f"  closed form for a band-limited unit impulse: peak 2(f2 - f1)/fs = {2*(f2 - f1)/fs:.4f}")
    out = pf.dsp.deconvolve(pf.Signal(rec, fs), pf.Signal(ours, fs), frequency_range=(f1, f2))
    t = np.squeeze(out.time)
    print(f"  pyfar.dsp.deconvolve (regularized spectral division): direct {t[480]:.4f}, "
          f"reflection {t[480 + int(0.0032*fs)]:.4f}")


# --------------------------------------------------------------------------------------------- air
def _python_acoustics_iso9613():
    """python-acoustics' ISO 9613-1 module, loaded by file: the package itself fails to import."""
    spec = importlib.util.find_spec("acoustics")
    if spec is None:
        return None
    path = spec.submodule_search_locations[0] + "/standards/iso_9613_1_1993.py"
    loader = importlib.util.spec_from_file_location("iso_9613_1_1993", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def air():
    """ISO 9613-1 pure-tone attenuation vs pyfar and python-acoustics."""
    import pyfar as pf
    from sensorsim.acoustics.absorption import alpha_db_per_m
    f = np.array([63., 125., 250., 500., 1000., 2000., 4000., 8000., 16000.])
    pa = _python_acoustics_iso9613()
    rows = []
    for t, rh, p in ((20., 50., 101.325), (10., 80., 101.325), (30., 20., 90.0), (-10., 70., 101.325)):
        ours = alpha_db_per_m(f, t, rh, p)
        theirs = np.asarray(pf.constants.air_attenuation(t, f, rh/100., p*1000.)[0].freq if hasattr(
            pf.constants.air_attenuation(t, f, rh/100., p*1000.)[0], "freq") else pf.constants.air_attenuation(t, f, rh/100., p*1000.)[0]).ravel()
        row = [f"{t:g} C, {rh:g}%, {p:g} kPa", f"{np.max(np.abs(theirs/ours - 1)):.1e}"]
        if pa is not None:
            T = t + 273.15
            h = pa.molar_concentration_water_vapour(rh, pa.saturation_pressure(T, 101.325, 273.16), p)
            a = pa.attenuation_coefficient(p, T, 101.325, 293.15, pa.relaxation_frequency_nitrogen(p, T, h),
                                           pa.relaxation_frequency_oxygen(p, h), f)
            row.append(f"{np.max(np.abs(a/ours - 1)):.1e}")
        rows.append(row)
    _table(rows, ["atmosphere", "max rel. diff pyfar", "max rel. diff python-acoustics"])


# ------------------------------------------------------------------------------------------- speed
def speed():
    """Speed of sound: our humid-air Cramer formula (the default) and dry-air formula vs pyfar and pyroomacoustics."""
    import pyfar as pf
    import pyroomacoustics as pra
    from sensorsim.core.geometry import speed_of_sound, speed_of_sound_cramer
    rows, worst = [], 0.
    for t, rh in ((0., 50.), (20., 0.), (20., 50.), (25., 80.), (30., 60.)):
        theirs = float(pf.constants.speed_of_sound_cramer(t, rh/100.))
        same_co2 = speed_of_sound_cramer(t, rh, co2_ppm=425.19)                 # pyfar's default CO2 fraction
        worst = max(worst, abs(same_co2 - theirs))
        rows.append([f"{t:g} C, {rh:g}%", f"{speed_of_sound_cramer(t, rh):.3f}", f"{theirs:.3f}", f"{speed_of_sound(t):.3f}",
                     f"{pra.parameters.calculate_speed_of_sound(t, rh, 101.325):.3f}",
                     f"{float(pf.constants.speed_of_sound_simple(t)):.3f}"])
    _table(rows, ["air", "ours: Cramer (default)", "pyfar Cramer", "ours: dry-air-core/1", "pyroomacoustics", "pyfar simple"])
    print(f"  ours vs pyfar Cramer at the same CO2 fraction: max |difference| {worst:.1e} m/s")
    c_dry, c_humid = speed_of_sound(20.), speed_of_sound_cramer(20., 50.)
    dt = 3./c_dry - 3./c_humid
    print(f"  20 C, 50% RH over 3 m: dry {c_dry:.2f} vs humid {c_humid:.2f} m/s -> {dt*1e6:.1f} us = {dt*48000:.2f} samples at 48 kHz")


# ------------------------------------------------------------------------------------------ reverb
def reverb():
    """Sabine/Eyring T60 vs pyroomacoustics' rt60_theory for one room, air absorption off."""
    import pyroomacoustics as pra
    from sensorsim.acoustics.room.model import RoomModel, uniform_bands
    dims = (5.4, 4.2, 2.7)
    for alpha in (0.1, 0.3, 0.6):
        model = RoomModel(shoebox_dims_m=dims, absorption=uniform_bands(alpha), temperature_c=20.)
        room = pra.ShoeBox(dims, fs=16000, materials=pra.Material(alpha), max_order=1, air_absorption=False,
                           temperature=20., humidity=50.)                  # the same air as our default
        rows = []
        for method in ("sabine", "eyring"):
            ours = model.t60_estimates(method, air_absorption=False)[3]
            theirs = room.rt60_theory(formula=method)
            rows.append([alpha, method, f"{ours:.4f}", f"{float(np.mean(theirs)):.4f}",
                         f"{100*(ours/float(np.mean(theirs)) - 1):+.2f}%", f"c ours {model.sound_speed_m_s:.2f}, pra {room.c:.2f}"])
        _table(rows, ["alpha", "formula", "ours T60 (s)", "pyroomacoustics", "difference", "sound speed"])


SECTIONS = {"images": images, "delay": delay, "decay": decay, "clarity": clarity, "bands": bands, "sweep": sweep,
            "air": air, "speed": speed, "reverb": reverb}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(SECTIONS):
        print(f"== {name}: {(SECTIONS[name].__doc__ or '').strip().splitlines()[0]}")
        SECTIONS[name]()
