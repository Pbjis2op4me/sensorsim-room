"""Independent-RNG paired path C2ST: positive helper (make_example's mixer) vs negative composer.

Same dry sources, same RIR (hash-checked), same background recording / analog bed.
Level, device-variation and capture draws are INDEPENDENT between paths (unlike the lane's
registered test, which shares one RNG and therefore yields byte-identical outputs).
argv: out.json
"""
import json, sys
import numpy as np
from scipy import signal
from tests.validation.bench_negatives.final_common import FS, POS, PROFILE, ROOM, RX, bed_analog, c2st, h, neutral_audio, rir, stats120
from sensorsim import iot_mic, nn
from sensorsim.training_device import mix_device_audio
from sensorsim.negatives.composer import Event, RenderPath, captured_context, render_context
from sensorsim.negatives.procedural import Source

PATH = RenderPath(ROOM, RX, PROFILE, PROFILE.profile_id)


def run(seed, domain, shift_db=0.):
    pos_w, neg_w, maxdiff = [], [], []
    for g in range(144):
        x = neutral_audio(seed, g)
        src = Source(x, 'neutral', f'n{g}', {'fs': FS, 'acoustic_stage': 'dry', 'license_id': 'Apache-2.0'})
        draw_seed = int(np.random.SeedSequence([seed, g, 51]).generate_state(1)[0])
        rir_seed = int(np.random.SeedSequence([draw_seed, 0, 11]).generate_state(1)[0])
        r = rir(POS[g % 3], rir_seed)
        wet = signal.oaconvolve(x, r)
        event = np.zeros(10*FS); take = min(len(wet), 7*FS); event[3*FS:3*FS+take] = wet[:take]
        analog = bed_analog(1000*seed+g)
        if domain == 'device_recorded':
            cap_rng = np.random.default_rng([seed, g, 3])
            background = PROFILE.with_variation(cap_rng).apply(analog, FS, cap_rng, input_reference_spl_db=94.)
        else:
            background = analog
        cfg = dict(nn.DEFAULTS, background_domain=domain)
        pos = mix_device_audio(background, event, (3*FS, 3*FS+take), np.random.default_rng([seed, g, 9]), cfg)
        neg_seed = int(np.random.SeedSequence([seed, g, 13]).generate_state(1)[0])
        ncfg = {'event_spl_db': [60.+shift_db, 80.+shift_db]}
        ev = Event(src, 3., POS[g % 3])
        if domain == 'device_recorded':
            capture = Source(background, 'room_tone', f'bed{g}', {
                'fs': FS, 'acoustic_stage': 'captured', 'capture_device': 'proxy-pi', 'profile_id': PROFILE.profile_id,
                'deployed_device': True, 'native_s16': True})
            ctx = captured_context(capture, fs=FS, path=PATH, seed=neg_seed, events=[ev], cfg=ncfg,
                                   room_draw=PATH.room_draw(draw_seed))
        else:
            ctx = render_context([ev], path=PATH, seed=neg_seed, background=background, cfg=ncfg,
                                 room_draw=PATH.room_draw(draw_seed))
        assert ctx.events[0]['rir_sha256'] == h(r)
        pos_w.append(pos[3*FS:4*FS]); neg_w.append(ctx.audio[3*FS:4*FS])
        maxdiff.append(float(np.max(np.abs(pos - ctx.audio))*32768))
    res = c2st(stats120(pos_w), stats120(neg_w), seed)
    lv = lambda w: np.percentile([20*np.log10(np.sqrt(np.mean(np.square(v)))*np.sqrt(2)) for v in w], [5, 50, 95]).round(1).tolist()
    return res | {'seed': seed, 'domain': domain, 'negative_shift_db': shift_db,
                  'max_pair_difference_lsb_median': float(np.median(maxdiff)),
                  'window_dbfs_pos_p5_p50_p95': lv(pos_w), 'window_dbfs_neg_p5_p50_p95': lv(neg_w)}


def main(output_file):
    out = []
    for seed in (1701, 1702, 1703):
        for domain in ('device_recorded', 'spl_referenced'):
            out.append(run(seed, domain)); print(json.dumps(out[-1]), flush=True)
    for domain in ('device_recorded', 'spl_referenced'):
        out.append(run(1701, domain, shift_db=5.)); print(json.dumps(out[-1]), flush=True)
    open(output_file, 'w').write(json.dumps(out, indent=1) + '\n')
