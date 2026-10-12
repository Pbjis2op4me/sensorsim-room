"""L4 checks on the merged tree (main a620399 + w3-negatives-fix).

1. Positives: conditional capture vs single-capture float render of a KNOWN analog background,
   and main's old floor(b)+floor(e) method; errors vs analog truth; paired C2ST vs single capture.
2. Negatives: nn.make_example background-only windows are sample-identical to the recording crop.
argv: out.json
"""
import copy, json, sys
import numpy as np
from scipy import signal
from tests.validation.bench_negatives.final_common import FS, POS, PROFILE, bed_analog, c2st, neutral_audio, rir, stats120
from sensorsim import iot_mic, nn
from sensorsim.training_device import conditional_capture
from sensorsim.recognizer import _crop

def main(output_file):
    out = {'positives': [], 'negatives': {}}
    for seed in (1701, 1702, 1703):
        single_w, cond_w, old_w = [], [], []
        err = {'single': [], 'conditional': [], 'old_main': []}
        for g in range(144):
            rng = np.random.default_rng([seed, g, 21])
            prof = PROFILE.with_variation(rng)
            a = prof.apply(bed_analog(1000*seed+g), FS, rng, input_reference_spl_db=94., quantize=False)  # analog incl. self-noise
            b = iot_mic.quantize_profile(prof, a)                                                          # the recording
            wet = signal.oaconvolve(neutral_audio(seed, g), rir(POS[g % 3], 7*g+seed))[:7*FS]
            ev = np.zeros(10*FS); ev[3*FS:3*FS+len(wet)] = wet
            p = float(np.mean(wet[:FS]**2)); level = float(rng.uniform(60, 80))
            ev *= 10**((level-94)/20)/np.sqrt(2*p)
            e = prof.apply(ev, FS, rng, input_reference_spl_db=94., include_self_noise=False, quantize=False)
            t = a + e
            single = iot_mic.quantize_profile(prof, t)
            cond = conditional_capture(b, e, np.random.default_rng([seed, g, 22]), prof)
            old = iot_mic.quantize_profile(prof, b + e)
            s = slice(3*FS, 4*FS)
            for k, y in (('single', single), ('conditional', cond), ('old_main', old)):
                err[k].append((y[s]-t[s])*32768)
            single_w.append(single[s]); cond_w.append(cond[s]); old_w.append(old[s])
        summ = {k: {'mean_lsb': float(np.mean(np.concatenate(v))), 'rms_lsb': float(np.sqrt(np.mean(np.concatenate(v)**2)))} for k, v in err.items()}
        fs_, fc, fo = stats120(single_w), stats120(cond_w), stats120(old_w)
        out['positives'].append({'seed': seed, 'error_vs_analog_truth': summ,
                                 'c2st_conditional_vs_single': c2st(fc, fs_, seed),
                                 'c2st_old_main_vs_single': c2st(fo, fs_, seed)})
        print(json.dumps(out['positives'][-1]), flush=True)


    # --- negatives through the real make_example (background-only), fake corpus
    class Corpus:
        pass


    def make_corpus():
        c = Corpus()
        rng = np.random.default_rng(5)
        a = PROFILE.apply(bed_analog(77, seconds=30), FS, rng, input_reference_spl_db=94., quantize=False)
        c.background = [iot_mic.quantize_profile(PROFILE, a).astype(np.float32)]
        c.classes = ['knock']
        c.targets = {'knock': [np.r_[np.zeros(4000), np.hanning(800)*np.random.default_rng(1).standard_normal(800)*.1, np.zeros(4000)].astype(np.float32)]}
        c.spans = {'knock': [(4000, 4800)]}
        return c


    corpus = make_corpus()
    cfg = dict(nn.DEFAULTS, p_background_only=1.0)
    same = 0; total = 200
    for k in range(total):
        rng = np.random.default_rng([9, k]); clone = copy.deepcopy(rng)
        audio, label = nn.make_example(corpus, rng, cfg)
        bg = corpus.background[int(clone.integers(len(corpus.background)))]
        crop = _crop(bg, 2*FS, clone).astype(float)
        same += bool(np.array_equal(audio.astype(np.float64), crop[FS:].astype(np.float32).astype(np.float64)) and not label.any())
    out['negatives'] = {'make_example_background_only_windows': total, 'sample_identical_to_recording_crop': same}
    print(json.dumps(out['negatives']))
    open(output_file, 'w').write(json.dumps(out, indent=1) + '\n')
