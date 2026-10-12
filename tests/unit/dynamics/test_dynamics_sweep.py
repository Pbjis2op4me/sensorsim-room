# SPDX-License-Identifier: Apache-2.0
"""Metrics use known signals; tracing must preserve the production stream and RNG."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import nn, recognizer
from tools import dynamics_sweep as sweep


def corpus():
    rng = np.random.default_rng(93)
    x = rng.normal(0, .02, 16000)
    return SimpleNamespace(classes=['knock'], targets={'knock': [x]},
                           spans={'knock': [(0, 16000)]},
                           background=[rng.normal(0, .001, 36000)])


@pytest.mark.parametrize('mic_probability', [0, 1])
def test_trace_is_byte_identical_to_make_example_and_preserves_rng(mic_probability):
    cfg = dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), p_background_only=0, p_mic_chain=mic_probability, snr_db=[3, 3])
    rng = np.random.default_rng(57)
    control_rng = np.random.default_rng(57)
    expected, label = nn.make_example(corpus(), control_rng, cfg)
    actual, actual_label, trace = sweep.trace_example(corpus(), rng, cfg)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_label, label)
    assert rng.bit_generator.state == control_rng.bit_generator.state
    assert 'floor_full' in trace
    assert trace['chain']['profile_id'] == 'pi5_inmp441/1'
    # Main's capture keeps the recording and adds the rendered increment without requantizing.
    expected_floor = np.clip(trace['input_floor'], -1, 1)
    np.testing.assert_array_equal(trace['floor'], expected_floor[16000:].astype(np.float32))


def test_sine_dbfs_conventions_and_floor_ratio():
    fs = 16000
    x = .01 * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)
    assert sweep.levels(x, fs)['rms_dbfs'] == pytest.approx(-43.0103, abs=1e-4)
    assert sweep.levels(x, fs)['a_sine_dbfs'] == pytest.approx(-40, abs=.01)
    assert sweep.levels(x, fs)['crest_db'] == pytest.approx(3.0103, abs=1e-4)


def test_audibility_rejects_silence_and_accepts_audible_event():
    f = sweep.edge.Features(nn._feature_cfg((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"})))
    x = .01 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)
    assert sweep.audibility(np.zeros_like(x), x, f)['audible_frames'] == 0
    assert sweep.audibility(x, .1*x, f)['audible_frames'] == f.frames
    assert sweep.audibility(.1*x, x, f)['audible_frames'] == 0


def test_mask_ledger_counts_time_and_frequency_separately():
    power = np.ones((8, 10))
    mask = np.zeros_like(power, dtype=bool)
    mask[:, :3] = True
    mask[0, :] = True
    ledger = sweep.mask_metrics(mask, power)
    assert ledger['masked_event_frame_fraction'] == pytest.approx(.3)
    assert ledger['masked_event_tf_energy_fraction'] == pytest.approx(31/80)


def test_repeat_detector_ignores_silence_but_finds_tiled_blocks():
    assert sweep.fingerprints(np.zeros(16000), 16000)['repeated_block_fraction'] == 0
    x = np.tile(np.random.default_rng(3).normal(size=320), 50)
    assert sweep.fingerprints(x, 16000)['repeated_block_fraction'] == pytest.approx(49/50)


def test_trace_restores_functions_after_failure():
    original = nn._event_window
    c = corpus()
    c.background = []
    with pytest.raises(ValueError):
        sweep.trace_example(c, np.random.default_rng(0), dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), snr_db=[3,3]))
    assert nn._event_window is original


def test_prereg_hash_mismatch_refuses_run(tmp_path):
    p = tmp_path/'preregistration.yaml'
    p.write_text('{}')
    h = tmp_path/'hash.txt'
    h.write_text('0'*64)
    with pytest.raises(ValueError, match='hash'):
        sweep.verify_preregistration(p, h)


def test_quiet_pcm_loss_is_reported_not_hidden():
    x = .000003*np.sin(2*np.pi*1000*np.arange(16000)/16000)
    y = np.rint(x*32768)/32768
    assert not y.any()
    assert sweep.levels(x-y, 16000)['a_sine_dbfs'] > -120


def test_complete_sweep_writes_boolean_quantiles_and_is_repeatable(tmp_path):
    import hashlib
    import json
    from scipy.io import wavfile
    rng = np.random.default_rng(8)
    wavfile.write(tmp_path/'target.wav', 16000, rng.normal(0, .01, 16000).astype(np.float32))
    wavfile.write(tmp_path/'floor.wav', 16000, rng.normal(0, .001, 36000).astype(np.float32))
    config = {'cases': [{'id':'test', 'source_group':'source', 'label':'knock',
                         'evidence_kind':'model-based','proxy':'unit',
                         'target':'target.wav','floor':'floor.wav'}],
              'profiles':['random_chain'], 'seeds':[4,5,6], 'snr_db':[3], 'augmentations':['off','spec']}
    (tmp_path/'config.json').write_text(json.dumps(config))
    p=tmp_path/'prereg.yaml'; p.write_text('{}')
    h=tmp_path/'hash.txt'; h.write_text(hashlib.sha256(p.read_bytes()).hexdigest())
    for name in ('one','two'):
        result=sweep.run(tmp_path/'config.json',p,h,tmp_path/name)
        assert result['rows']==6
        assert result['measured_rows']==0
    for name in ('windows.csv','cells.csv','audit.json'):
        assert (tmp_path/'one'/name).read_bytes()==(tmp_path/'two'/name).read_bytes()


def test_direct_reference_windows_use_device_hop_and_reject_group_overlap(tmp_path):
    from scipy.io import wavfile
    x=np.sin(2*np.pi*1000*np.arange(32000)/16000).astype(np.float32)*.01
    wavfile.write(tmp_path/'take.wav',16000,x)
    f=sweep.edge.Features(nn._feature_cfg((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"})))
    ref={'source_group':'heldout-1','label':'knock','evidence_kind':'measured',
         'split':'heldout','audio':'take.wav'}
    rows=sweep.reference_windows([ref],tmp_path,{'train-1'},f)
    assert [r['start_sample'] for r in rows]==[0,4000,8000,12000,16000]
    assert all(r['evidence_kind']=='measured' for r in rows)
    with pytest.raises(ValueError,match='overlaps'):
        sweep.reference_windows([ref],tmp_path,{'heldout-1'},f)
