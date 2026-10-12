# SPDX-License-Identifier: Apache-2.0
"""Reporting controls: a simultaneous right/wrong detection must occupy both cells."""
from types import SimpleNamespace
import numpy as np

from sensorsim import edge
from sensorsim.evaluation.recall import evaluate_recall


def test_multilabel_confusion_keeps_wrong_only_both_and_misses(monkeypatch):
    calls = []
    events = [[], ['knock'], ['knock', 'doorbell', 'knock'], ['doorbell']]
    def detector(model, audio):
        calls.append(int(audio[0]))
        return [dict(label=c) for c in events[int(audio[0])]]
    monkeypatch.setattr(edge, 'detect', detector)
    model = SimpleNamespace(classes=['doorbell', 'knock', 'kettle'],
                            meta={'features': {'fs': 10}, 'detector': {'on': .5}})
    rows = [dict(label='doorbell', group=f'g{i}', audio=np.full(20, i)) for i in range(4)]
    result = evaluate_recall(model, rows, data_kind='proxy')
    assert calls == [0, 1, 2, 3]
    r = result['per_class']['doorbell']
    assert r['hits'] == 2 and r['wrong_class_takes'] == 2 and r['wrong_class_events'] == 3
    assert r['both_correct_and_wrong_takes'] == 1
    matrix = result['confusion_matrix']
    assert matrix['counts']['doorbell'] == {'doorbell': 2, 'knock': 2, 'kettle': 0}
    assert matrix['no_detection']['doorbell'] == 1
    assert result['takes'][2]['wrong_class_labels'] == ['knock']
    assert result['takes'][2]['confirmed_events_by_class'] == {'doorbell': 1, 'knock': 2, 'kettle': 0}


def test_repeated_group_is_one_wrong_class_trial_but_all_events_are_reported(monkeypatch):
    monkeypatch.setattr(edge, 'detect', lambda *a: [dict(label='knock'), dict(label='doorbell')])
    model = SimpleNamespace(classes=['doorbell', 'knock'], meta={'features': {'fs': 10}, 'detector': {}})
    result = evaluate_recall(model, [dict(label='doorbell', group='same', audio=np.ones(20))]*2, data_kind='proxy')
    r = result['per_class']['doorbell']
    assert r['takes'] == 1 and r['wrong_class_takes'] == 1 and r['wrong_class_recordings'] == 2
    assert result['confusion_matrix']['counts']['doorbell']['knock'] == 1
    empty = evaluate_recall(model, [], data_kind='proxy')
    assert empty['status'] == 'inconclusive' and empty['confusion_matrix']['counts'] == {}
