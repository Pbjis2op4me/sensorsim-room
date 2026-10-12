# SPDX-License-Identifier: Apache-2.0
"""Shared event power convention for mixing and partial-window compensation.

The production in-distribution mixer (``training_device.mix_events``) normalizes
on event-active samples (``training_device.active_power``: 10 ms energy gate at
-25 dB within the span). This helper delegates to that function, so changing the
convention is one change, in one place, for both paths. The explicit stress split
keeps its span-mean SNR convention, matching ``mix_device_audio``'s stress branch.
"""
import numpy as np


def level_power(event, fs, span, *, split='in_distribution'):
    if split == 'stress':
        a, b = span
        return float(np.mean(np.asarray(event, float)[a:b]**2)) if b > a else 0.
    from sensorsim.training_device import active_power
    return active_power(event, fs, span)[0]
