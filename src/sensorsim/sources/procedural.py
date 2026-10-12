"""Procedural sources are unavailable pending a validated generative model.

The former implementation was gain replay. It is explicitly available through
sources.replay and cannot supply WP6(c) or C2ST diversity evidence.
"""


def fit_procedural(*args, **kwargs):
    raise ValueError('procedural sources are unavailable; use opt-in sources.replay.fit_replay '
                     'for gain replay, excluded from C2ST and diversity claims')


def generate_procedural(*args, **kwargs):
    raise ValueError('procedural sources are unavailable; use opt-in sources.replay.generate_replay '
                     'for gain replay, excluded from C2ST and diversity claims')
