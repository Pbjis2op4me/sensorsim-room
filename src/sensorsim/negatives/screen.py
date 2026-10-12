# SPDX-License-Identifier: Apache-2.0
"""Screen-only comparison policy. Selection failure cannot suppress endpoints."""
FIXED_DETECTOR = dict(on=.5, off_ratio=.6, smooth_hops=3, min_hops=2, hop_s=.25)


def scoring_plan(reports, expected_runs, *, terminal_step):
    if set(reports) != set(expected_runs) or not reports:
        raise ValueError('all registered candidates must finish before any audit scoring')
    statuses = [r.get('selection', {}).get('status') for r in reports.values()]
    if any(s not in ('feasible', 'infeasible') for s in statuses):
        raise ValueError('missing or unexpected selection outcome; no silent fallback')
    fallback = 'infeasible' in statuses
    return dict(mode='terminal_fixed' if fallback else 'selected',
                reason='at least one cell infeasible' if fallback else 'every cell feasible',
                terminal_step=terminal_step, detector=FIXED_DETECTOR if fallback else None,
                selection_outcomes={k:r['selection']['status'] for k,r in reports.items()})
