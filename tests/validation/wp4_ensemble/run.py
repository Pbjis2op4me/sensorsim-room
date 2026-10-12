"""Shared WP4 scoring; the legacy dataset runner is retired.

HDF5 development tools live in review_acceptance.py. No archive data ship.
"""

_RETIRED = "legacy WP4 dataset route is retired; use the HDF5 development tools"


def read_pair(*args, **kwargs):
    """Refuse retired callers before any input read."""
    raise RuntimeError(_RETIRED)


def score_cells(truth, prior, posterior, state, source, seed):
    rows=[]
    for band in (500,1000,2000,4000):
        for metric in ('t30_s','edt_s','c50_db','drr_db'):
            key=f'{band}.{metric}'
            row=dict(label='proxy result',state=state,source=source,seed=seed,band_hz=band,metric=metric,
                     truth=truth.get(key),prior=prior.get(key),posterior=posterior.get(key),covered=False)
            if all(key in x for x in (truth,prior,posterior)):
                value=truth[key]
                for name, intervals in [('prior',prior),('posterior',posterior)]:
                    lo,mid,hi=intervals[key]
                    # Dimensionless score: normalize seconds by truth and dB by
                    # the declared discrepancy so unlike metrics are not summed.
                    scale=max(abs(value),.025) if metric.endswith('_s') else 4. if metric=='c50_db' else 6.
                    score=hi-lo+20*max(lo-value,0)+20*max(value-hi,0)
                    row[name+'_winkler']=score/scale
                lo,mid,hi=posterior[key]
                row['covered']=lo<=value<=hi
                if metric=='t30_s':row['relative_half_width']=(hi-lo)/(2*truth[key])
            rows.append(row)
    return rows



def main():
    """Refuse the retired command before reading registrations or creating output."""
    raise RuntimeError(_RETIRED)


if __name__ == '__main__':
    main()
