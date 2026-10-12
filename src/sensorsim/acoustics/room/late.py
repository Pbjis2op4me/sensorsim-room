"""Stationary causal noise and its modulated, analysed second moment.

For y=A*(e (B*w)), output modulation sits BETWEEN the two filters. The
white-excitation identity (A*B)^2 * e^2 does not apply. These routines share
that ordering without drawing a random waveform in the inverse model.
"""
from collections import OrderedDict, namedtuple
from dataclasses import dataclass, is_dataclass
from functools import update_wrapper
from threading import RLock
import sys
import numpy as np
from scipy import signal


_CacheInfo = namedtuple('CacheInfo', 'hits misses currsize bytes peak_bytes limit_bytes evictions oversize')


def _retained_bytes(value, seen=None):
    """Conservative retained size, including array owners and Python metadata."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        if value.base is not None:
            size += _retained_bytes(value.base, seen)
    elif isinstance(value, (tuple, list)):
        size += sum(_retained_bytes(x, seen) for x in value)
    elif is_dataclass(value):
        size += _retained_bytes(vars(value), seen)
    elif isinstance(value, dict):
        size += sum(_retained_bytes(k, seen)+_retained_bytes(v, seen) for k, v in value.items())
    return size


class ByteCache:
    """Thread-safe LRU with a byte budget; oversized entries are not retained.

    Values must be immutable. Array owners, byte keys and object metadata count
    towards the budget, plus 256 bytes per entry for the LRU index. This bounds
    cache storage, not caller references or working RSS. Computation runs outside
    the lock; concurrent misses may compute twice.
    """
    def __init__(self, function, limit_bytes):
        if limit_bytes < 0:
            raise ValueError('nonnegative byte budget required')
        update_wrapper(self, function)
        self.limit_bytes = int(limit_bytes)
        self._lock = RLock()
        self.cache_clear()

    def __getstate__(self):
        # Spawned fitting workers reuse the parent's immutable warm entries.
        # Locks are process-local; take a coherent snapshot before serializing.
        with self._lock:
            state = {k: v for k, v in vars(self).items() if k != '_lock'}
            state['_entries'] = self._entries.copy()
            return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = RLock()

    def cache_clear(self):
        with self._lock:
            self._entries = OrderedDict()
            self.hits = self.misses = self.bytes = self.peak_bytes = 0
            self.evictions = self.oversize = 0

    def cache_info(self):
        with self._lock:
            return _CacheInfo(self.hits, self.misses, len(self._entries), self.bytes,
                              self.peak_bytes, self.limit_bytes, self.evictions, self.oversize)

    def __call__(self, *key):
        with self._lock:
            if key in self._entries:
                self.hits += 1
                self._entries.move_to_end(key)
                return self._entries[key][0]
            self.misses += 1
        value = self.__wrapped__(*key)
        weight = _retained_bytes((key, value))+256
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key][0]
            if weight > self.limit_bytes:
                self.oversize += 1
                return value
            while self.bytes+weight > self.limit_bytes:
                _, (_, old) = self._entries.popitem(last=False)
                self.bytes -= old
                self.evictions += 1
            self._entries[key] = value, weight
            self.bytes += weight
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return value


def byte_cache(mebibytes):
    return lambda function: ByteCache(function, int(mebibytes*1024**2))


def stationary_noise(sos, length, fs, rng):
    """Discard hidden warmup; never normalize a finite random draw."""
    warm = round(.5*fs)
    return signal.sosfilt(sos, rng.standard_normal(warm+length))[warm:]


@byte_cache(1)
def _modes(blob, sections):
    """Stable SOS realization -> simple poles and residues, avoiding polynomials.

    s_i[n+1] = p_i s_i[n] + w[n], z[n] = sum_i r_i s_i[n] + d*w[n].
    Our Butterworth filters have distinct poles. No polynomial expansion of a
    low-frequency high-order transfer function is used.
    """
    sos = np.frombuffer(blob, dtype=np.float64).reshape(sections, 6).copy()
    size = 2*sections
    a = np.zeros((size,size)); c = np.zeros(size)
    for i in range(size):
        state = np.eye(size)[i].reshape(sections,2)
        y, end = signal.sosfilt(sos, [0.], zi=state)
        a[:,i], c[i] = end.ravel(), y[0]
    y, end = signal.sosfilt(sos, [1.], zi=np.zeros((sections,2)))
    poles, vectors = np.linalg.eig(a)
    residues = (c@vectors)*np.linalg.solve(vectors,end.ravel())
    if np.any(abs(poles) >= 1):
        raise ValueError("stationary noise requires stable filters")
    for x in (poles,residues): x.setflags(write=False)
    return poles, residues, float(y[0])


def _filter_modes(sos):
    sos = np.asarray(sos, dtype=np.float64)
    return _modes(sos.tobytes(), len(sos))


def expected_filtered_power(synthesis_sos, analysis_sos, envelope, *, return_state=False):
    """Exact stationary covariance propagated through output modulation.

    Synthesis state covariance is P_ij=1/(1-p_i*p_j). Cross-covariance with
    analysis state j follows q_ij[n+1]=p_i*a_j*q_ij[n]+k_i*e[n].
    Analysis covariance follows another scalar stable recurrence per pair.
    scipy's compiled filters evaluate these recurrences in O(N*state_count^2).
    Transposes here are ordinary, not Hermitian: conjugate modal pairs already
    represent a real process. The analysis starts at zero; synthesis starts
    stationary. Only roundoff-level negative powers are clipped.
    """
    e = np.asarray(envelope, float)
    if e.ndim != 1 or not np.isfinite(e).all():
        raise ValueError("finite one-dimensional envelope required")
    p, r, d = _filter_modes(synthesis_sos)
    a, b, f = _filter_modes(analysis_sos)
    if not len(e):
        out = np.empty(0)
        cross = np.zeros((len(p), len(a)), complex)
        state = np.zeros((len(a), len(a)), complex)
        return (out, cross, state) if return_state else out
    cov = 1/(1-p[:,None]*p[None,:])
    variance = float(np.real(r@cov@r+d*d))
    k = p*(cov@r)+d
    u = np.zeros((len(a),len(e)),complex)
    cross = np.zeros((len(p),len(a)),complex)
    state = np.zeros((len(a),len(a)),complex)
    for j, aj in enumerate(a):
        for i, (pi, ri, ki) in enumerate(zip(p,r,k)):
            value, end = signal.lfilter([0.,1.], [1.,-pi*aj],e, zi=np.zeros(1,complex))
            u[j] += ri*ki*value
            cross[i,j] = ki*end[0]
    out = f*f*variance*e*e + 2*f*e*np.real(b@u)
    for i, ai in enumerate(a):
        for j in range(i,len(a)):
            drive = e*(ai*u[i]+a[j]*u[j])+variance*e*e
            value, end = signal.lfilter([0.,1.], [1.,-ai*a[j]],drive, zi=np.zeros(1,complex))
            state[i,j] = state[j,i] = end[0]
            out += (1 if i==j else 2)*np.real(b[i]*b[j]*value)
    # Far-separated bands can cancel to ~1e-18 absolute power. Bound roundoff
    # against the branch's input variance, not its near-zero stopband leakage.
    roundoff = 256*np.finfo(float).eps*variance*np.max(e*e, initial=0.)
    if out.min(initial=0.) < -max(roundoff, 1e-300):
        raise ValueError("unstable expected-noise covariance")
    out = np.maximum(out,0.)
    return (out, cross, state) if return_state else out


@byte_cache(1)
def _rate_basis(synthesis_blob, analysis_blob):
    """Decay-independent rate structure shared by every edge block.

    Algebra from project research T04. Only exactly equal filter products are
    consolidated; modal states/residues are never assumed to be conjugate.
    """
    p, _, _ = _modes(synthesis_blob, len(synthesis_blob)//48)
    a, _, _ = _modes(analysis_blob, len(analysis_blob)//48)
    base = np.r_[p, a]
    base = (base[:, None]*base[None, :]).ravel()
    degree = np.r_[np.ones(len(p), int), np.zeros(len(a), int)]
    degree = (degree[:, None]+degree[None, :]).ravel()
    conjugate = base.imag < 0
    canonical = np.where(conjugate, base.conj(), base)
    mapping, bases, degrees = {}, [], []
    inverse = np.empty(len(base), int)
    for i, (value, k) in enumerate(zip(canonical, degree)):
        key = value, int(k)
        if key not in mapping:
            mapping[key] = len(bases)
            bases.append(value)
            degrees.append(k)
        inverse[i] = mapping[key]
    base, degree = np.asarray(bases), np.asarray(degrees)
    for value in (base, degree, inverse, conjugate):
        value.setflags(write=False)
    return base, degree, inverse, conjugate


# The desktop fitter trades memory for reuse across realistic observation grids.
# This is a retained-cache budget, not a whole-process RSS guarantee.
@byte_cache(1024)
def _edge_geometry(synthesis_blob, analysis_blob, lower_blob, count_blob):
    """Cache integer powers on a relative grid; the caller applies its origin."""
    base, degree, inverse, conjugate = _rate_basis(synthesis_blob, analysis_blob)
    lower = np.frombuffer(lower_blob, np.int64)
    count = np.frombuffer(count_blob, np.int64)
    widths, wi = np.unique(count, return_inverse=True)
    increments, ii = np.unique(np.diff(lower, prepend=0), return_inverse=True)
    # Integer powers avoid a logarithm branch choice for negative real bases.
    powers = np.cumprod((base[:, None]**increments)[:, ii], axis=1)
    groups = []
    for j, width in enumerate(widths):
        if width == 0:
            continue
        idx = np.flatnonzero(wi == j)
        for k in range(3):
            rows = np.flatnonzero(degree == k)
            part = powers[np.ix_(rows, idx)]
            groups.append((j, k, idx, rows, np.ascontiguousarray(part.real),
                           np.ascontiguousarray(part.imag), np.abs(part)))
    result = base, degree, tuple(groups), widths, inverse, conjugate
    def freeze(value):
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
        elif isinstance(value, tuple):
            for item in value:
                freeze(item)
    freeze(result)
    return result


def _sample_edges(edges):
    raw = np.asarray(edges)
    if (raw.ndim != 1 or len(raw) < 2 or raw.dtype.kind not in 'iuf'
            or not np.isfinite(raw).all() or np.any(raw < 0)
            or np.any(raw >= 2.**63) or np.any(raw != np.floor(raw))
            or np.any(raw[1:] < raw[:-1])):
        raise ValueError('ordered nonnegative integer sample edges required')
    return np.asarray(raw, np.int64)


@dataclass(frozen=True)
class PreparedTail:
    """Stationary synthesis / zero analysis state and its finite fade.

    Filter and envelope identity use exact normalized bytes; amplitude and ratio
    are never quantized. The modal conditioning limits remain, with observable
    fallback reasons. Exact means the same discrete model, not bitwise identity.
    """
    synthesis_blob: bytes
    analysis_blob: bytes
    prefix_blob: bytes
    amplitude: float
    ratio: float
    cumulative: np.ndarray
    coefficients: np.ndarray | None
    last_coefficient: complex
    fallback_reason: str | None

    def integrate_edges(self, edges, *, diagnostics=False):
        """Sum expected power in half-open native-sample bins, including repeats."""
        edges = _sample_edges(edges)
        prefix = np.frombuffer(self.prefix_blob, float)
        stop = len(prefix)
        result = np.diff(self.cumulative[np.minimum(edges, stop)])
        def finish(value, branch):
            trace = dict(branch=branch, peak_energy=float(np.max(value, initial=0.)))
            return (value, trace) if diagnostics else value
        if edges[-1] <= stop:
            return finish(result, 'prefix-only')
        def fallback(reason):
            envelope = np.r_[prefix, self.amplitude*self.ratio**np.arange(edges[-1]-stop)]
            power = expected_filtered_power(
                np.frombuffer(self.synthesis_blob, float).reshape(-1, 6),
                np.frombuffer(self.analysis_blob, float).reshape(-1, 6), envelope)
            return finish(np.diff(np.r_[0., np.cumsum(power)][edges]), reason)
        if self.fallback_reason:
            return fallback(self.fallback_reason)
        lower = np.maximum(edges[:-1]-stop, 0)
        count = np.maximum(edges[1:]-stop, 0)-lower
        base, degree, inverse, conjugate = _rate_basis(self.synthesis_blob, self.analysis_blob)
        coefficients = np.zeros(len(base), complex)
        np.add.at(coefficients, inverse, np.where(conjugate, self.coefficients.conj(), self.coefficients))
        magnitude = np.zeros(len(base))
        np.add.at(magnitude, inverse, abs(self.coefficients))
        rates = base*self.ratio**degree
        log_rates = np.log(rates[:, None])
        qpower = np.exp(np.log(self.ratio)*np.arange(3)[:, None]*lower)
        tail, bound = np.zeros(len(lower)), np.zeros(len(lower))
        # Bound matrix size and remove the absolute onset from reusable powers.
        # Block boundaries refer to the original observation grid, so durations
        # share complete blocks. The exact integer origin is applied below on
        # every call; it is not dropped or rounded out of the physical model.
        for start in range(0, len(lower), 256):
            idx = start+np.flatnonzero(count[start:start+256])
            if not len(idx):
                continue
            origin = int(lower[idx[0]])
            relative = lower[idx]-origin
            base, degree, groups, widths, inverse, conjugate = _edge_geometry(
                self.synthesis_blob, self.analysis_blob, relative.tobytes(), count[idx].tobytes())
            geometric = -np.expm1(log_rates*widths)/(1-rates[:, None])
            shift = base**origin
            for j, k, selected, rows, real, imag, absolute in groups:
                target = idx[selected]
                c = coefficients[rows]*shift[rows]*geometric[rows, j]
                tail[target] += qpower[k, target]*(c.real@real-c.imag@imag)
                bound[target] += qpower[k, target]*(
                    (magnitude[rows]*abs(shift[rows])*abs(geometric[rows, j]))@absolute)
        last_geom = -np.expm1(2*np.log(self.ratio)*count)/(1-self.ratio**2)
        last = self.last_coefficient*qpower[2]*last_geom
        tail += last.real
        bound += abs(last)
        roundoff = 512*np.finfo(float).eps*bound
        if (np.any(roundoff > 1e-7*max(float(np.max(abs(tail))), 1e-300))
                or np.any(tail < -roundoff)):
            return fallback('native-conditioning')
        return finish(result+np.maximum(tail, 0.), 'separable')


@byte_cache(16)
def _prepare_tail(synthesis_blob, analysis_blob, prefix_blob, amplitude, ratio):
    synthesis_sos = np.frombuffer(synthesis_blob, float).reshape(-1, 6)
    analysis_sos = np.frombuffer(analysis_blob, float).reshape(-1, 6)
    prefix = np.frombuffer(prefix_blob, float)
    nonzero = np.flatnonzero(prefix)
    first = int(nonzero[0]) if len(nonzero) else len(prefix)
    early, cross, state = expected_filtered_power(
        synthesis_sos, analysis_sos, prefix[first:], return_state=True)
    cumulative = np.r_[np.zeros(first+1), np.cumsum(early)]
    cumulative.setflags(write=False)
    def prepared(coefficients=None, last=0j, reason=None):
        return PreparedTail(synthesis_blob, analysis_blob, prefix_blob, amplitude, ratio,
                            cumulative, coefficients, last, reason)
    if first == len(prefix):
        return prepared(reason='native-empty-prefix')
    p, r, d = _filter_modes(synthesis_sos)
    a, b, f = _filter_modes(analysis_sos)
    delta = ratio*p[None, :]-a[:, None]
    poles = np.r_[ratio*p, a]
    rates = poles[:, None]*poles[None, :]
    denominator = rates-ratio*ratio
    if np.min(abs(delta)) < 1e-9 or np.min(abs(denominator)) < 1e-9:
        return prepared(reason='native-near-pole')
    g = r[None, :]/delta
    transform = np.block([[np.eye(len(p)), np.zeros((len(p), len(a)))],
                          [-g, np.eye(len(a))]])
    covariance = np.block([[amplitude**2/(1-p[:, None]*p[None, :]), amplitude*cross],
                           [amplitude*cross.T, state]])
    modal = transform@covariance@transform.T
    injection = np.r_[np.full(len(p), ratio), np.full(len(a), d)-ratio*g.sum(axis=1)]
    output = np.r_[f*r+b@g, b]
    force = amplitude**2*np.outer(injection, injection)/denominator
    weight = np.outer(output, output)
    coefficients = (weight*(modal+force)).ravel()
    coefficients.setflags(write=False)
    return prepared(coefficients, (f*d*amplitude)**2-np.sum(weight*force))


def prepare_tail(synthesis_sos, analysis_sos, prefix, amplitude, ratio):
    """Prepare once, integrate many grids; accepts float32 SOS coefficients."""
    synthesis_sos = np.ascontiguousarray(synthesis_sos, dtype=np.float64)
    analysis_sos = np.ascontiguousarray(analysis_sos, dtype=np.float64)
    prefix = np.asarray(prefix, float)
    for sos in (synthesis_sos, analysis_sos):
        if (sos.ndim != 2 or sos.shape[1] != 6 or len(sos) == 0
                or not np.isfinite(sos).all() or np.any(sos[:, 3] != 1)):
            raise ValueError('finite normalized SOS coefficients required')
    if (prefix.ndim != 1 or not np.isfinite(prefix).all() or not 0 < ratio < 1
            or not np.isfinite(amplitude) or amplitude < 0):
        raise ValueError('finite envelope and decaying exponential required')
    return _prepare_tail(synthesis_sos.tobytes(), analysis_sos.tobytes(), prefix.tobytes(),
                         float(amplitude), float(ratio))


def exponential_tail_energy(synthesis_sos, analysis_sos, prefix, amplitude, ratio, edges):
    """Exact discrete tail expectation; convenience wrapper for prepared state."""
    return prepare_tail(synthesis_sos, analysis_sos, prefix, amplitude, ratio).integrate_edges(edges)


def energy_cache_info():
    return {name: cache.cache_info() for name, cache in
            (('modes', _modes), ('rate_basis', _rate_basis),
             ('prepared', _prepare_tail), ('geometry', _edge_geometry))}


def clear_energy_caches():
    for cache in (_modes, _rate_basis, _prepare_tail, _edge_geometry):
        cache.cache_clear()
