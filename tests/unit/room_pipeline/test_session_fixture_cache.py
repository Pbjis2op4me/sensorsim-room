"""Generated-input fixture controls; these are synthetic proxies, not room acceptance."""
import numpy as np

from tests.unit.room_pipeline import test_pipeline as fixture


def _contents(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def test_repeated_session_reuses_generation_without_sharing_outputs(tmp_path, monkeypatch):
    original = fixture.synthesize
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs['seed'])
        return original(*args, **kwargs)

    monkeypatch.setattr(fixture, 'synthesize', observed)
    first = fixture.fake_session(tmp_path/'first')
    expected = _contents(first[0])
    model_json = first[1].model_dump_json()
    truth = {key: value.copy() for key, value in first[2].items()}
    # The caller owns every returned file and object, including the cache miss.
    (first[0]/'M1_sweep.wav').write_bytes(b'caller mutation')
    first[1].provenance['caller mutation'] = True
    first[2]['M1'][:] = 0
    first[2]['extra'] = np.zeros(1)
    second = fixture.fake_session(tmp_path/'second')
    assert _contents(second[0]) == expected
    assert second[1].model_dump_json() == model_json
    assert second[2].keys() == truth.keys()
    for key, values in truth.items():
        assert second[2][key].dtype == values.dtype
        assert second[2][key].tobytes() == values.tobytes()
    second[1].provenance['cache hit mutation'] = True
    second[2]['M1'][:] = 0
    third = fixture.fake_session(tmp_path/'third')
    assert third[1].model_dump_json() == model_json
    assert third[2]['M1'].tobytes() == truth['M1'].tobytes()
    assert calls == [11, 12, 13, 14]


def test_cache_variants_and_generation_hooks_do_not_collide(tmp_path):
    from tests.room_session_cache import generated_session, session_cache
    calls = []
    hook = [b'first']

    def build(folder, *, full):
        calls.append((full, hook[0]))
        folder.mkdir()
        (folder/'audio').write_bytes(hook[0] + bytes([full]))
        return folder, {}, {}, 64

    with session_cache():
        for index, (full, dependency) in enumerate(((False, b'first'), (True, b'first'),
                                                   (True, b'second'), (True, b'second'))):
            hook[0] = dependency
            folder = tmp_path/str(index)
            generated_session(folder, full=full, build=build, dependencies=(dependency,))
            assert (folder/'audio').read_bytes() == dependency + bytes([full])
    assert calls == [(False, b'first'), (True, b'first'), (True, b'second')]


def test_cache_preserves_empty_constant_duplicate_data_and_destination_errors(tmp_path):
    import pytest
    from tests.room_session_cache import generated_session, session_cache

    def build(folder, *, full):
        folder.mkdir()
        for name, data in [('empty', b''), ('constant', b'\0'*8), ('duplicate', b'\0'*8)]:
            (folder/name).write_bytes(data)
        return folder, {'nested': []}, {'empty': np.array([]), 'constant': np.zeros(4)}, 64

    with session_cache():
        first = generated_session(tmp_path/'first', full=False, build=build, dependencies=())
        second = generated_session(tmp_path/'second', full=False, build=build, dependencies=())
        assert _contents(first[0]) == _contents(second[0])
        assert set(second[2]) == {'empty', 'constant'} and second[2]['empty'].shape == (0,)
        first[1]['nested'].append('mutation')
        first[2]['constant'][:] = 1
        assert second[1] == {'nested': []}
        assert second[2]['constant'].tobytes() == np.zeros(4).tobytes()
        with pytest.raises(FileExistsError):
            generated_session(first[0], full=False, build=build, dependencies=())
        with pytest.raises(FileNotFoundError):
            generated_session(tmp_path/'absent'/'child', full=False, build=build, dependencies=())


def test_cache_lifetime_failure_and_eviction(tmp_path, monkeypatch):
    import pytest
    from tests.room_session_cache import generated_session, session_cache
    calls = []

    def build(folder, *, full):
        calls.append(full)
        folder.mkdir()
        (folder/'input').write_bytes(b'input')
        return folder, {}, {}, 64

    def call(index, dependency=0):
        return generated_session(tmp_path/str(index), full=False, build=build, dependencies=(dependency,))

    with session_cache():
        for index, dependency in enumerate([0, 1, 2, 0, 0]):
            call(index, dependency)
    assert len(calls) == 4  # two-entry bound, followed by a genuine hit
    with session_cache():
        call(5)
    assert len(calls) == 5  # no values survive a new session
    # Direct use outside pytest still calls the original builder every time.
    from tests import room_session_cache
    with monkeypatch.context() as patch:
        patch.setattr(room_session_cache, '_active', None)
        call(6)
        call(7)
    assert len(calls) == 7
    def broken(folder, *, full):
        raise RuntimeError('fixture generation failed')
    with session_cache():
        for _ in range(2):
            with pytest.raises(RuntimeError, match='fixture generation failed'):
                generated_session(tmp_path/'broken', full=False, build=broken, dependencies=())
