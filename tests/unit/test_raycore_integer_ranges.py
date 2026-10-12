import numpy as np
import pytest

from sensorsim.scene.raycore import TriangleMesh


@pytest.mark.parametrize("field,value", [
    ("triangles", np.array([[0, 1, 2**64 - 1]], dtype=np.uint64)),
    ("face_ids", np.array([2**64 - 1], dtype=np.uint64)),
    ("material_ids", np.array([2**64 - 1], dtype=np.uint64)),
    ("object_ids", np.array([2**64 - 1], dtype=np.uint64)),
])
def test_uint64_mesh_indices_and_ids_cannot_wrap_to_signed_sentinels(field, value):
    arrays = dict(
        vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        face_ids=np.array([0], dtype=np.int64),
        material_ids=np.array([0], dtype=np.int64),
        object_ids=np.array([0], dtype=np.int64),
    )
    arrays[field] = value

    with pytest.raises(ValueError):
        TriangleMesh(**arrays)


def _one_triangle_core():
    from sensorsim.scene.raycore import RayCore
    mesh = TriangleMesh(np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]), [[0, 1, 2]], [7], [0], [0], ("m",))
    return RayCore(mesh=mesh, backend="cpu")


@pytest.mark.parametrize("value", [np.uint64(2**64 - 1), 2**64 - 1, np.array([2**64 - 1], dtype=np.uint64)])
@pytest.mark.parametrize("field", ["ignore_box", "start_face"])
def test_uint64_exclusion_ids_cannot_wrap_to_the_none_sentinel(field, value):
    # Same root cause as above: exclusion IDs were narrowed into int64 before validation, so 2**64-1 became
    # the -1 "no exclusion" sentinel and the exclusion was silently dropped (lead review of z-prune).
    a, b = [[.25, .25, -1.]], [[.25, .25, 1.]]
    with _one_triangle_core() as rc, pytest.raises(ValueError):
        if field == "ignore_box":
            rc.any_hit(a, b, ignore_box=value)
        else:
            rc.segment_crossings(a, b, start_face=value)


def test_small_unsigned_exclusion_ids_still_work():
    a, b = [[.25, .25, -1.]], [[.25, .25, 1.]]
    with _one_triangle_core() as rc:
        assert len(rc.segment_crossings(a, b)) == 1
        assert len(rc.segment_crossings(a, b, start_face=np.uint64(7))) == len(rc.segment_crossings(a, b, start_face=7)) == 0
