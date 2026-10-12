"""3D sound-path page: it draws the renderer's own paths (same delays, bounces on the surfaces they
reflect from, unfolded lengths equal to image distances, obstacles applied), overlays a measured
response on the same time axis, and the page loads nothing from anywhere."""
from collections import Counter
import re

import numpy as np
import pytest

from sensorsim.acoustics.room import visualize as view
from sensorsim.acoustics.room.model import SURFACES, BoxObstacle, RoomModel
from sensorsim.acoustics.room.synth import SynthConfig, early_paths, image_paths, synthesize

ROOM = RoomModel(shoebox_dims_m=(6, 5, 3), late_t60_s=(.6,)*7)
SOURCE, MIC = (1.2, 1.1, 1.25), (4.3, 3.2, 1.4)


@pytest.fixture(scope="module")
def data():
    return view.scene(ROOM, SOURCE, MIC, max_order=3)


def test_drawn_paths_are_the_renderers_paths(data):
    rendered = early_paths(ROOM, np.array(SOURCE), np.array(MIC), SynthConfig(max_order=3, late=False))[0]
    np.testing.assert_allclose([p["delay_s"] for p in data["paths"]], np.sort(rendered.delay_s), rtol=0, atol=1e-12)
    assert Counter(p["order"] for p in data["paths"]) == {0: 1, 1: 6, 2: 18, 3: 38}


def test_every_bounce_lies_on_the_surface_it_reflects_from(data):
    dims = np.array(ROOM.shoebox_dims_m)
    images = image_paths(ROOM, SOURCE, MIC, max_order=3)
    expected = sorted((round(float(d), 9), tuple(int(n) for n in hits.reshape(3, 2).sum(axis=1)))
                      for d, hits in zip(images.delay_s, images.wall_hits))
    found = []
    for p in data["paths"]:
        pts = np.array(p["points"])
        assert np.allclose(pts[0], SOURCE) and np.allclose(pts[-1], MIC)
        for point in pts[1:-1]:
            on = [(abs(point[a]) < 1e-6) or (abs(point[a] - dims[a]) < 1e-6) for a in range(3)]
            assert any(on), f"bounce {point} is not on a room surface"
        per_axis = Counter(s[0] for s in p["surfaces"])
        found.append((round(p["delay_s"], 9), tuple(per_axis.get(a, 0) for a in "xyz")))
        assert len(p["surfaces"]) == p["order"] and set(p["surfaces"]) <= set(SURFACES)
    assert sorted(found) == expected


def test_unfolded_path_lengths_equal_the_image_distances(data):
    for p in data["paths"]:
        pts = np.array(p["points"])
        assert np.isclose(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum(), p["length_m"], rtol=0, atol=1e-6)
        assert np.isclose(p["length_m"]/ROOM.sound_speed_m_s, p["delay_s"], rtol=1e-12)


def test_an_obstacle_on_the_direct_line_is_applied_as_the_renderer_does():
    wall = BoxObstacle(name="cabinet", min_xyz=(2.5, 1.6, 0.), size_xyz=(.4, .8, 2.))   # between source and mic
    room = ROOM.model_copy(update={"obstacles": (wall,)})
    data = view.scene(room, SOURCE, MIC, max_order=1)
    rendered = early_paths(room, np.array(SOURCE), np.array(MIC), SynthConfig(max_order=1, late=False))[0]
    np.testing.assert_allclose(sorted(p["delay_s"] for p in data["paths"]), np.sort(rendered.delay_s), atol=1e-12)
    direct = next(p for p in data["paths"] if p["kind"] == "room" and p["order"] == 0)
    assert direct["status"] in ("detoured", "dropped")
    if direct["status"] == "detoured":
        assert direct["length_m"] > np.linalg.norm(np.subtract(SOURCE, MIC)) and direct["level_db"] < 0
    assert data["room"]["obstacles"][0]["name"] == "cabinet"


def test_a_measured_response_is_overlaid_on_the_same_time_axis():
    fs = 48000
    rir = synthesize(ROOM, SOURCE, MIC, fs, seed=3)
    samples = rir.samples[rir.metadata["latency_samples"]:]          # sample 0 = emission, as out/fit/rirs
    direct = np.linalg.norm(np.subtract(SOURCE, MIC))/ROOM.sound_speed_m_s
    env = view.measured_envelope(samples, fs, direct, view_s=.05, label="synthetic")
    assert abs(env["peak_s"] - direct) < 2.5e-4
    k = int(round(direct/env["step_s"]))
    assert max(env["level_db"][k - 1:k + 2]) > -1. and min(env["level_db"]) >= view.FLOOR_DB - 10


def test_the_page_is_self_contained(data):
    page = view.render_html(data, title="Test room paths")
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    assert not re.search(r"https?:|//[a-z]|<link|@import|url\(|\bsrc\s*=", page, re.I)
    assert len(re.findall(r"<script", page)) == 2                    # the data and the renderer, both inline
    fragment = view.render_html(data, title="Test room paths", standalone=False)
    assert "<title>Test room paths</title>" in fragment[:8192] and "<html" not in fragment and "<body" not in fragment


def test_text_from_the_project_cannot_break_out_of_the_page(data):
    hostile = {**data, "source": {**data["source"], "name": "</script><img onerror=alert(1)>"}}
    page = view.render_html(hostile, title="<b>x</b>")
    assert "</script><img" not in page and "<title>&lt;b&gt;x&lt;/b&gt;</title>" in page


def test_orders_above_three_are_refused():
    with pytest.raises(ValueError, match="third-order"):
        view.scene(ROOM, SOURCE, MIC, max_order=4)
