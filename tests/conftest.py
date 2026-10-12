"""Suite-wide settings for the room-chain tests.

Numerical goldens in these tests were frozen with BLAS/OpenMP pools capped at
two threads; some reductions are thread-order sensitive at the 1e-17 level. Cap
the pools the same way so results are reproducible on any machine. torch is not
required (the parent project's conftest imported it; this package does not).
"""
import os
from pathlib import Path
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[1]
# Exercise this checkout even when another worktree owns the editable install.
sys.path.insert(0, str(_ROOT / "src"))

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)
for _variable in _THREAD_VARIABLES:
    os.environ[_variable] = "2"

# Goldens assume the shipped defaults: plug-in data installed through the environment
# (THIRD_PARTY.md, "Plug-in parts") would change priors and absorption, so tests ignore it.
for _variable in ("SENSORSIM_ACOUSTIC_CATALOGUE", "SENSORSIM_SCENE_ABSORPTION"):
    os.environ.pop(_variable, None)

try:  # optional: also cap pools already loaded before the environment took effect
    from threadpoolctl import threadpool_limits
except ImportError:
    pass
else:
    threadpool_limits(2)


@pytest.fixture
def python_subprocess_env():
    """Child interpreters need explicit paths; pytest's pythonpath is local."""
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(_ROOT / "src"), str(_ROOT))),
        **dict.fromkeys(_THREAD_VARIABLES, "2"),
    }


@pytest.fixture(scope="session", autouse=True)
def generated_room_session_cache():
    """Reuse procedural input bytes; each caller still fits/trains independently."""
    from tests.room_session_cache import session_cache

    with session_cache():
        yield


# Explicit groups, measured on the serial suite. The room-pipeline pair shares
# expensive setup, so either test on its own can incur that cost. Keep this list
# independent of the current machine's speed and never auto-skip slow tests.
_SLOW_TESTS = {
    "tests/unit/licences/test_matcher_fields.py::test_review_benign_metadata_trains",
    "tests/unit/project/test_project.py::test_an_edited_install_reruns_every_stage",
    "tests/unit/project/test_project.py::test_installing_an_absorption_catalogue_outside_the_package_reruns_the_fit",
    "tests/unit/room_pipeline/test_pipeline.py::test_fast_fake_user_session",
    "tests/unit/room_pipeline/test_pipeline.py::test_pipeline_outputs_are_byte_identical",
}

# These were already slow. Distinguish their frozen campaigns from ordinary
# slow regression tests so profiling can select `slow and not acceptance`.
_ACCEPTANCE_TESTS = {
    "tests/unit/evaluation/test_acceptance_calibrated.py::test_registered_calibrated_proxy_acceptance",
    "tests/unit/evaluation/test_acceptance_final.py::test_registered_e2_refit_permutation_calibration",
    "tests/unit/evaluation/test_acceptance_final.py::test_registered_final_truth_framing_control",
    "tests/unit/evaluation/test_acceptance_truth.py::test_registered_wp1_acceptance_on_truth_proxy",
    "tests/unit/evaluation/test_fa_rate_acceptance.py::test_wp2_fresh_six_hour_proxy_acceptance",
}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        test = item.nodeid.split("[", 1)[0]
        if test in _SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
        if test in _ACCEPTANCE_TESTS:
            item.add_marker(pytest.mark.acceptance)
