"""Strict PCM clipping must detect the signed integer minimum."""
import numpy as np
import pytest

from sensorsim.acoustics.audio_boundary import encode_pcm16
from sensorsim.core.audio_units import AudioCalibration

CAL = AudioCalibration(2., "hardware", "known 2 Pa peak full scale")


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_strict_clipping_rejects_the_signed_integer_minimum(dtype):
    for peak in (np.iinfo(dtype).min, -2):                # integer minimum; exact negative full scale
        with pytest.raises(ValueError, match="clip"):
            encode_pcm16(np.array([peak], dtype=dtype), CAL, clipping="raise")
    in_range = np.array([-1, 0, 1], dtype=dtype)          # |p| < 2 Pa: encoded exactly like floats
    np.testing.assert_array_equal(encode_pcm16(in_range, CAL, clipping="raise"),
                                  encode_pcm16(in_range.astype(float), CAL, clipping="raise"))
