"""Reference geometry and small shared physics helpers."""
from __future__ import annotations

import numpy as np

SOUND_SPEED_MODEL = "dry-air-core/1"


def speed_of_sound(temperature_c: float | np.ndarray) -> float | np.ndarray:
    """dry-air-core/1: 331.3 sqrt(1 + T_C/273.15) m/s (D02).

    Runtime temperature-only approximation, excluding humidity/CO2. Measured and
    analytic reference fixtures may explicitly pin a different fixed sound speed.
    """
    return 331.3 * np.sqrt(1.0 + temperature_c / 273.15)


HUMID_SOUND_SPEED_MODEL = "humid-air-cramer/1"
CRAMER_RANGE = {"temperature_c": (0.0, 30.0), "pressure_kpa": (75.0, 102.0)}
# Cramer (1993), J. Acoust. Soc. Am. 93(5), 2510-2516, eq. (15): zero-frequency speed of sound
# in real air as a function of temperature t (C), pressure p (Pa), water-vapour and CO2 mole
# fractions x_w and x_c.
_CRAMER_A = (331.5024, 0.603055, -0.000528, 51.471935, 0.1495874, -0.000782, -1.82e-7, 3.73e-8,
             -2.93e-10, -85.20931, -0.228525, 5.91e-5, -2.835149, -2.15e-13, 29.179762, 0.000486)


def water_vapour_mole_fraction(temperature_c, relative_humidity_pct, pressure_kpa=101.325):
    """x_w from relative humidity: Davis (1992) saturation vapour pressure and enhancement factor."""
    t = np.asarray(temperature_c, float)
    kelvin = t + 273.15
    p = np.asarray(pressure_kpa, float) * 1000.0
    p_sv = np.exp(1.2811805e-5 * kelvin**2 - 1.9509874e-2 * kelvin + 34.04926034 - 6.3536311e3 / kelvin)
    enhancement = 1.00062 + 3.14e-8 * p + 5.6e-7 * t**2
    return np.asarray(relative_humidity_pct, float) / 100.0 * enhancement * p_sv / p


def speed_of_sound_cramer(temperature_c, relative_humidity_pct, pressure_kpa=101.325, co2_ppm=420.0):
    """humid-air-cramer/1: speed of sound in real (humid, CO2-bearing) air, m/s.

    Cramer (1993) eq. (15), valid for 0-30 C and 75-102 kPa; outside that range it raises rather
    than extrapolating (use dry-air-core/1 or a fixed reference speed). At 20 C and 50 % RH it is
    about 0.77 m/s above the dry-air approximation.
    """
    t = np.asarray(temperature_c, float)
    pressure = np.asarray(pressure_kpa, float)
    (lo_t, hi_t), (lo_p, hi_p) = CRAMER_RANGE["temperature_c"], CRAMER_RANGE["pressure_kpa"]
    if (not np.all(np.isfinite(t)) or np.any((t < lo_t) | (t > hi_t)) or np.any((pressure < lo_p) | (pressure > hi_p))
            or np.any((np.asarray(relative_humidity_pct, float) < 0) | (np.asarray(relative_humidity_pct, float) > 100))):
        raise ValueError(f"{HUMID_SOUND_SPEED_MODEL} is defined for {lo_t:g}-{hi_t:g} C, {lo_p:g}-{hi_p:g} kPa and "
                         "0-100 % RH; outside it use dry-air-core/1 or a fixed reference sound speed")
    a = _CRAMER_A
    x_w = water_vapour_mole_fraction(t, relative_humidity_pct, pressure)
    x_c = np.asarray(co2_ppm, float) * 1e-6
    p = pressure * 1000.0
    c = (a[0] + a[1]*t + a[2]*t**2 + (a[3] + a[4]*t + a[5]*t**2)*x_w + (a[6] + a[7]*t + a[8]*t**2)*p
         + (a[9] + a[10]*t + a[11]*t**2)*x_c + a[12]*x_w**2 + a[13]*p**2 + a[14]*x_c**2 + a[15]*x_w*p*x_c)
    return float(c) if np.ndim(c) == 0 else c


def sound_speed(model: str, temperature_c, relative_humidity_pct=50.0, pressure_kpa=101.325):
    """The one place a sound-speed policy becomes a number: humid-air-cramer/1 or dry-air-core/1."""
    if model == HUMID_SOUND_SPEED_MODEL:
        return speed_of_sound_cramer(temperature_c, relative_humidity_pct, pressure_kpa)
    if model == SOUND_SPEED_MODEL:
        return float(speed_of_sound(temperature_c)) if np.ndim(temperature_c) == 0 else speed_of_sound(temperature_c)
    raise ValueError(f"unknown sound-speed model {model!r}")


