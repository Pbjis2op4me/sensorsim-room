"""Domestic rendering assumptions, separate from calibration search support.

These are project engineering choices, not copied material data or measured home
population bounds. The 500–1000 Hz bootstrap is an engineering assumption; no validated household distribution is claimed.
No archive-specific fitted parameters enter this policy.
"""

DOMESTIC_POLICY = "domestic-room-assumption/1"
# Octaves: 125, 250, 500, 1000, 2000, 4000, 8000 Hz.
DOMESTIC_T30_S = ((.35, 1.2), (.30, 1.), (.25, .9), (.25, .9),
                  (.20, .75), (.15, .60), (.10, .45))
DOMESTIC_RATIONALES = (
    "Assumption: longer low-frequency decay allowed; modal behavior is not resolved.",
    "Assumption: transition from longer low-frequency decay to mid-band support.",
    "Assumption: 0.25--0.9 s bootstrap from programme WP4; not a home population estimate.",
    "Assumption: same mid-band bootstrap; exclude near-anechoic uncalibrated training draws.",
    "Assumption: lower support for increasing air/furnishing loss; no measured band slope.",
    "Assumption: further high-frequency loss; no measured household distribution.",
    "Assumption: strongest high-frequency loss; not calibrated to the user's room.",
)
ASSUMED_ROOM_WARNING = (
    "WARNING: ROOM IS ASSUMED, NOT MEASURED. No usable calibration: rendering uses "
    "domestic-room-assumption/1, not measured in-distribution room statistics. "
    "Add 30 s of broadband noise playback as played/noise.wav and its room recording "
    "as room_recordings/noise__room.wav, then rerun fit and dataset. "
    "Keep recording pre-roll and the full decay tail. Noise or repeated broadband "
    "sweep packets are preferred to a single sweep for clock identification."
)

CALIBRATION_ONLY_PREFIXES = ("noise_", "noise-")


def calibration_only(group: str) -> bool:
    """Broadband noise playback (played/noise*.wav) calibrates the room; it is never held out automatically."""
    return group == "noise" or group.startswith(CALIBRATION_ONLY_PREFIXES)


def assumed_room_warning(*, paired=(), held_out=(), rejected=None) -> str:
    """The assumed-room warning, naming why calibration did not happen."""
    paired, held_out, rejected = sorted(paired), sorted(set(held_out) & set(paired)), dict(rejected or {})
    if not paired:
        return ASSUMED_ROOM_WARNING
    head = "WARNING: ROOM IS ASSUMED, NOT MEASURED. "
    tail = " Rendering uses domestic-room-assumption/1, not measured in-distribution room statistics."
    if len(held_out) == len(paired):
        return (head + f"Every paired playback ({', '.join(held_out)}) was held out for validation, so none "
                "calibrated the room." + tail + " List the groups to hold out under holdout_groups in "
                "calibration.yaml, keeping the noise playback for calibration, then rerun fit and dataset.")
    reasons = "; ".join(f"{group}: {'; '.join(why) or 'not qualified'}" for group, why in sorted(rejected.items()))
    return (head + f"No paired playback calibrated the room ({reasons or 'no accepted observation'})." + tail +
            " Check played/noise.wav and room_recordings/noise__room.wav (pre-roll, full decay tail, no "
            "clipping), then rerun fit and dataset.")


def conflict_warning(ensemble):
    """An arrival mismatch needs a timing review, not a wider decay prior."""
    keys = {k for e in ensemble.evidence for k in e.get("conflicts", {})}
    actions = []
    if "direct_delay_s" in keys:
        actions.append("check the hardware anchor, DAC/ADC/buffer latency, clock and surveyed geometry; "
                       "widening the decay prior cannot repair a timing mismatch")
    if keys - {"direct_delay_s"} or not keys:
        actions.append("review excitation, decay support and calibration prior against independent evidence")
    return "prior-conflict: calibration refused; " + "; ".join(actions) + ". Rerun fit. Rendering blocked."


def select_render_ensemble(ensemble):
    """Use the posterior unchanged, or the explicit domestic assumption.

    Converting here also protects callers holding an older prior-only JSON or
    passing build_prior() directly to dataset generation. Conflicts never fall
    back to assumed rendering.
    """
    from .prior import build_prior
    if ensemble.status == "prior-conflict":
        raise ValueError(conflict_warning(ensemble))
    if ensemble.status != "assumed" or ensemble.purpose == "domestic-render":
        return ensemble
    return build_prior(ensemble.survey, purpose="domestic-render").model_copy(
        update={"evidence": ensemble.evidence})
