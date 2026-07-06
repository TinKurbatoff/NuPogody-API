"""The CL-shaped contract enforces the real hardware constraints."""

from __future__ import annotations

import pytest

from nupogodi.cl1 import MAX_RATE_HZ, BurstDesign, ChannelSet, StimDesign


def test_channelset_accepts_varargs_and_iterable():
    assert tuple(ChannelSet(8, 9, 10)) == (8, 9, 10)
    assert tuple(ChannelSet([8, 9, 10])) == (8, 9, 10)
    assert len(ChannelSet(range(4))) == 4


def test_stimdesign_pulse_must_be_multiple_of_20us():
    StimDesign(160, -1.0)  # ok
    with pytest.raises(ValueError):
        StimDesign(150, -1.0)  # not a multiple of 20
    with pytest.raises(ValueError):
        StimDesign(0, -1.0)


def test_burstdesign_rate_capped_at_hardware_limit():
    BurstDesign(10, MAX_RATE_HZ)  # ok at the cap
    with pytest.raises(ValueError):
        BurstDesign(10, MAX_RATE_HZ + 1)  # over 200 Hz/channel
    with pytest.raises(ValueError):
        BurstDesign(0, 100)  # non-positive count


def test_realdish_absence_is_graceful():
    """Without cl-sdk installed, the hardware backend reports unavailability
    rather than exploding at import time."""
    from nupogodi.cl1 import hardware

    if hardware.HAVE_CL:  # pragma: no cover — only where a device/SDK is present.
        pytest.skip("cl-sdk is installed; nothing to assert about its absence")
    with pytest.raises(RuntimeError, match="cl-sdk is not installed"):
        hardware.RealDish()
