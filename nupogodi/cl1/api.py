"""The CL-shaped contract — the slice of the Cortical Labs ``cl`` API we use.

We deliberately speak the *real* CL1 vocabulary rather than inventing a bespoke
"neural interface". The names here mirror ``cl-sdk`` (``docs.corticallabs.com``):
:class:`ChannelSet`, :class:`StimDesign`, :class:`BurstDesign`, and a
``Neurons``-like object you drive with ``loop()`` / ``stim()``, reading spikes
off ``tick.analysis.spikes``. Because the agent is written against this contract
(the :class:`NeuronsLike` Protocol), the *same* agent code drives either the
responsive local :class:`~nupogodi.cl1.dish.SpikingDish` or the real ``import
cl`` hardware/simulator backend — the whole point of the exercise: rehearse the
integration on a substrate that learns, then swap in wetware unchanged.

The value classes also enforce the hardware's real constraints (pulse widths in
multiples of 20 µs, ≤200 Hz per channel), so code that passes here won't be
rejected by a CL1.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

#: Stim pulse widths are quantised to multiples of this many microseconds.
PULSE_QUANTUM_US = 20
#: Maximum stimulation rate a single channel accepts.
MAX_RATE_HZ = 200


class ChannelSet:
    """A set of target electrodes, e.g. ``ChannelSet(8, 9, 10)`` or
    ``ChannelSet([8, 9, 10])`` — mirrors ``cl.ChannelSet``."""

    __slots__ = ("channels",)

    def __init__(self, *channels: int | Iterable[int]) -> None:
        if len(channels) == 1 and isinstance(channels[0], Iterable):
            channels = tuple(channels[0])  # type: ignore[assignment]
        chans = tuple(int(c) for c in channels)  # type: ignore[arg-type]
        if any(c < 0 for c in chans):
            raise ValueError("channel indices must be non-negative")
        self.channels: tuple[int, ...] = chans

    def __iter__(self) -> Iterator[int]:
        return iter(self.channels)

    def __len__(self) -> int:
        return len(self.channels)

    def __repr__(self) -> str:
        return f"ChannelSet{self.channels}"


@dataclass(frozen=True)
class StimDesign:
    """A stimulation pulse: ``pulse_us`` wide (multiple of 20 µs) at
    ``current_ua`` microamps. Negative current = negative leading edge, which
    CL1 recommends for charge balancing. Mirrors ``cl.StimDesign``."""

    pulse_us: int
    current_ua: float

    def __post_init__(self) -> None:
        if self.pulse_us <= 0 or self.pulse_us % PULSE_QUANTUM_US:
            raise ValueError(
                f"pulse_us must be a positive multiple of {PULSE_QUANTUM_US} µs"
            )


@dataclass(frozen=True)
class BurstDesign:
    """``count`` pulses delivered at ``rate_hz`` — mirrors ``cl.BurstDesign``.
    Rate is capped at :data:`MAX_RATE_HZ`, the per-channel hardware limit."""

    count: int
    rate_hz: float

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("burst count must be positive")
        if not 0 < self.rate_hz <= MAX_RATE_HZ:
            raise ValueError(f"rate_hz must be in (0, {MAX_RATE_HZ}] Hz")


@dataclass(frozen=True)
class Spike:
    """A detected spike on an electrode — mirrors ``tick.analysis.spikes`` items
    (a real ``Spike`` also carries a ``samples`` waveform we don't need here)."""

    channel: int
    timestamp: int


@dataclass
class TickAnalysis:
    """The ``tick.analysis`` payload — spikes detected during this tick."""

    spikes: list[Spike] = field(default_factory=list)


@dataclass
class Tick:
    """One iteration of ``neurons.loop()`` — mirrors the real ``tick``."""

    index: int
    analysis: TickAnalysis


@runtime_checkable
class NeuronsLike(Protocol):
    """The subset of the CL ``Neurons`` interface the agent depends on.

    A backend is anything satisfying this: the local :class:`SpikingDish` (which
    learns) or a wrapper over the real ``cl`` ``Neurons`` (hardware / official
    simulator). The agent imports neither backend directly.
    """

    def loop(
        self, *, ticks_per_second: int, stop_after_ticks: int | None = None
    ) -> Iterator[Tick]:
        """Yield ticks in real (or simulated) time; read ``tick.analysis.spikes``."""
        ...

    def stim(
        self,
        channel_set: ChannelSet,
        stim_design: StimDesign,
        burst: BurstDesign | None = None,
    ) -> None:
        """Stimulate ``channel_set``; a single pulse, or a burst if given."""
        ...

    def deliver_feedback(self, signal: float) -> None:
        """Deliver a neuromodulatory feedback signal (the reward / dopamine).

        On the local dish this modulates STDP (three-factor learning). On real
        wetware this is where a DishBrain-style predictable-vs-unpredictable
        stimulation scheme would live.
        """
        ...

    def close(self) -> None: ...
