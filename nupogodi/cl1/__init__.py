"""CL1-shaped neural interface: one contract, swappable backends.

The agent talks to a :class:`~nupogodi.cl1.api.NeuronsLike` object and never
imports a backend directly. :func:`open` hands it either the responsive local
learning culture (``backend="sim"``) or a real Cortical Labs ``cl`` connection
(``backend="hardware"``), mirroring ``cl.open()`` ergonomics.
"""

from __future__ import annotations

from .api import (
    MAX_RATE_HZ,
    PULSE_QUANTUM_US,
    BurstDesign,
    ChannelSet,
    NeuronsLike,
    Spike,
    StimDesign,
    Tick,
)

__all__ = [
    "BurstDesign",
    "ChannelSet",
    "MAX_RATE_HZ",
    "NeuronsLike",
    "PULSE_QUANTUM_US",
    "Spike",
    "StimDesign",
    "Tick",
    "open",
]


def open(backend: str = "sim", **cfg: object) -> NeuronsLike:  # noqa: A001 — mirrors cl.open()
    """Open a neural interface. ``backend="sim"`` is the local learning dish
    (needs the ``snn`` extra); ``backend="hardware"`` wraps the real ``cl`` SDK
    (needs the ``cl`` extra / a CL1)."""
    if backend == "sim":
        from .dish import SpikingDish

        return SpikingDish(**cfg)  # type: ignore[arg-type]
    if backend == "hardware":
        from .hardware import RealDish

        return RealDish(**cfg)
    raise ValueError(f"unknown backend {backend!r}; use 'sim' or 'hardware'")
