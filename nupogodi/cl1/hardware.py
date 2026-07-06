"""``RealDish`` — a thin adapter over the real Cortical Labs ``cl`` SDK.

Wraps ``import cl`` / ``cl.open()`` to the same
:class:`~nupogodi.cl1.api.NeuronsLike` surface the agent already speaks, so the
identical agent drives real CL1 hardware or the official (non-learning)
simulator. Optional: absent ``cl-sdk`` it stays unavailable rather than erroring
at import.

Note the honest asymmetry captured by :meth:`deliver_feedback`: on the local
:class:`~nupogodi.cl1.dish.SpikingDish` feedback is a δ that modulates STDP; on
wetware there is no reward wire — learning is shaped by *how* you stimulate.
That DishBrain predictable-vs-unpredictable feedback scheme is the seam here,
deliberately left for the hardware-in-the-loop phase.
"""

from __future__ import annotations

from collections.abc import Iterator

from .api import BurstDesign, ChannelSet, Spike, StimDesign, Tick, TickAnalysis

try:  # the SDK is only present on a CL1 device or when cl-sdk is installed.
    import cl  # type: ignore

    HAVE_CL = True
except ImportError:  # pragma: no cover — exercised only where cl-sdk is absent.
    cl = None  # type: ignore
    HAVE_CL = False


class RealDish:
    """Adapts a live ``cl`` ``Neurons`` object to :class:`NeuronsLike`."""

    def __init__(self, **open_kwargs: object) -> None:
        if not HAVE_CL:
            raise RuntimeError(
                "cl-sdk is not installed; install it with `pip install -e '.[cl]'` "
                "to target CL1 hardware or the official simulator."
            )
        self._ctx = cl.open(**open_kwargs)  # type: ignore[union-attr]
        self._neurons = self._ctx.__enter__()

    def stim(
        self,
        channel_set: ChannelSet,
        stim_design: StimDesign,
        burst: BurstDesign | None = None,
    ) -> None:
        chans = cl.ChannelSet(list(channel_set))  # type: ignore[union-attr]
        design = cl.StimDesign(stim_design.pulse_us, stim_design.current_ua)  # type: ignore[union-attr]
        if burst is None:
            self._neurons.stim(chans, design)
        else:
            cl_burst = cl.BurstDesign(burst.count, burst.rate_hz)  # type: ignore[union-attr]
            self._neurons.stim(chans, design, cl_burst)

    def loop(
        self, *, ticks_per_second: int = 1000, stop_after_ticks: int | None = None
    ) -> Iterator[Tick]:
        kwargs: dict[str, object] = {"ticks_per_second": ticks_per_second}
        if stop_after_ticks is not None:
            kwargs["stop_after_ticks"] = stop_after_ticks
        for i, tick in enumerate(self._neurons.loop(**kwargs)):
            analysis = TickAnalysis(
                spikes=[Spike(channel=s.channel, timestamp=s.timestamp)
                        for s in tick.analysis.spikes]
            )
            yield Tick(index=i, analysis=analysis)

    def deliver_feedback(self, signal: float) -> None:
        # Seam: on real wetware, feedback is delivered as structured stimulation
        # (DishBrain free-energy scheme), not a reward scalar. Not built for v1.
        raise NotImplementedError(
            "hardware feedback (predictability-based stimulation) is a v2 seam; "
            "learning locally uses the SpikingDish backend."
        )

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)
