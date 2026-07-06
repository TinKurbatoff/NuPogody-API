"""``SpikingDish`` — the responsive, *learning* local backend.

A small biological-ish neural culture emulated with BindsNET, exposed through
the CL-shaped :class:`~nupogodi.cl1.api.NeuronsLike` contract so the agent can't
tell it from wetware. Unlike the official CL simulator (which emits Poisson
noise that ignores stimulation), this one actually *responds* to stimulation and
*learns*, so you can watch a policy improve locally before ever touching a CL1.

Anatomy (fixed topology — you get a culture, you don't design its wiring):

* an **input electrode** layer that stimulation drives (rate-coded),
* a recurrent **culture** of LIF neurons with fixed random sparse internal
  wiring and a constant background drive, so it fires *spontaneously* — real
  cultures are never silent, and that intrinsic noise is what gives the agent
  its exploration (no ε-greedy anywhere),
* a small **motor** layer whose spikes are read out as the action.

The two afferent pathways (input→culture, culture→motor) are **plastic** under
BindsNET's :class:`MSTDPET` — modulated STDP with eligibility traces, i.e. exact
three-factor reward-modulated STDP. Eligibility traces accumulate over a
decision window at zero reward; :meth:`deliver_feedback` then broadcasts the
reward-prediction-error δ (the dopamine signal) which converts that standing
eligibility into a weight change. No backprop touches this network.
"""

from __future__ import annotations

from collections.abc import Iterator

from . import _compat

# The torch._six shim MUST be installed before bindsnet is imported. This call
# is a hard ordering barrier isort will not reorder imports across — keep it.
_compat.install()

import torch  # noqa: E402  — deliberately after the compat shim above.
from bindsnet.learning import MSTDPET  # noqa: E402
from bindsnet.network import Network  # noqa: E402
from bindsnet.network.nodes import Input, LIFNodes  # noqa: E402
from bindsnet.network.topology import Connection  # noqa: E402

from .api import (  # noqa: E402
    MAX_RATE_HZ,
    BurstDesign,
    ChannelSet,
    Spike,
    StimDesign,
    Tick,
    TickAnalysis,
)

DEFAULT_IN_CHANNELS = tuple(range(7))  # one electrode per env observation feature.
DEFAULT_MOTOR_CHANNELS = (100, 101, 102, 103)  # four motor electrodes = four actions.


class SpikingDish:
    """A learning spiking culture speaking the CL ``NeuronsLike`` contract."""

    def __init__(
        self,
        *,
        neurons: int = 100,
        in_channels: tuple[int, ...] = DEFAULT_IN_CHANNELS,
        motor_channels: tuple[int, ...] = DEFAULT_MOTOR_CHANNELS,
        dt_ms: float = 1.0,
        thresh: float = -58.0,
        nu: float = 0.02,
        tc_e_trace: float = 25.0,
        w_in_init: float = 1.0,
        w_in_max: float = 2.5,
        w_out_init: float = 0.35,
        w_out_max: float = 1.2,
        out_density: float = 0.4,
        recurrent_density: float = 0.1,
        recurrent_scale: float = 0.8,
        noise_rate_hz: float = 20.0,
        noise_scale: float = 3.0,
        feedback_ticks: int = 2,
        seed: int | None = None,
    ) -> None:
        # These magnitudes are calibrated so the LIF layers sit in a *graded*,
        # pattern-sensitive regime (culture rate ~0.05–0.13, motor sub-saturation)
        # rather than the silent or fully-saturated extremes — that band is what
        # lets the readout express, and learn, four distinct actions.
        self.dt_ms = dt_ms
        self.in_channels = tuple(in_channels)
        self.motor_channels = tuple(motor_channels)
        self.feedback_ticks = feedback_ticks
        self._gen = torch.Generator().manual_seed(seed if seed is not None else 0)

        n_in = len(self.in_channels)
        n_motor = len(self.motor_channels)
        self._in_index = {c: i for i, c in enumerate(self.in_channels)}
        self._motor_index = {c: i for i, c in enumerate(self.motor_channels)}

        # Per-channel firing probability for the pending stimulation window.
        self._in_rate = torch.zeros(n_in)
        self._noise_p = min(1.0, noise_rate_hz * dt_ms / 1000.0)
        self._tick = 0

        net = Network(dt=dt_ms)
        electrodes = Input(n=n_in, traces=True)
        culture = LIFNodes(n=neurons, traces=True, thresh=thresh)
        motor = LIFNodes(n=n_motor, traces=True, thresh=thresh)
        noise = Input(n=neurons, traces=False)
        net.add_layer(electrodes, name="in")
        net.add_layer(culture, name="culture")
        net.add_layer(motor, name="motor")
        net.add_layer(noise, name="noise")

        def _rand(*shape: int) -> torch.Tensor:
            return torch.rand(*shape, generator=self._gen)

        # Plastic afferents under MSTDPET (three-factor reward-modulated STDP).
        # Separate caps: the input→culture and culture→motor pathways live at
        # very different natural scales, so they can't share one wmax.
        c_in = Connection(
            source=electrodes, target=culture, update_rule=MSTDPET,
            nu=nu, wmin=0.0, wmax=w_in_max, tc_e_trace=tc_e_trace,
            w=_rand(n_in, neurons) * w_in_init,
        )
        out_mask = (_rand(neurons, n_motor) < out_density).float()
        c_out = Connection(
            source=culture, target=motor, update_rule=MSTDPET,
            nu=nu, wmin=0.0, wmax=w_out_max, tc_e_trace=tc_e_trace,
            w=_rand(neurons, n_motor) * w_out_init * out_mask,
        )
        # Fixed recurrent culture wiring: sparse random, non-plastic.
        rec_mask = (_rand(neurons, neurons) < recurrent_density).float()
        rec_mask.fill_diagonal_(0.0)
        c_rec = Connection(
            source=culture, target=culture,
            w=_rand(neurons, neurons) * recurrent_scale * rec_mask,
        )
        # Fixed background drive → spontaneous activity (culture is never silent).
        c_noise = Connection(
            source=noise, target=culture,
            w=torch.eye(neurons) * noise_scale,
        )
        net.add_connection(c_in, source="in", target="culture")
        net.add_connection(c_out, source="culture", target="motor")
        net.add_connection(c_rec, source="culture", target="culture")
        net.add_connection(c_noise, source="noise", target="culture")

        self.net = net
        self.culture = culture
        self.motor = motor
        self._c_in = c_in
        self._c_out = c_out
        self._n_in = n_in
        self._n_motor = n_motor
        self._neurons = neurons

    # -- CL NeuronsLike contract ------------------------------------------

    def stim(
        self,
        channel_set: ChannelSet,
        stim_design: StimDesign,
        burst: BurstDesign | None = None,
    ) -> None:
        """Schedule rate-coded stimulation on ``channel_set`` for the window.

        Rate coding: a burst's ``rate_hz`` sets the channel's per-tick firing
        probability (``rate·dt``); ``stim_design.current_ua`` scales its drive.
        Matches how you'd stimulate a CL1 electrode, within the ≤200 Hz cap.
        """
        rate_hz = burst.rate_hz if burst is not None else MAX_RATE_HZ
        p = min(1.0, rate_hz * self.dt_ms / 1000.0)
        drive = p * min(1.0, abs(stim_design.current_ua))
        for c in channel_set:
            idx = self._in_index.get(int(c))
            if idx is not None:
                self._in_rate[idx] = drive

    def loop(
        self, *, ticks_per_second: int = 1000, stop_after_ticks: int | None = None
    ) -> Iterator[Tick]:
        """Advance the culture tick-by-tick, yielding detected spikes.

        Faithful to ``neurons.loop()``: a continuous stream. The agent pulls a
        fixed window of ticks per decision. ``ticks_per_second`` is accepted for
        API parity; simulated time is governed by the dish ``dt``.
        """
        emitted = 0
        while stop_after_ticks is None or emitted < stop_after_ticks:
            yield self._advance_one(reward=0.0)
            emitted += 1

    def deliver_feedback(self, signal: float) -> None:
        """Broadcast the dopamine signal δ, converting standing eligibility
        (accumulated over the decision window) into a weight change via MSTDPET."""
        for _ in range(self.feedback_ticks):
            self._advance_one(reward=float(signal))

    def close(self) -> None:  # nothing to release for the sim backend.
        return None

    # -- internals ---------------------------------------------------------

    def _advance_one(self, *, reward: float) -> Tick:
        """Run exactly one simulated tick and package its motor spikes."""
        in_draw = torch.rand(self._n_in, generator=self._gen)
        in_spikes = (in_draw < self._in_rate).float()
        noise_spikes = (
            torch.rand(self._neurons, generator=self._gen) < self._noise_p
        ).float()
        inputs = {
            "in": in_spikes.view(1, 1, self._n_in),
            "noise": noise_spikes.view(1, 1, self._neurons),
        }
        self.net.run(inputs=inputs, time=1, reward=reward)

        analysis = TickAnalysis()
        fired = self.motor.s.view(-1).nonzero(as_tuple=False).view(-1).tolist()
        for i in fired:
            analysis.spikes.append(
                Spike(channel=self.motor_channels[i], timestamp=self._tick)
            )
        self._tick += 1
        return Tick(index=self._tick, analysis=analysis)

    def clear_stim(self) -> None:
        """Drop any pending stimulation (called between decision windows)."""
        self._in_rate.zero_()

    # -- introspection for training telemetry ------------------------------

    def plastic_weight_norm(self) -> float:
        return float(self._c_in.w.norm() + self._c_out.w.norm())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "w_in": self._c_in.w.detach().clone(),
            "w_out": self._c_out.w.detach().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._c_in.w.copy_(state["w_in"])
        self._c_out.w.copy_(state["w_out"])
