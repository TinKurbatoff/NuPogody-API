"""``CL1Agent`` — a spiking policy that plays through the CL-shaped interface.

Satisfies the standard :class:`~nupogodi.agents.base.Agent` contract
(``act``/``observe``), so it drops into ``rollout.run`` and the recorder /
dashboard exactly like :class:`~nupogodi.agents.random_agent.RandomAgent`. What
makes it special is *how* it decides and learns — entirely through a
:class:`~nupogodi.cl1.api.NeuronsLike` neural interface, never touching the
network directly:

* **act** — rate-encode the observation into electrode stimulation, run a short
  decision window on the culture, read the motor electrodes, take the
  most-active action. Exploration comes from the culture's intrinsic noise.
* **observe** — a tiny linear **critic** turns the reward into a
  reward-prediction error δ (= dopamine); ``deliver_feedback(δ)`` broadcasts it
  so the dish's three-factor STDP converts the decision window's standing
  eligibility into a weight change. No backprop.

The same code runs against the learning :class:`~nupogodi.cl1.dish.SpikingDish`
or a real CL1 via ``cl`` — only the substrate behind ``neurons`` changes.
"""

from __future__ import annotations

import numpy as np

from ..cl1 import MAX_RATE_HZ, BurstDesign, ChannelSet, NeuronsLike, StimDesign
from ..cl1.dish import DEFAULT_IN_CHANNELS, DEFAULT_MOTOR_CHANNELS
from .base import Transition

_STIM = StimDesign(160, -1.0)  # a fixed charge-balanced pulse; rate carries the signal.


class CL1Agent:
    """Reward-modulated spiking agent driven through a CL neural interface."""

    def __init__(
        self,
        neurons: NeuronsLike,
        *,
        in_channels: tuple[int, ...] = DEFAULT_IN_CHANNELS,
        motor_channels: tuple[int, ...] = DEFAULT_MOTOR_CHANNELS,
        window: int = 12,
        burst_pulses: int = 10,
        gamma: float = 0.9,
        critic_lr: float = 0.05,
        reward_scale: float = 1.0,
        delta_clip: float = 5.0,
        learning: bool = True,
        seed: int | None = None,
    ) -> None:
        self.neurons = neurons
        self.in_channels = tuple(in_channels)
        self.motor_channels = tuple(motor_channels)
        self.window = window
        self.burst_pulses = burst_pulses
        self.gamma = gamma
        self.critic_lr = critic_lr
        self.reward_scale = reward_scale
        self.delta_clip = delta_clip
        self.learning = learning
        self._rng = np.random.default_rng(seed)

        # Linear value critic V(obs) = w·[obs, 1]; +1 for the bias feature.
        self._critic = np.zeros(len(self.in_channels) + 1, dtype=np.float64)
        self._last_value = 0.0
        self._last_delta = 0.0  # exposed for training telemetry.

    # -- Agent contract ----------------------------------------------------

    def act(self, obs: np.ndarray) -> int:
        """Encode → stimulate → run a decision window → read the motor readout."""
        self.neurons.clear_stim() if hasattr(self.neurons, "clear_stim") else None
        for ch, value in zip(self.in_channels, obs, strict=False):
            if value <= 0:
                continue  # a silent electrode *is* the signal for an empty feature.
            rate = int(np.clip(round(float(value) * MAX_RATE_HZ), 1, MAX_RATE_HZ))
            self.neurons.stim(
                ChannelSet(ch), _STIM, BurstDesign(self.burst_pulses, rate)
            )

        counts = dict.fromkeys(self.motor_channels, 0)
        for tick in self.neurons.loop(stop_after_ticks=self.window):
            for spike in tick.analysis.spikes:
                if spike.channel in counts:
                    counts[spike.channel] += 1

        self._last_value = self._value(obs)
        return self._argmax_action(counts)

    def observe(self, transition: Transition) -> None:
        """Turn reward into a reward-prediction error and deliver it as dopamine."""
        if not self.learning:
            return
        next_value = 0.0 if transition.terminated else self._value(transition.next_obs)
        delta = transition.reward + self.gamma * next_value - self._last_value
        delta = float(np.clip(delta, -self.delta_clip, self.delta_clip))
        self._last_delta = delta
        # TD(0) update of the linear critic.
        self._critic += self.critic_lr * delta * self._features(transition.obs)
        # Broadcast δ; the dish's MSTDPET shapes synapses against standing eligibility.
        self.neurons.deliver_feedback(delta * self.reward_scale)

    # -- helpers -----------------------------------------------------------

    def _features(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(obs, dtype=np.float64), [1.0]])

    def _value(self, obs: np.ndarray) -> float:
        return float(self._critic @ self._features(obs))

    def _argmax_action(self, counts: dict[int, int]) -> int:
        best = max(counts.values())
        winners = [i for i, c in enumerate(self.motor_channels) if counts[c] == best]
        return int(self._rng.choice(winners))  # random tie-break (incl. all-zero)

    @property
    def last_delta(self) -> float:
        return self._last_delta

    # -- checkpointing -----------------------------------------------------

    def save(self, path: str) -> None:
        import torch

        state = {"critic": self._critic}
        if hasattr(self.neurons, "state_dict"):
            state["dish"] = self.neurons.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        import torch

        state = torch.load(path, weights_only=False)
        self._critic = np.asarray(state["critic"], dtype=np.float64)
        if "dish" in state and hasattr(self.neurons, "load_state_dict"):
            self.neurons.load_state_dict(state["dish"])
