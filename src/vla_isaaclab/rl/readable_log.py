"""A short, plain training summary per PPO iteration.

RSL-RL prints ``Episode_Termination/<term>`` as a per-step average of reset
counts, which does not read as a rate. This runner attributes every episode
that ends to the termination that ended it and prints how episodes end, e.g.::

    iteration 480/500 (1.6 s each, 7.5 min so far, about 0.5 min left)
      of all episodes: success 77% (1.7 s) | lifted, off pose 3% (2.0 s) | object fell 0% (-) | timed out 20% (8.0 s)
      1071 episodes ended this iteration; average return 2.37, action noise std 1.35

Short episodes end more often than long ones, so the plain share of episodes
that ended in an iteration overstates quick outcomes (above: 94% success). The
printed rates weight each outcome by its episode duration, which estimates the
share of started episodes ending that way (exact for a fixed policy; the
episode running at the start of training is counted from there). They go to
TensorBoard as ``Outcome/<term>``; ``verbose=True`` also prints RSL-RL's block.
"""

from __future__ import annotations

import contextlib
import io
import statistics
import time

import torch
from rsl_rl.runners import OnPolicyRunner


# Plain names for known termination terms; others print as their term name.
OUTCOME_LABELS = {
    "success": "success",
    "lifted_off_pose": "lifted, off pose",
    "fell": "object fell",
    "invalid_state": "invalid state",
    "time_out": "timed out",
}


class ReadableOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", verbose: bool = False):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.verbose = verbose
        self.step_dt = env.unwrapped.step_dt
        manager = env.unwrapped.termination_manager
        # An episode flagged by several terms in one step counts once: for the
        # first non-time-out term in config order, time-out last.
        names = list(manager.active_terms)
        self.outcome_names = [n for n in names if n != "time_out"] + [n for n in names if n == "time_out"]
        device = env.unwrapped.device
        self.outcome_counts = torch.zeros(len(self.outcome_names), dtype=torch.long, device=device)
        self.outcome_steps = torch.zeros_like(self.outcome_counts)
        self.episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        self._learn_start = None
        self._first_iteration = None

        step = env.step

        def counting_step(actions):
            result = step(actions)
            self.episode_steps += 1
            unclaimed = torch.ones(env.num_envs, dtype=torch.bool, device=self.outcome_counts.device)
            for index, name in enumerate(self.outcome_names):
                ended = manager.get_term(name) & unclaimed
                self.outcome_counts[index] += ended.sum()
                self.outcome_steps[index] += self.episode_steps[ended].sum()
                unclaimed &= ~ended
            self.episode_steps[~unclaimed] = 0
            return result

        env.step = counting_step

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        if self._learn_start is None:
            self._learn_start = time.time() - locs["collection_time"] - locs["learn_time"]
            self._first_iteration = locs["it"]
        if self.verbose:
            super().log(locs, width, pad)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                super().log(locs, width, pad)

        counts, steps = self.outcome_counts.tolist(), self.outcome_steps.tolist()
        self.outcome_counts.zero_()
        self.outcome_steps.zero_()
        ended, total_steps = sum(counts), sum(steps)
        rates = [s / total_steps if total_steps else 0.0 for s in steps]
        for name, rate in zip(self.outcome_names, rates):
            self.writer.add_scalar(f"Outcome/{name}", rate, locs["it"])
        outcomes = " | ".join(
            f"{OUTCOME_LABELS.get(name, name)} {rate:.0%} "
            + (f"({s / c * self.step_dt:.1f} s)" if c else "(-)")
            for name, rate, c, s in zip(self.outcome_names, rates, counts, steps)
        )

        done = locs["it"] - self._first_iteration + 1
        elapsed = time.time() - self._learn_start
        left = (locs["tot_iter"] - locs["it"] - 1) * elapsed / done
        lines = [
            f"iteration {locs['it']}/{locs['tot_iter'] - 1} "
            f"({elapsed / done:.1f} s each, {elapsed / 60:.1f} min so far, about {left / 60:.1f} min left)",
            f"  of all episodes: {outcomes}",
            f"  {ended} episodes ended this iteration; "
            + (f"average return {statistics.mean(locs['rewbuffer']):.2f}, " if locs["rewbuffer"] else "")
            + f"action noise std {self.alg.actor_critic.std.mean().item():.2f}",
        ]
        print("\n".join(lines), flush=True)
