# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Skip optimizer steps whose grad norm is far above median of recent steps."""

import math
from collections import deque

import numpy as np
from omegaconf import OmegaConf

SPIKE_GUARD_DEFAULTS = {
    "enabled": False,
    # skip step if pre-clip grad norm > factor x median of last window finite norms (skipped incl.)
    "factor": 10.0,
    "window": 200,
    # no skips of finite norms in first warmup_steps steps of each job, history is not saved thus
    # median is refilled after restart. Also no skips during lr warmup (set by trainer)
    "warmup_steps": 100,
    # after this many finite skips in a row next step is taken thus training can not stall
    "max_consecutive": 20,
}


class SpikeGuard:
    """
    Decides per step if optimizer step is skipped.

    All finite norms go into history also of skipped steps. Median only counts ranks thus a short
    burst of k spikes moves it by k entries only, a lasting rise is accepted after ~window/2 steps.
    With taken steps only in history median never caught up with normal rise of norm and guard
    skipped 483 of first 700 steps (gfvbczcg).

    Non-finite norms are always skipped and do not count toward max_consecutive. Taking such a
    step would write nan into weights and optimizer state, therefore persistent nan skips forever.
    """

    def __init__(self, cfg=None, lr_warmup_steps: int = 0):
        self.cfg = OmegaConf.merge(OmegaConf.create(SPIKE_GUARD_DEFAULTS), cfg or {})
        self.enabled = bool(self.cfg.enabled)
        self.factor = float(self.cfg.factor)
        self.warmup_steps = int(self.cfg.warmup_steps)
        # norm rises fast and normally during lr warmup (gfvbczcg x10 in 40 steps) thus no skips
        # until global istep is past it. History is filled meanwhile
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.max_consecutive = int(self.cfg.max_consecutive)
        self.history = deque(maxlen=int(self.cfg.window))
        self.num_steps = 0
        self.num_skipped = 0
        self.num_consecutive = 0
        self.grad_norm_max = float("nan")

    def median(self) -> float:
        return float(np.median(self.history)) if self.history else float("nan")

    def should_skip(self, grad_norm: float, istep: int | None = None) -> bool:
        """istep = global optimizer step of run (as lr scheduler), None ignores lr warmup"""
        if not self.enabled:
            return False
        if not math.isfinite(grad_norm):
            return True
        if istep is not None and istep < self.lr_warmup_steps:
            return False
        if self.num_steps < self.warmup_steps or self.num_consecutive >= self.max_consecutive:
            return False
        return grad_norm > self.factor * self.median()

    def record(self, grad_norm: float, skipped: bool) -> None:
        self.num_steps += 1
        if skipped:
            self.num_skipped += 1
        if not math.isfinite(grad_norm):
            return
        # max since last metrics log, skipped ones included
        if not grad_norm <= self.grad_norm_max:
            self.grad_norm_max = grad_norm
        self.history.append(grad_norm)
        self.num_consecutive = self.num_consecutive + 1 if skipped else 0

    def metrics(self) -> dict[str, float]:
        """Metrics for train log, resets max norm."""
        out = {
            "spike_guard.num_skipped": self.num_skipped,
            "spike_guard.grad_norm_max": self.grad_norm_max,
            "spike_guard.grad_norm_median": self.median(),
        }
        self.grad_norm_max = float("nan")
        return out
