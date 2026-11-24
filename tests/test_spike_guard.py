# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CPU tests of spike guard: skip optimizer step of Muon + AdamW on grad norm spikes."""

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.train.optimizer import build_optimizer
from weathergen.train.spike_guard import SpikeGuard
from weathergen.train.trainer import Trainer

LR_CFG = OmegaConf.create(
    {"lr_start": 1e-3, "lr_max": 1e-3, "lr_final_decay": 1e-4, "lr_final": 0.0}
)
OPT_CFG = OmegaConf.create(
    {
        "name": "muon",
        "weight_decay": 0.1,
        "adamw": {"beta1": 0.9, "beta2": 0.95, "eps": 1e-8},
        "muon": {"adjust_lr_fn": "match_rms_adamw", "momentum": 0.95},
    }
)
GUARD = {"enabled": True, "factor": 10.0, "window": 20, "warmup_steps": 5, "max_consecutive": 3}


def _model():
    torch.manual_seed(0)
    # Linear weights to muon, biases + norm to adamw
    return torch.nn.Sequential(
        torch.nn.Linear(6, 8), torch.nn.LayerNorm(8), torch.nn.GELU(), torch.nn.Linear(8, 3)
    )


def _fake_trainer(model, guard_cfg, scaler=None, sync=None):
    built = build_optimizer(model, OPT_CFG, LR_CFG, kappa=1)
    fake = SimpleNamespace(
        model=model,
        optimizers=built.optimizers,
        grad_scaler=scaler or torch.amp.GradScaler("cpu", enabled=False),
        spike_guard=SpikeGuard(OmegaConf.create(guard_cfg) if guard_cfg is not None else None),
        cf=OmegaConf.create({"general": {"istep": 0}}),
        device=torch.device("cpu"),
        _get_tensor_item=lambda t: t.item(),
    )
    fake._sync_loss_spike_skip = sync or (lambda s: Trainer._sync_loss_spike_skip(fake, s))
    return fake


def _step(fake, loss_scale=1.0, seed=0):
    """Train step as in Trainer.train from backward to optimizer step."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(16, 6, generator=gen)
    for opt in fake.optimizers:
        opt.zero_grad()
    loss = fake.model(x).pow(2).mean() * loss_scale
    fake.grad_scaler.scale(loss).backward()
    for opt in fake.optimizers:
        fake.grad_scaler.unscale_(opt)
    total_norm = torch.nn.utils.clip_grad_norm_(fake.model.parameters(), max_norm=1.0)
    skipped = Trainer._optimizer_step(fake, total_norm, 0, 0)
    fake.cf.general.istep += 1
    return skipped


def _snapshot(fake):
    params = {n: p.detach().clone() for n, p in fake.model.named_parameters()}
    states = [copy.deepcopy(opt.state_dict()["state"]) for opt in fake.optimizers]
    return params, states


def _assert_unchanged(fake, snap):
    params, states = snap
    for n, p in fake.model.named_parameters():
        torch.testing.assert_close(p, params[n], rtol=0, atol=0)
    for opt, old in zip(fake.optimizers, states, strict=True):
        new = opt.state_dict()["state"]
        assert new.keys() == old.keys()
        for idx, entry in new.items():
            for key, val in entry.items():
                torch.testing.assert_close(val, old[idx][key], rtol=0, atol=0)


def test_off_same_as_before():
    # guard off = old loop, spike step taken, no grad norm .item() call
    ref = _fake_trainer(_model(), None)
    off = _fake_trainer(_model(), {"enabled": False})
    off._get_tensor_item = lambda t: pytest.fail("no sync when guard off")
    for i, scale in enumerate([1.0] * 6 + [1e4, 1.0]):
        gen = torch.Generator().manual_seed(i)
        x = torch.randn(16, 6, generator=gen)
        for opt in ref.optimizers:
            opt.zero_grad()
        (ref.model(x).pow(2).mean() * scale).backward()
        torch.nn.utils.clip_grad_norm_(ref.model.parameters(), max_norm=1.0)
        for opt in ref.optimizers:
            ref.grad_scaler.step(opt)
        ref.grad_scaler.update()
        assert not _step(off, scale, seed=i)
    for (n, p), q in zip(ref.model.named_parameters(), off.model.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0, msg=n)


def test_spike_skipped_state_unchanged():
    fake = _fake_trainer(_model(), GUARD)
    for i in range(6):
        assert not _step(fake, seed=i)
    assert type(fake.optimizers[0]) is torch.optim.Muon
    assert any("momentum_buffer" in s for s in fake.optimizers[0].state.values())
    assert any("exp_avg_sq" in s for s in fake.optimizers[1].state.values())
    history = list(fake.spike_guard.history)
    snap = _snapshot(fake)

    assert _step(fake, loss_scale=1e3, seed=6)
    # params, Muon momentum and AdamW moments / step count unchanged, grads dropped
    _assert_unchanged(fake, snap)
    assert all(p.grad is None for p in fake.model.parameters())
    # skipped norm in history too (median moves by one rank only)
    assert list(fake.spike_guard.history)[:-1] == history
    assert fake.spike_guard.num_skipped == 1
    assert fake.spike_guard.grad_norm_max > 10 * fake.spike_guard.median()

    # normal steps continue
    assert not _step(fake, seed=7)
    assert len(fake.spike_guard.history) == len(history) + 2
    assert fake.spike_guard.num_consecutive == 0


def test_warmup_respected():
    fake = _fake_trainer(_model(), GUARD)
    for i in range(3):
        assert not _step(fake, seed=i)
    snap = _snapshot(fake)
    # history shorter than warmup_steps thus spike taken and kept in history
    assert not _step(fake, loss_scale=1e3, seed=3)
    with pytest.raises(AssertionError):
        _assert_unchanged(fake, snap)
    assert len(fake.spike_guard.history) == 4
    assert fake.spike_guard.num_skipped == 0


def test_non_finite_skipped_even_in_warmup():
    fake = _fake_trainer(_model(), GUARD)
    assert not _step(fake, seed=0)
    snap = _snapshot(fake)
    assert _step(fake, loss_scale=float("nan"), seed=1)
    _assert_unchanged(fake, snap)
    assert len(fake.spike_guard.history) == 1
    assert fake.spike_guard.num_skipped == 1


def test_non_finite_with_grad_scaler():
    # scaler enabled, guard skip of non-finite step still lets scale back off as scaler.step skip
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**10)
    fake = _fake_trainer(_model(), GUARD, scaler=scaler)
    for i in range(6):
        assert not _step(fake, seed=i)
    scale = scaler.get_scale()
    snap = _snapshot(fake)
    assert _step(fake, loss_scale=float("inf"), seed=6)
    _assert_unchanged(fake, snap)
    assert scaler.get_scale() == scale * scaler.get_backoff_factor()
    # finite spike skipped too and next normal step works after update()
    assert _step(fake, loss_scale=1e3, seed=7)
    assert not _step(fake, seed=8)


def test_max_consecutive():
    fake = _fake_trainer(_model(), GUARD)
    for i in range(6):
        assert not _step(fake, seed=i)
    n = len(fake.spike_guard.history)
    skips = [_step(fake, loss_scale=1e3, seed=10 + i) for i in range(GUARD["max_consecutive"] + 1)]
    assert skips == [True] * GUARD["max_consecutive"] + [False]
    assert len(fake.spike_guard.history) == n + GUARD["max_consecutive"] + 1


def test_skip_from_other_rank():
    # all-reduced decision says skip (spike seen on other rank) thus no step
    fake = _fake_trainer(_model(), GUARD, sync=lambda s: True)
    snap = _snapshot(fake)
    assert _step(fake, seed=0)
    _assert_unchanged(fake, snap)
    assert fake.spike_guard.num_skipped == 1


def test_metrics_reset_max():
    guard = SpikeGuard(OmegaConf.create(GUARD))
    for g in [1.0, 2.0, 3.0]:
        guard.record(g, skipped=False)
    guard.record(50.0, skipped=True)
    guard.record(float("nan"), skipped=True)
    m = guard.metrics()
    assert m == {
        "spike_guard.num_skipped": 2,
        "spike_guard.grad_norm_max": 50.0,
        "spike_guard.grad_norm_median": 2.5,
    }
    assert math.isnan(guard.metrics()["spike_guard.grad_norm_max"])


# gfvbczcg grad_norm_max per 20 steps, first 800 steps. Norm rose naturally from 0.03 to 0.5 at
# ~115 steps (lr warmup). Old guard (median of taken steps only, warmup 100) skipped 483 steps here
# fmt: off
GFVBCZCG_MAX_20 = [
    0.0263, 0.0279, 0.0312, 0.0363, 0.0735, 0.334, 0.446, 0.487, 0.498, 0.473,
    0.502, 0.5, 0.528, 0.518, 0.509, 0.506, 0.444, 0.441, 0.547, 0.541,
    0.532, 0.509, 0.456, 0.459, 0.439, 0.437, 0.426, 0.433, 0.433, 0.41,
    0.37, 0.364, 0.213, 0.132, 0.0888, 0.0849, 0.0724, 0.0709, 0.0695, 0.0661,
]
# fmt: on


def _noise(n, seed=0):
    # per step log noise, gfvbczcg max of 20 steps is ~1.07 x median thus 0.15 is generous
    return np.exp(0.15 * np.random.default_rng(seed).standard_normal(n))


def _replay_series(n_steps=18000, spikes=None):
    """Per step proxy = block max x noise <= 1, tail at gfvbczcg level 0.06, x factor at spikes."""
    early = np.repeat(GFVBCZCG_MAX_20, 20)
    series = np.concatenate([early, np.full(n_steps - len(early), 0.06)])
    series = series * np.minimum(_noise(n_steps), 1.0)
    for step, factor in (spikes or {}).items():
        series[step] *= factor
    return series


def _replay(series, guard=None, istep0=0, lr_warmup=256, **cfg):
    """istep0 = global step of first entry (job restart), lr_warmup as our 256 step lr warmup"""
    guard = guard or SpikeGuard(OmegaConf.create({"enabled": True, **cfg}), lr_warmup)
    skips = []
    for i, x in enumerate(series):
        skip = guard.should_skip(float(x), istep0 + i)
        guard.record(float(x), skip)
        skips.append(skip)
    return np.array(skips)


def test_slow_rise_not_skipped():
    # 0.03 -> 0.3 over 700 steps, no lr warmup thus guard is active during rise
    series = np.geomspace(0.03, 0.3, 700) * _noise(700)
    assert _replay(series, lr_warmup=0, warmup_steps=100).sum() == 0


def test_replay_gfvbczcg_no_lock_in():
    series = _replay_series(n_steps=1500)
    # defaults: rise is inside lr warmup thus no skips
    assert _replay(series).sum() == 0
    # also without lr warmup level shift is accepted after about window/2 steps (old: 483 skips)
    assert _replay(series, lr_warmup=0, warmup_steps=100).sum() <= 110


def test_single_spike_skipped():
    series = 0.05 * _noise(1000)
    series[600] *= 100.0
    skips = _replay(series)
    assert skips[600] and skips.sum() == 1


def test_burst_skipped_median_stays():
    series = 0.05 * _noise(1000)
    series[600:610] *= 100.0
    guard = SpikeGuard(OmegaConf.create({"enabled": True}))
    _replay(series[:600], guard)
    median = guard.median()
    skips = _replay(series[600:], guard)
    assert skips[:10].all() and skips.sum() == 10
    # 10 of 200 entries are spikes thus median moves by a few percent at most
    assert guard.median() == pytest.approx(median, rel=0.1)


def test_replay_synthetic_spikes_skipped():
    spikes = {6500: 20.0, 17500: 50.0}
    burst = {12000 + i: 30.0 for i in range(20)}
    skips = _replay(_replay_series(spikes={**spikes, **burst}))
    assert skips[6500] and skips[17500]
    assert skips[12000:12020].all()
    assert skips.sum() == 22


def test_lasting_rise_accepted():
    # norm x15 for good after warmup, skipped until about half of window is at new level
    series = np.concatenate([np.full(1500, 0.05), np.full(1000, 0.75)])
    skips = _replay(series)
    assert 0 < skips.sum() <= 110
    assert not skips[1500 + 120 :].any()


def test_warmup_per_job():
    # history not in checkpoint thus each new guard (job restart) has own warmup of 100 steps
    series = 0.05 * _noise(3000)
    series[[200, 2000, 2100]] *= 50.0
    skips = _replay(series)
    # 200 is inside lr warmup of 256 steps
    assert not skips[200] and skips[2000] and skips[2100]
    # restart at 1950 thus spike at 2000 is inside warmup of new job, at 2100 median is refilled
    skips = _replay(series[1950:], istep0=1950)
    assert not skips[50] and skips[150]


def test_no_skips_during_lr_warmup():
    # lr warmup 512 (from scratch default): spike at 400 taken, at 600 skipped
    series = 0.05 * _noise(1000)
    series[[400, 600]] *= 50.0
    skips = _replay(series, lr_warmup=512)
    assert not skips[400] and skips[600]
    # without istep (None) lr warmup is ignored, only job warmup counts
    guard = SpikeGuard(OmegaConf.create({"enabled": True}), 512)
    for x in series[:400]:
        guard.record(float(x), guard.should_skip(float(x)))
    assert guard.should_skip(float(series[400]))


def test_continued_run_past_lr_warmup():
    # continued job starts far past lr warmup thus only job warmup (100) protects refill
    series = 0.05 * _noise(500)
    series[[50, 300]] *= 50.0
    skips = _replay(series, istep0=20000, lr_warmup=512)
    assert not skips[50] and skips[300]


def test_nan_skipped_forever_not_in_history():
    guard = SpikeGuard(OmegaConf.create(GUARD))
    _replay(0.05 * _noise(10), guard)
    history = list(guard.history)
    # persistent nan is never taken, also past max_consecutive
    skips = _replay([float("nan")] * (5 * GUARD["max_consecutive"]) + [float("inf")], guard)
    assert skips.all()
    assert list(guard.history) == history
    # nan skips do not count toward max_consecutive thus spike right after is still skipped
    assert guard.num_consecutive == 0
    assert _replay([5.0], guard).all()
    assert not _replay([0.05], guard).any()
