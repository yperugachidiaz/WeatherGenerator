# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CPU tests of muon / adamw param split, split_fused and optimizer state save / load."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

import weathergen.train.trainer as trainer_module
from weathergen.model.embeddings import StreamEmbedLinear
from weathergen.model.engines import EnsPredictionHead
from weathergen.model.layers import MLP
from weathergen.model.norms import AdaLayerNorm
from weathergen.train.optimizer import build_optimizer
from weathergen.train.optimizer.optimizer import MuonSplitFused, _muon_adjust_lr_factor
from weathergen.train.trainer import Trainer

LR_CFG = OmegaConf.create(
    {"lr_start": 1e-4, "lr_max": 1e-3, "lr_final_decay": 1e-4, "lr_final": 0.0}
)
MOVED = {
    "encoder.embed_engine.embeds.ERA5_in.layer.weight",
    "embed_target_coords.ERA5.weight",
    "target_token_engines.ERA5.embed_aux.0.weight",
    "pred_heads.ERA5.pred_heads.0.0.weight",
}


class TinyModel(torch.nn.Module):
    """param names as in real model"""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.embed_engine = torch.nn.Module()
        self.encoder.embed_engine.embeds = torch.nn.ModuleDict(
            {"ERA5_in": StreamEmbedLinear(12, 16)}
        )
        self.encoder.q_cells = torch.nn.Parameter(torch.rand(1, 1, 16))
        self.blocks = torch.nn.ModuleList(
            [MLP(16, 16, mlp_type="swiglu", with_residual=True), torch.nn.Linear(16, 16)]
        )
        self.embed_target_coords = torch.nn.ModuleDict({"ERA5": torch.nn.Linear(6, 16)})
        self.target_token_engines = torch.nn.ModuleDict({"ERA5": AdaLayerNorm(16, 6)})
        self.pred_heads = torch.nn.ModuleDict({"ERA5": EnsPredictionHead(16, 5, 1, 1, "ERA5")})

    def forward(self, x, coords):
        h = self.encoder.embed_engine.embeds["ERA5_in"](x).squeeze(0) + self.encoder.q_cells[0]
        for block in self.blocks:
            h = block(h)
        h = h + self.embed_target_coords["ERA5"](coords)
        h = self.target_token_engines["ERA5"](h, coords)
        return self.pred_heads["ERA5"](h)


def _optimizer_cfg(**muon):
    return OmegaConf.create(
        {
            "name": "muon",
            "weight_decay": 0.1,
            "adamw": {"beta1": 0.9, "beta2": 0.95, "eps": 1e-8},
            "muon": {"adjust_lr_fn": "match_rms_adamw", **muon},
        }
    )


def _owner_names(model, built):
    names = {id(p): n for n, p in model.named_parameters()}
    return {
        oname: [names[id(p)] for g in opt.param_groups for p in g["params"]]
        for opt, oname in zip(built.optimizers, built.optimizer_names, strict=True)
    }


def _train_steps(model, built, num_steps=2, seed=0):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(num_steps):
        x, coords = torch.randn(3, 4, 3, generator=gen), torch.randn(3, 6, generator=gen)
        for opt in built.optimizers:
            opt.zero_grad()
        model(x, coords).pow(2).mean().backward()
        for opt in built.optimizers:
            opt.step()


def test_muon_default_split_unchanged():
    model = TinyModel()
    built = build_optimizer(model, _optimizer_cfg(), LR_CFG, kappa=1)
    owners = _owner_names(model, built)
    # old rule, every trainable 2D param to muon in model order
    assert owners["muon"] == [n for n, p in model.named_parameters() if p.ndim == 2]
    assert owners["adamw"] == [n for n, p in model.named_parameters() if p.ndim != 2]
    assert type(built.optimizers[0]) is torch.optim.Muon
    shapes = [p.shape for p in model.parameters() if p.ndim == 2]
    factor = np.median([_muon_adjust_lr_factor(s, "match_rms_adamw") for s in shapes])
    assert built.muon_effective_lr_factor == pytest.approx(factor)


def test_muon_adamw_embed_head_moves_embed_and_head():
    model = TinyModel()
    built = build_optimizer(model, _optimizer_cfg(adamw_embed_head=True), LR_CFG, kappa=1)
    owners = _owner_names(model, built)
    assert set(owners["adamw"]) >= MOVED
    assert not MOVED & set(owners["muon"])
    two_d = {n for n, p in model.named_parameters() if p.ndim == 2}
    assert set(owners["muon"]) == two_d - MOVED
    assert "encoder.q_cells" in owners["adamw"]
    shapes = [p.shape for n, p in model.named_parameters() if n in two_d - MOVED]
    factor = np.median([_muon_adjust_lr_factor(s, "match_rms_adamw") for s in shapes])
    assert built.muon_effective_lr_factor == pytest.approx(factor)


def test_muon_adamw_moved_lr_max_own_adamw():
    model = TinyModel()
    cfg = _optimizer_cfg(adamw_embed_head=True, adamw_moved_lr_max=5e-3)
    built = build_optimizer(model, cfg, LR_CFG, kappa=1)
    owners = _owner_names(model, built)
    assert built.optimizer_names == ["muon", "adamw", "adamw_moved"]
    assert isinstance(built.optimizers[2], torch.optim.AdamW)
    assert set(owners["adamw_moved"]) == MOVED
    assert not MOVED & (set(owners["muon"]) | set(owners["adamw"]))
    assert "encoder.q_cells" in owners["adamw"]
    # moved params get own lr schedule scaled to their lr_max, rest unchanged
    assert built.lr_cfgs[1].lr_max == LR_CFG.lr_max
    assert built.lr_cfgs[2].lr_max == 5e-3
    assert built.optimizers[2].param_groups[0]["lr"] == pytest.approx(5 * LR_CFG.lr_start)
    # nothing moved thus no extra optimizer
    built = build_optimizer(TinyModel(), _optimizer_cfg(adamw_moved_lr_max=5e-3), LR_CFG, kappa=1)
    assert built.optimizer_names == ["muon", "adamw"]


def test_muon_adamw_param_patterns():
    model = TinyModel()
    cfg = _optimizer_cfg(adamw_param_patterns=[r"^blocks\.1\."])
    owners = _owner_names(model, build_optimizer(model, cfg, LR_CFG, kappa=1))
    assert "blocks.1.weight" in owners["adamw"]
    assert "blocks.1.weight" not in owners["muon"]
    assert "pred_heads.ERA5.pred_heads.0.0.weight" in owners["muon"]


def test_muon_split_fused_per_block():
    torch.manual_seed(0)
    model = TinyModel()
    ref = TinyModel()
    ref.load_state_dict(model.state_dict())
    built = build_optimizer(model, _optimizer_cfg(split_fused=True), LR_CFG, kappa=1)
    muon = built.optimizers[0]
    assert isinstance(muon, MuonSplitFused)
    named = model.named_parameters()
    fused = {n: muon.num_parts[id(p)] for n, p in named if id(p) in muon.num_parts}
    assert fused == {
        "blocks.0.layers.1.weight": 2,
        "target_token_engines.ERA5.embed_aux.2.weight": 2,
    }

    # zero momentum after one step thus update = orthogonalized (1 - momentum) grad per block
    _train_steps(model, built, num_steps=1)
    _train_steps(ref, SimpleNamespace(optimizers=[]), num_steps=1)
    group = muon.param_groups[0]
    lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
    ns = (group["ns_coefficients"], group["ns_steps"], group["eps"])
    for (name, p), p_ref in zip(model.named_parameters(), ref.parameters(), strict=True):
        if p.ndim != 2:
            continue
        k = fused.get(name, 1)
        g = p_ref.grad
        update = g.lerp(g * (1 - mom), mom)
        blocks = [_zeropower_via_newtonschulz(b, *ns) for b in update.chunk(k, dim=0)]
        adj = _adjust_lr(lr, "match_rms_adamw", (p.shape[0] // k, p.shape[1]))
        expected = p_ref.detach() * (1 - lr * wd) - adj * torch.cat(blocks, dim=0).float()
        torch.testing.assert_close(p.detach(), expected)

    # non fused params same as torch.optim.Muon. One step only since fused params change next
    # grads
    torch.manual_seed(1)
    a, b = TinyModel(), TinyModel()
    b.load_state_dict(a.state_dict())
    built_a = build_optimizer(a, _optimizer_cfg(), LR_CFG, kappa=1)
    built_b = build_optimizer(b, _optimizer_cfg(split_fused=True), LR_CFG, kappa=1)
    _train_steps(a, built_a, num_steps=1)
    _train_steps(b, built_b, num_steps=1)
    num_parts = built_b.optimizers[0].num_parts
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        if id(pb) not in num_parts and pa.ndim == 2:
            torch.testing.assert_close(pa, pb)


def _fake_trainer(model, built, tmp_path):
    fake = SimpleNamespace(
        model=model,
        optimizers=built.optimizers,
        cf=OmegaConf.create({"general": {"run_id": "test"}, "with_ddp": False, "with_fsdp": False}),
        training_cfg=OmegaConf.create({"num_mini_epochs": 1}),
        _get_full_model_state_dict=model.state_dict,
        _get_ema_teacher=lambda: None,
    )
    fake._get_full_optimizer_state_dict = lambda: Trainer._get_full_optimizer_state_dict(fake)
    return fake


ON = {"adamw_embed_head": True}
ON_OWN = {"adamw_embed_head": True, "adamw_moved_lr_max": 5e-3}


# flag off as before, flag on in new run, flag switched on when continuing a flag off run, same
# with own adamw for moved params and moving from shared to own adamw
@pytest.mark.parametrize(
    ("save_cfg", "load_cfg"),
    [({}, {}), (ON, ON), ({}, ON), (ON_OWN, ON_OWN), ({}, ON_OWN), (ON, ON_OWN)],
)
def test_optimizer_state_save_load(tmp_path, monkeypatch, save_cfg, load_cfg):
    monkeypatch.setattr(trainer_module.config, "get_path_model", lambda *a, **k: tmp_path)
    monkeypatch.setattr(trainer_module.config, "save", lambda *a, **k: None)
    torch.manual_seed(0)
    model = TinyModel()
    built = build_optimizer(model, _optimizer_cfg(**save_cfg), LR_CFG, kappa=1)
    _train_steps(model, built)
    Trainer.save_model(_fake_trainer(model, built, tmp_path), 0)

    model2 = TinyModel()
    model2.load_state_dict(model.state_dict())
    built2 = build_optimizer(model2, _optimizer_cfg(**load_cfg), LR_CFG, kappa=1)
    Trainer._load_optimizer_state(_fake_trainer(model2, built2, tmp_path), "test", 0)

    def state_by_name(m, optimizers):
        return {
            n: opt.state[p] for opt in optimizers for n, p in m.named_parameters() if p in opt.state
        }

    old_state = state_by_name(model, built.optimizers)
    new_state = state_by_name(model2, built2.optimizers)
    for name, _ in model2.named_parameters():
        state = new_state.get(name, {})
        if ("adamw_embed_head" in save_cfg) != ("adamw_embed_head" in load_cfg) and name in MOVED:
            # muon state can't be used by adamw thus these start fresh
            assert state == {}
            continue
        assert state.keys() == old_state[name].keys()
        for key, val in state.items():
            torch.testing.assert_close(val, old_state[name][key])

    # training continues on loaded state
    _train_steps(model2, built2, num_steps=1)
