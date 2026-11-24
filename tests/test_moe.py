# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the mixture-of-experts MLP."""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

from weathergen.model.engines import ForecastingEngine
from weathergen.model.layers import MLP
from weathergen.model.moe import MoEMLP, update_moe_routing_biases

DIM = 16


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def _moe(**kwargs):
    defaults = dict(num_experts=4, top_k=2, hidden_factor=1.0, with_residual=True)
    return MoEMLP(DIM, DIM, **(defaults | kwargs))


def _per_token_reference(moe, x):
    """Slow MoE forward, one token and one expert at a time."""
    x_norm = moe.norm(x).reshape(-1, DIM)
    expert_idx, weights = moe.route(x_norm)
    ref = []
    for t in range(x_norm.shape[0]):
        out = torch.zeros(DIM)
        for e, w in zip(expert_idx[t].tolist(), weights[t], strict=True):
            h = moe.nonlin(x_norm[t] @ moe.w1[e] + moe.b1[e])
            out = out + w * (h @ moe.w2[e] + moe.b2[e])
        ref.append(out)
    return x + torch.stack(ref).reshape(x.shape)


def test_matches_per_token_reference():
    moe = _moe().eval()
    x = torch.randn(3, 5, DIM)

    torch.testing.assert_close(moe(x), _per_token_reference(moe, x))


def test_gradients_match_per_token_reference():
    moe = _moe().eval()
    x = torch.randn(3, 5, DIM)
    params = (moe.w1, moe.b1, moe.w2, moe.b2, moe.router.weight)

    moe(x).pow(2).sum().backward()
    grads = [p.grad.clone() for p in params]
    moe.zero_grad()
    _per_token_reference(moe, x).pow(2).sum().backward()

    for g, p in zip(grads, params, strict=True):
        torch.testing.assert_close(g, p.grad)


def test_single_expert_equals_dense_mlp():
    moe = _moe(num_experts=1, top_k=1, hidden_factor=2.0).eval()
    mlp = MLP(DIM, DIM, hidden_factor=2.0, with_residual=True).eval()
    with torch.no_grad():
        mlp.layers[0].load_state_dict(moe.norm.state_dict())
        mlp.layers[1].weight.copy_(moe.w1[0].t())
        mlp.layers[1].bias.copy_(moe.b1[0])
        mlp.layers[4].weight.copy_(moe.w2[0].t())
        mlp.layers[4].bias.copy_(moe.b2[0])
    x = torch.randn(2, 7, DIM)

    torch.testing.assert_close(moe(x, None, None), mlp(x, None, None))


def test_routing_selects_distinct_experts_with_normalised_weights():
    moe = _moe(num_experts=8, top_k=3)
    expert_idx, weights = moe.route(torch.randn(50, DIM))

    assert expert_idx.shape == weights.shape == (50, 3)
    assert all(len(set(row.tolist())) == 3 for row in expert_idx)
    torch.testing.assert_close(weights.sum(-1), torch.ones(50))


def test_tokens_counted_once_with_activation_checkpointing():
    moe = _moe().train()
    x = torch.randn(2, 10, DIM, requires_grad=True)

    y = checkpoint(moe, x, None, None, use_reentrant=False)
    y.sum().backward()

    counts = moe.take_token_counts()
    assert counts.sum().item() == 2 * 10 * moe.top_k
    assert moe.take_token_counts().sum().item() == 0


def test_no_tokens_counted_in_eval():
    moe = _moe().eval()
    moe(torch.randn(4, DIM))
    assert moe.tokens_per_expert is None


def test_gradients_reach_router_and_experts():
    moe = _moe().train()
    moe(torch.randn(4, 6, DIM)).pow(2).sum().backward()

    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0
    for p in (moe.w1, moe.b1, moe.w2, moe.b2):
        assert p.grad is not None
    assert moe.routing_bias.grad is None


def _skewed_moe(num_experts, top_k):
    """MoE whose router sends (almost) every token to expert 0."""
    moe = _moe(num_experts=num_experts, top_k=top_k, pre_layer_norm=False).train()
    with torch.no_grad():
        moe.router.weight.normal_(0.0, 0.1)
        moe.router.weight[0] += 0.3
    return moe


def test_bias_update_lowers_overused_expert():
    num_experts, rate = 4, 1e-3
    moe = _skewed_moe(num_experts, top_k=1)
    moe(torch.randn(64, DIM) + 2.0)

    stats = update_moe_routing_biases(moe, rate)

    bias = moe.routing_bias.detach()
    assert bias[0] < 0 and (bias[1:] > 0).all()
    torch.testing.assert_close(bias.sum(), torch.tensor(0.0))
    assert stats["moe..max_load"].item() == pytest.approx(num_experts)


def test_bias_update_balances_load_over_steps():
    moe = _skewed_moe(num_experts=4, top_k=1)
    x = torch.randn(256, DIM) + 2.0

    loads = []
    for _ in range(300):
        moe(x)
        loads.append(update_moe_routing_biases(moe, 0.01)["moe..max_load"].item())

    assert loads[0] == pytest.approx(4.0)
    assert loads[-1] < 1.5


def test_update_without_moe_layers_is_noop():
    assert update_moe_routing_biases(nn.Linear(4, 4), 1e-3) == {}


def _forecasting_cfg(**overrides):
    cf = dict(
        forecast_att_dense_rate=1.0,
        fe_num_blocks=3,
        fe_num_heads=2,
        fe_dropout_rate=0.0,
        fe_with_qk_lnorm=True,
        ae_global_dim_embed=DIM,
        ae_local_num_queries=1,
        ae_global_block_factor=4,
        with_flash_attention=True,
        norm_type="LayerNorm",
        norm_eps=1e-5,
        mlp_norm_eps=1e-5,
        attention_dtype="fp32",
    )
    return OmegaConf.create(cf | overrides)


@pytest.mark.parametrize("mlp_type", ["mlp", "moe"])
def test_forecasting_engine_mlp_type(mlp_type):
    cf = _forecasting_cfg(fe_mlp_type=mlp_type, fe_moe_num_dense_blocks=1)
    mode_cfg = OmegaConf.create({"forecast": {"policy": "fixed"}})

    engine = ForecastingEngine(cf, mode_cfg, num_healpix_cells=12)

    mlps = [b for b in engine.fe_blocks if isinstance(b, MLP | MoEMLP)]
    expected = [MLP, MLP, MLP] if mlp_type == "mlp" else [MLP, MoEMLP, MoEMLP]
    assert [type(b) for b in mlps] == expected
    # experts get the small init of the forecasting engine
    for moe in mlps[1:] if mlp_type == "moe" else []:
        assert moe.w1.std().item() == pytest.approx(1e-3, rel=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("mlp_type", ["mlp", "moe"])
def test_forecasting_engine_forward_on_gpu(mlp_type):
    cf = _forecasting_cfg(fe_mlp_type=mlp_type, attention_dtype="bf16")
    mode_cfg = OmegaConf.create({"forecast": {"policy": "fixed"}})
    engine = ForecastingEngine(cf, mode_cfg, num_healpix_cells=12).cuda().eval()

    x = torch.randn(2, 12, DIM, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = engine(x, 0)

    assert y.shape == x.shape and torch.isfinite(y).all()


def test_forecasting_engine_rejects_unknown_mlp_type():
    cf = _forecasting_cfg(fe_mlp_type="dense")
    mode_cfg = OmegaConf.create({"forecast": {"policy": "fixed"}})
    with pytest.raises(ValueError, match="fe_mlp_type"):
        ForecastingEngine(cf, mode_cfg, num_healpix_cells=12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_bf16_autocast_with_checkpointing_on_gpu():
    moe = _moe(num_experts=8, top_k=2).cuda().train()
    x = torch.randn(2, 128, DIM, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = checkpoint(moe, x, None, None, use_reentrant=False)
    y.float().sum().backward()

    assert torch.isfinite(x.grad).all()
    assert moe.take_token_counts().sum().item() == 2 * 128 * 2
