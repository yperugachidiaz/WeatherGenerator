# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, distribute_tensor

from weathergen.model.norms import AdaLayerNorm, RMSNorm


class MoEMLP(nn.Module):
    """
    Mixture-of-experts version of MLP, with the same call signature.

    Each token goes to its top_k of num_experts small MLPs (sigmoid router as in DeepSeek-V3),
    no tokens are dropped. The routing bias only changes which experts are picked and is set by
    update_moe_routing_biases() to keep the experts evenly used.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        num_experts=8,
        top_k=2,
        hidden_factor=1.0,
        pre_layer_norm=True,
        dropout_rate=0.0,
        nonlin=nn.GELU,
        with_residual=False,
        norm_type="LayerNorm",
        dim_aux=None,
        norm_eps=1e-5,
        name: str | None = None,
    ):
        super().__init__()

        if name is not None:
            self.name = name

        assert 1 <= top_k <= num_experts, "top_k must be in [1, num_experts]"
        assert not with_residual or dim_in == dim_out, "residual requires dim_in == dim_out"

        self.num_experts = num_experts
        self.top_k = top_k
        self.with_residual = with_residual
        self.with_aux = dim_aux is not None
        dim_hidden = int(dim_in * hidden_factor)

        self.norm = None
        if pre_layer_norm:
            norm = nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm
            self.norm = (
                norm(dim_in, eps=norm_eps)
                if dim_aux is None
                else AdaLayerNorm(dim_in, dim_aux, norm_eps=norm_eps)
            )

        self.router = nn.Linear(dim_in, num_experts, bias=False)
        # frozen parameter (not a buffer) so checkpoints, FSDP and EMA handle it like other weights
        self.routing_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)

        # weights of all experts stacked along dim 0
        self.w1 = nn.Parameter(torch.empty(num_experts, dim_in, dim_hidden))
        self.b1 = nn.Parameter(torch.empty(num_experts, dim_hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim_hidden, dim_out))
        self.b2 = nn.Parameter(torch.empty(num_experts, dim_out))
        self.nonlin = nonlin()
        self.dropout = nn.Dropout(p=dropout_rate)

        # tokens per expert since the last bias update (plain attribute, so not saved or synced)
        self.tokens_per_expert = None

        self.reset_parameters()

    def reset_parameters(self):
        """
        Init experts like nn.Linear and zero the routing bias; router and norm reset themselves.
        """
        for w, b in ((self.w1, self.b1), (self.w2, self.b2)):
            bound = 1.0 / math.sqrt(w.shape[1])
            nn.init.uniform_(w, -bound, bound)
            nn.init.uniform_(b, -bound, bound)
        with torch.no_grad():
            self.routing_bias.zero_()
        self.tokens_per_expert = None

    def init_experts_normal(self, std):
        """Init expert weights from N(0, std), like the forecasting engine's MLPs."""
        for p in (self.w1, self.b1, self.w2, self.b2):
            nn.init.normal_(p, mean=0.0, std=std)

    def route(self, x):
        """
        Pick the top_k experts per token.

        Args:
            x: tokens [num_tokens, dim_in].

        Returns:
            Expert indices and routing weights, both [num_tokens, top_k].
        """
        # route in fp32 for stability
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = torch.sigmoid(x.float() @ self.router.weight.float().t())
            _, expert_idx = torch.topk(scores + self.routing_bias.float(), self.top_k, dim=-1)
            weights = scores.gather(-1, expert_idx)
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return expert_idx, weights

    def _count_tokens(self, counts):
        # count only in training, and skip the recompute of activation checkpointing in backward
        if not self.training or torch._C._current_graph_task_id() != -1:
            return
        counts = counts.detach()
        if self.tokens_per_expert is None:
            self.tokens_per_expert = counts
        else:
            self.tokens_per_expert = self.tokens_per_expert + counts

    def take_token_counts(self):
        """Return the tokens per expert since the last call and reset the count."""
        counts = self.tokens_per_expert
        self.tokens_per_expert = None
        if counts is None:
            counts = torch.zeros(
                self.num_experts, dtype=torch.long, device=self.routing_bias.device
            )
        return counts

    @torch.no_grad()
    def add_to_routing_bias(self, delta):
        """Add delta [num_experts] to the routing bias, also when FSDP shards it."""
        bias = self.routing_bias
        if isinstance(bias, DTensor):
            delta = distribute_tensor(
                delta.to(bias.dtype), bias.device_mesh, bias.placements, src_data_rank=None
            )
        bias.add_(delta)

    def _run_experts(self, x, onehot, counts):
        """
        Apply all experts with one grouped matmul per layer, without a host sync.

        Args:
            x: tokens sorted by expert [num_pairs, dim_in].
            onehot: expert of each row as one-hot [num_pairs, num_experts].
            counts: tokens per expert [num_experts].
        """
        # grouped_mm ignores autocast, so cast like nn.Linear would
        dev = x.device.type
        dtype = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype
        offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
        # biases via a one-hot matmul, whose backward needs no atomic adds
        onehot = onehot.to(dtype)

        h = torch._grouped_mm(x.to(dtype), self.w1.to(dtype), offs=offs)
        h = self.dropout(self.nonlin(h + onehot @ self.b1.to(dtype)))
        y = torch._grouped_mm(h, self.w2.to(dtype), offs=offs)
        return y + onehot @ self.b2.to(dtype)

    def forward(self, *args):
        # FSDP skips its input cast when activation checkpointing recomputes this block,
        # so cast here or tokens near a tie can switch experts in the recompute
        x = x_in = args[0].to(self.w1.dtype)
        aux = args[-1]

        if self.norm is not None:
            x = self.norm(x, aux) if self.with_aux else self.norm(x)

        shape = x.shape
        x = x.reshape(-1, shape[-1])
        num_tokens = x.shape[0]

        expert_idx, weights = self.route(x)

        # sort (token, expert) pairs by expert, so each expert gets one contiguous chunk
        flat_expert_idx = expert_idx.flatten()
        order = torch.argsort(flat_expert_idx, stable=True)
        onehot = F.one_hot(flat_expert_idx[order], self.num_experts)
        counts = onehot.sum(dim=0)  # not bincount, which syncs with the host
        self._count_tokens(counts)

        # index_select, since the backward of x[idx] uses a slow sort-based kernel
        x_sorted = x.index_select(0, order // self.top_k)
        y_sorted = self._run_experts(x_sorted, onehot, counts)

        # unsort, then weighted sum over the top_k experts of each token
        y = torch.empty_like(y_sorted)
        y[order] = y_sorted
        y = y.reshape(num_tokens, self.top_k, -1)
        y = (y * weights.unsqueeze(-1).to(y.dtype)).sum(dim=1)
        y = y.reshape(*shape[:-1], -1)

        if self.with_residual:
            y = x_in + y

        return y


@torch.no_grad()
def update_moe_routing_biases(model: nn.Module, update_rate: float) -> dict[str, torch.Tensor]:
    """
    Update the routing bias of each MoEMLP so its experts are used evenly.

    Call after every optimizer step. Experts above the mean token count (summed over ranks) get
    their bias lowered by update_rate, the others raised.

    Args:
        model: model with MoEMLP layers.
        update_rate: step size of the bias, 0 only returns the load.

    Returns:
        Per MoE layer, busiest expert load / mean load (1.0 = even); empty without MoE layers.
    """
    layers = [(name, m) for name, m in model.named_modules() if isinstance(m, MoEMLP)]
    if len(layers) == 0:
        return {}

    counts = torch.stack([m.take_token_counts() for _, m in layers]).float()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts)

    mean = counts.mean(dim=-1, keepdim=True)
    delta = update_rate * torch.sign(mean - counts)
    delta = delta - delta.mean(dim=-1, keepdim=True)  # zero mean, so the biases don't drift

    max_load = counts.max(dim=-1).values / mean.squeeze(-1).clamp_min(1.0)
    stats = {}
    for (name, m), d, load in zip(layers, delta, max_load, strict=True):
        if update_rate > 0.0:
            m.add_to_routing_bias(d)
        stats[f"moe.{name}.max_load"] = load
    return stats
