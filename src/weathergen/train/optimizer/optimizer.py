# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import re

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.distributed.tensor import DTensor, Replicate
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz
from torch.optim.optimizer import _to_scalar

from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)

# 2D params that are not hidden layers (input embeddings, output heads). Muon is meant for hidden
# layers only thus with adamw_embed_head these go to adamw (Jordan 2024, Moonlight 2502.16982)
MUON_ADAMW_EMBED_HEAD_PATTERNS = (
    # first layer of stream embed nets
    r"embed_engine\.embeds\.[^.]+\.(embed|layer)\.",
    # target coord embedding and first decoder adaln layer
    r"(^|\.)embed_target_coords\.",
    r"target_token_engines\..*\.embed_aux\.0\.",
    # diffusion noise embedding
    r"noise_embedder\.mlp\.0\.",
    # output heads
    r"(^|\.)pred_heads\.",
    # query tokens and position tables (not 2D thus adamw anyway)
    r"(^|\.)(q_cells|q_aux|pos_embed)$",
)


def _adamw_betas_eps(optimizer_cfg, kappa: float) -> dict:
    """
    DDP-scaled adamw betas/eps, shared by AdamW and Muon (muon falls back to adamw for
    non-2D parameters).

    https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    aiming for beta1=0.9 and beta2=0.95 following the MAE paper https://arxiv.org/pdf/2111.06377
    """
    # aiming for beta1 = 0.9 at one node, ie kappa=B=4
    beta1 = max(0.5, 1.0 - kappa * (1.0 - optimizer_cfg.adamw.beta1))
    # aiming for beta2 = 0.95 at one node, ie B=4
    beta2 = max(0.9, 1.0 - kappa * (1.0 - optimizer_cfg.adamw.beta2))
    eps = optimizer_cfg.adamw.get("eps", 2e-08) / np.sqrt(kappa)
    return {"betas": (beta1, beta2), "eps": eps}


def _scale_lr_cfg(lr_cfg, lr_max_override: float | None):
    """
    Rescale a learning_rate_scheduling config to a different peak lr, keeping the
    warmup/decay/cooldown timing and relative shape identical.
    """
    if lr_max_override is None:
        return lr_cfg
    scale = lr_max_override / lr_cfg.lr_max
    return OmegaConf.merge(
        lr_cfg,
        {
            "lr_start": lr_cfg.lr_start * scale,
            "lr_max": lr_max_override,
            "lr_final_decay": lr_cfg.lr_final_decay * scale,
            "lr_final": lr_cfg.lr_final * scale,
        },
    )


def _muon_adjust_lr_factor(shape, adjust_lr_fn: str) -> float:
    """
    Mirrors torch.optim.Muon's internal per-parameter lr-adjustment factor
    (torch/optim/_muon.py::_adjust_lr), so a representative effective lr can be logged.
    """
    a, b = shape[0], shape[1]
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2 * max(a, b) ** 0.5
    return max(1.0, a / b) ** 0.5


def _fused_num_parts(model: torch.nn.Module) -> dict[int, int]:
    """
    Some layers stack several weight matrices in one tensor (swiglu gate and up, adaln scale shift
    and gate). Returns for each such weight (key = param id) how many matrices are stacked.
    """
    # only needed for split_fused
    from weathergen.model.diffusion import SpatialAdaLN
    from weathergen.model.layers import MLP, LinearNormConditioning
    from weathergen.model.norms import AdaLayerNorm, AdaLayerNormLayer, AdaLNZero, SwiGLU

    num_parts = {}
    for module in model.modules():
        if isinstance(module, MLP):
            for layer, act in zip(module.layers[:-1], module.layers[1:], strict=True):
                if isinstance(layer, torch.nn.Linear) and isinstance(act, SwiGLU):
                    num_parts[id(layer.weight)] = 2
        elif isinstance(module, AdaLayerNorm):
            num_parts[id(module.embed_aux[-1].weight)] = 2
        elif isinstance(module, AdaLNZero):
            num_parts[id(module.embed_aux[-1].weight)] = 3
        elif isinstance(module, AdaLayerNormLayer):
            num_parts[id(module.adaLN_modulation[-1].weight)] = 3
        elif isinstance(module, LinearNormConditioning):
            num_parts[id(module.conditional_linear_layer.weight)] = 3
        elif isinstance(module, SpatialAdaLN):
            num_parts[id(module.proj[-1].weight)] = 3
    return num_parts


class MuonSplitFused(torch.optim.Muon):
    """
    torch.optim.Muon that orthogonalizes each stacked matrix of a fused weight on its own instead
    of whole tensor. Otherwise same update as torch 2.9 Muon.
    """

    def __init__(self, params, num_parts: dict[int, int], **kwargs):
        super().__init__(params, **kwargs)
        self.num_parts = num_parts
        for group in self.param_groups:
            for p in group["params"]:
                k = num_parts.get(id(p), 1)
                assert p.shape[0] % k == 0, f"fused param {tuple(p.shape)} not divisible by {k}"

    def _orthogonalize(self, p, update, group):
        k = self.num_parts.get(id(p), 1)
        ns_args = (group["ns_coefficients"], group["ns_steps"], group["eps"])
        if k == 1:
            return _zeropower_via_newtonschulz(update, *ns_args)
        # fsdp2 splits rows over ranks thus gather full weight before splitting into matrices.
        # Newton-schulz runs in bf16 therefore gather in bf16
        full = update.bfloat16()
        full = full.full_tensor() if isinstance(full, DTensor) else full
        out = torch.cat(
            [_zeropower_via_newtonschulz(part, *ns_args) for part in full.chunk(k, dim=0)], dim=0
        )
        if isinstance(update, DTensor):
            mesh = update.device_mesh
            out = DTensor.from_local(out, mesh, [Replicate()] * mesh.ndim, run_check=False)
            out = out.redistribute(mesh, update.placements)
        return out

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params, grads, bufs = [], [], []
            self._init_group(group, params, grads, bufs)
            lr = _to_scalar(group["lr"])
            momentum = group["momentum"]
            for p, grad, buf in zip(params, grads, bufs, strict=True):
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if group["nesterov"] else buf
                update = self._orthogonalize(p, update, group)
                # lr scale from shape of one block
                k = self.num_parts.get(id(p), 1)
                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], (p.shape[0] // k, p.shape[1]))
                p.mul_(1 - lr * group["weight_decay"])
                p.add_(update, alpha=-adjusted_lr)
        return loss


class AdamW:
    """
    Single torch.optim.AdamW optimizer over all of the model's parameters.
    """

    def __init__(self, model: torch.nn.Module, optimizer_cfg, shared_lr_cfg, kappa: float):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=shared_lr_cfg.lr_start,
            weight_decay=optimizer_cfg.weight_decay,
            fused=True,
            **_adamw_betas_eps(optimizer_cfg, kappa),
        )

        self.optimizers: list[torch.optim.Optimizer] = [optimizer]
        self.optimizer_names: list[str] = ["adamw"]
        self.lr_cfgs: list = [shared_lr_cfg]
        self.muon_effective_lr_factor: float | None = None


class Muon:
    """
    Muon optimizer for the model's 2D weight matrices (hidden layers), paired with a separate
    AdamW optimizer for all other (non-2D) parameters -- biases, norms, ... -- as recommended by
    https://kellerjordan.github.io/posts/muon/ (torch.optim.Muon also hard-requires exactly 2D
    tensors and raises ValueError otherwise, e.g. for a [1, 1, 2048] param).

    optimizer.muon keys added here (all off by default which keeps old split)
      adamw_embed_head      2D params matching MUON_ADAMW_EMBED_HEAD_PATTERNS go to adamw
      adamw_param_patterns  extra regex list (re.search on param name) of 2D params for adamw
      adamw_moved_lr_max    lr_max of 2D params moved to adamw by the two keys above, these get
                            own adamw. Null = they share adamw lr_max
      split_fused           orthogonalize fused swiglu / adaln weights per row block
    """

    def __init__(self, model: torch.nn.Module, optimizer_cfg, shared_lr_cfg, kappa: float):
        muon_cfg = optimizer_cfg.muon
        patterns = list(muon_cfg.get("adamw_param_patterns", None) or [])
        if muon_cfg.get("adamw_embed_head", False):
            patterns += MUON_ADAMW_EMBED_HEAD_PATTERNS
        regex = re.compile("|".join(f"(?:{p})" for p in patterns)) if patterns else None
        # with match_rms_adamw muon lr_max here keeps their step size as when in muon (Moonlight
        # uses one lr for both)
        moved_lr_max = muon_cfg.get("adamw_moved_lr_max", None)

        # 2D params to muon, rest to adamw. Patterns move some 2D params to adamw, without
        # patterns same split as before
        muon_params, adamw_params, moved_params, moved = [], [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim != 2:
                adamw_params.append(p)
            elif regex is not None and regex.search(name):
                (adamw_params if moved_lr_max is None else moved_params).append(p)
                moved.append((name, p.numel()))
            else:
                muon_params.append(p)

        split_fused = muon_cfg.get("split_fused", False)
        num_parts = {}
        if split_fused:
            muon_ids = {id(p) for p in muon_params}
            num_parts = {i: k for i, k in _fused_num_parts(model).items() if i in muon_ids}

        muon_lr_cfg = _scale_lr_cfg(shared_lr_cfg, muon_cfg.get("lr_max", None))
        moved_lr_cfg = _scale_lr_cfg(shared_lr_cfg, moved_lr_max)
        adjust_lr_fn = muon_cfg.get("adjust_lr_fn", None) or "original"
        # row block shape of fused weights
        block_shapes = [(p.shape[0] // num_parts.get(id(p), 1), p.shape[1]) for p in muon_params]
        self.muon_effective_lr_factor: float = float(
            np.median([_muon_adjust_lr_factor(s, adjust_lr_fn) for s in block_shapes])
        )

        if is_root():
            logger.info(
                f"Using muon optimizer: {len(muon_params)} params "
                f"({sum(p.numel() for p in muon_params) / 1e6:.2f}M) via muon, "
                f"{len(adamw_params)} params "
                f"({sum(p.numel() for p in adamw_params) / 1e6:.2f}M) via adamw, "
                f"muon lr_max={muon_lr_cfg.lr_max:.3g} "
                f"(adamw lr_max={shared_lr_cfg.lr_max:.3g}), "
                f"median {adjust_lr_fn} factor={self.muon_effective_lr_factor:.3g}"
            )
            if moved:
                logger.info(
                    f"muon: {len(moved)} 2D params ({sum(n for _, n in moved) / 1e6:.3f}M) "
                    f"moved to adamw by name patterns {patterns}, "
                    f"lr_max={moved_lr_cfg.lr_max:.3g}"
                    + (" (own adamw): " if moved_params else ": ")
                    + ", ".join(name for name, _ in moved)
                )
            if split_fused:
                logger.info(
                    f"muon: split_fused orthogonalizes {len(num_parts)} fused params "
                    f"({sum(p.numel() for p in muon_params if id(p) in num_parts) / 1e6:.2f}M) "
                    "per row block"
                )

        muon_kwargs = dict(
            lr=muon_lr_cfg.lr_start,
            weight_decay=optimizer_cfg.weight_decay,
            momentum=muon_cfg.get("momentum", 0.95),
            nesterov=muon_cfg.get("nesterov", True),
            ns_steps=muon_cfg.get("ns_steps", 5),
            eps=muon_cfg.get("eps", 1e-7),
            adjust_lr_fn=muon_cfg.get("adjust_lr_fn", None),
        )
        if split_fused:
            muon_optimizer = MuonSplitFused(muon_params, num_parts, **muon_kwargs)
        else:
            muon_optimizer = torch.optim.Muon(muon_params, **muon_kwargs)

        def adamw(params, lr_cfg):
            return torch.optim.AdamW(
                params,
                lr=lr_cfg.lr_start,
                weight_decay=optimizer_cfg.weight_decay,
                fused=True,
                **_adamw_betas_eps(optimizer_cfg, kappa),
            )

        self.optimizers: list[torch.optim.Optimizer] = [
            muon_optimizer,
            adamw(adamw_params, shared_lr_cfg),
        ]
        self.optimizer_names: list[str] = ["muon", "adamw"]
        self.lr_cfgs: list = [muon_lr_cfg, shared_lr_cfg]
        if moved_params:
            self.optimizers.append(adamw(moved_params, moved_lr_cfg))
            self.optimizer_names.append("adamw_moved")
            self.lr_cfgs.append(moved_lr_cfg)
