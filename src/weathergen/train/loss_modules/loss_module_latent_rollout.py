# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import torch
from omegaconf import DictConfig
from torch import Tensor

import weathergen.train.loss_modules.loss_functions as loss_fns
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

_logger = logging.getLogger(__name__)


class LossLatentRollout(LossModuleBase):
    """
    Latent loss per forecast step: latent after each step vs. encoder latent of that step.
    Also logs norm ratios and token spread per step to check drift and collapse.
    """

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts: dict,
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossLatentRollout"

        self.loss_fcts = [
            [
                getattr(loss_fns, name),
                params.get("weight", 1.0),
                name,
            ]
            for name, params in loss_fcts.items()
        ]

        # register and class tokens first, then healpix cell tokens
        self.num_aux_tokens = self.cf.num_register_tokens + self.cf.num_class_tokens
        self.tokens = self.cf.get("latent_rollout_loss_tokens", "all")
        assert self.tokens in ("all", "cells", "aux"), (
            f"latent_rollout_loss_tokens must be 'all', 'cells' or 'aux', got {self.tokens!r}."
        )
        assert self.tokens != "aux" or self.num_aux_tokens > 0, (
            "latent_rollout_loss_tokens='aux' requires register or class tokens."
        )

    def _select_tokens(self, tokens: Tensor, which: str) -> Tensor:
        if which == "cells":
            return tokens[:, self.num_aux_tokens :]
        if which == "aux":
            return tokens[:, : self.num_aux_tokens]
        return tokens

    @staticmethod
    def _loss_per_loss_function(loss_fct, target: Tensor, pred: Tensor) -> Tensor:
        # loss functions expect target (points, channels) and pred (ens, points, channels)
        target = target.flatten(0, -2)
        pred = pred.flatten(0, -2).unsqueeze(0)
        loss, _ = loss_fct(target=target, pred=pred)
        return loss

    @staticmethod
    def _norm_ratio(target: Tensor, pred: Tensor) -> Tensor:
        return torch.linalg.vector_norm(pred) / torch.linalg.vector_norm(target)

    def compute_loss(self, preds, targets, **kwargs) -> LossValues:
        assert len(preds.latent) == len(targets.latent), (
            "LossLatentRollout needs the latent state of every forecast step, "
            "set compute_full_validation_loss=True."
        )
        is_spoof = targets.aux_outputs.get("is_spoof", {})
        nan = torch.tensor(torch.nan, device=self.device)

        losses_all: dict[str, Tensor] = {}
        sums = {
            f"{self.name}.{name}{suffix}": []
            for _, _, name in self.loss_fcts
            for suffix in ("", ".cells") + ((".aux",) if self.num_aux_tokens > 0 else ())
        }
        # collapse check: spread of cell tokens across cells
        sums[f"{self.name}.token_std_target"] = []
        sums[f"{self.name}.token_std_pred"] = []

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        ctr_steps = 0
        for step, (preds_cur, targets_cur) in enumerate(
            zip(preds.latent, targets.latent, strict=True)
        ):
            if "latent_state" not in preds_cur or "latent_rollout" not in targets_cur:
                continue

            pred = preds_cur["latent_state"].z_pre_norm.float()
            target = targets_cur["latent_rollout"].float()
            spoofed = is_spoof.get(step, False)

            loss_step = torch.tensor(0.0, device=self.device, requires_grad=True)
            for loss_fct, loss_fct_weight, loss_fct_name in self.loss_fcts:
                key = f"{self.name}.{loss_fct_name}"
                loss_lfct = self._loss_per_loss_function(
                    loss_fct,
                    self._select_tokens(target, self.tokens),
                    self._select_tokens(pred, self.tokens),
                )
                loss_step = loss_step + loss_fct_weight * loss_lfct

                losses_all[f"{key}.step{step}"] = nan if spoofed else loss_lfct.detach()
                if not spoofed:
                    sums[key].append(loss_lfct.detach())

                # loss per token group for logging
                with torch.no_grad():
                    for which in ("cells", "aux") if self.num_aux_tokens > 0 else ("cells",):
                        if not spoofed:
                            sums[f"{key}.{which}"].append(
                                self._loss_per_loss_function(
                                    loss_fct,
                                    self._select_tokens(target, which),
                                    self._select_tokens(pred, which),
                                )
                            )

            with torch.no_grad():
                for which in ("cells", "aux") if self.num_aux_tokens > 0 else ("cells",):
                    ratio = self._norm_ratio(
                        self._select_tokens(target, which), self._select_tokens(pred, which)
                    )
                    losses_all[f"{self.name}.norm_ratio_{which}.step{step}"] = (
                        nan if spoofed else ratio
                    )
                if not spoofed:
                    for name, tokens in (("target", target), ("pred", pred)):
                        sums[f"{self.name}.token_std_{name}"].append(
                            self._select_tokens(tokens, "cells").std(dim=1).mean()
                        )

            # spoofed steps do not count (keep graph connected)
            if spoofed:
                loss = loss + 0.0 * loss_step
            else:
                loss = loss + loss_step
                ctr_steps += 1

        loss = loss / (ctr_steps if ctr_steps > 0 else 1.0)

        for key, values in sums.items():
            losses_all[key] = torch.stack(values).mean() if values else nan

        return LossValues(
            loss=loss,
            losses_all=losses_all,
            stddev_all={"latent": nan},
        )
