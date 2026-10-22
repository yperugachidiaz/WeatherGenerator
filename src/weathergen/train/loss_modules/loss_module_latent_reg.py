# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ported unchanged from S. Hickman's branch develop-ssl-diffusion-v1-reg-latent

import logging
from collections import deque

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

logger = logging.getLogger(__name__)


class LossLatentRegularization(LossModuleBase):
    """
    Regularization of the encoder output (HEALPix cell tokens, aux tokens removed) towards a
    well-behaved distribution, e.g. against outlier channels such as channel 551 in wduc34ev and
    to make the latent easier to model with diffusion (EDM assumes data close to N(0, I)).

    Independent of the training mode: the encoder output is taken from
    preds.latent[0]["encoder_tokens"], which the model provides for masking (physical forecast)
    and student-teacher training alike.

    Regularizers, selected via loss_fcts (one or both):
        kurtosis: {weight}
            Excess kurtosis of all values above that of a normal, per sample (same definition as
            the kurtosis penalty used in kr9snxa0).
        sigreg: {weight, num_projections, num_knots, num_cells, queue_steps, queue_cells,
                 min_queue_steps}
            SIGReg (Balestriero & LeCun, 2025) with a queue of past steps, see sigreg_cross.

    Diagnostics of the latent are always logged (no gradient).
    """

    def __init__(self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **losses):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossLatentRegularization"

        valid = {"kurtosis", "sigreg"}
        assert set(losses).issubset(valid), f"Unknown latent regularizers {set(losses) - valid}"

        self.num_aux_tokens = int(cf.get("num_register_tokens", 0) or 0) + int(
            cf.get("num_class_tokens", 0) or 0
        )

        kurt_cfg = losses.get("kurtosis")
        self.kurtosis_weight = kurt_cfg.get("weight", 1.0) if kurt_cfg is not None else 0.0

        sig_cfg = losses.get("sigreg")
        self.sigreg_weight = sig_cfg.get("weight", 1.0) if sig_cfg is not None else 0.0
        if sig_cfg is not None:
            self.sigreg_num_projections = sig_cfg.get("num_projections", 256)
            self.sigreg_num_cells = sig_cfg.get("num_cells", 2048)
            self.sigreg_queue_cells = sig_cfg.get("queue_cells", 256)
            self.sigreg_min_queue_steps = sig_cfg.get("min_queue_steps", 8)
            # past (detached) cell tokens, one entry per step
            self.sigreg_queue = deque(maxlen=sig_cfg.get("queue_steps", 64))
            # frequencies and Gaussian-windowed trapezoidal quadrature weights on [0, 3]
            num_knots = sig_cfg.get("num_knots", 17)
            t = torch.linspace(0, 3, num_knots, device=device)
            w = torch.full((num_knots,), 2 * 3.0 / (num_knots - 1), device=device)
            w[0] = w[-1] = 3.0 / (num_knots - 1)
            self.sigreg_phi = torch.exp(-t.square() / 2.0)
            self.sigreg_t = t
            self.sigreg_quad_weights = w * self.sigreg_phi

    def compute_loss(self, preds, targets, metadata, **kwargs) -> LossValues:
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        losses_all: dict[str, float] = {}

        z = preds.latent[0].get("encoder_tokens", None)
        if z is None:
            logger.warning("LossLatentRegularization: no encoder_tokens in model output.")
            return LossValues(loss=loss, losses_all=losses_all, stddev_all={})

        # [samples, cells, channels] without aux (register/class) tokens
        z = z.reshape(-1, z.shape[-2], z.shape[-1])[:, self.num_aux_tokens :].float()

        losses_all.update(latent_diagnostics(z))

        if self.kurtosis_weight > 0.0:
            reg = kurtosis_penalty(z)
            loss = loss + self.kurtosis_weight * reg
            losses_all["kurtosis"] = reg.item()

        if self.sigreg_weight > 0.0:
            z_flat = z.reshape(-1, z.shape[-1])
            idx = torch.randperm(z_flat.shape[0], device=z_flat.device)
            z_now = z_flat[idx[: self.sigreg_num_cells]]

            if len(self.sigreg_queue) >= self.sigreg_min_queue_steps:
                reg = self.sigreg_cross(z_now, list(self.sigreg_queue))
                loss = loss + self.sigreg_weight * reg
                losses_all["sigreg"] = reg.item()

            # push after use, so the current step is never compared against itself
            z_push = z_flat[idx[: self.sigreg_queue_cells]].detach().to(torch.bfloat16)
            self.sigreg_queue.append(z_push)

        return LossValues(loss=loss, losses_all=losses_all, stddev_all={})

    def sigreg_cross(self, z_now: torch.Tensor, queue: list[torch.Tensor]) -> torch.Tensor:
        """
        SIGReg with a queue of past steps instead of a single batch.

        Plain SIGReg compares the empirical characteristic function (ECF) c_hat of one batch with
        the Gaussian CF phi: |c_hat - phi|^2. Its expectation is |c - phi|^2 + Var(c_hat). With
        one time step per rank all cells of a batch come from the same snapshot, so Var(c_hat)
        is dominated by differences between snapshots and its gradient pushes all snapshots to
        look alike, i.e. it suppresses signal that varies in time.

        Here the current ECF is paired with the ECF of past steps (other, independent times):
        Re[(c_now - phi) * conj(c_queue - phi)]. Its expectation is |c - phi|^2 (up to the
        staleness of the queue), without the variance term. Gradients only flow through c_now;
        the loss is linear in c_now, so averaging gradients over DDP ranks is equivalent to
        using the ECF over all ranks.

        Scaling follows the plain statistic (quadrature over frequencies, times the number of
        current cells, mean over projections); the factor 2 makes the gradient equal to that of
        the plain statistic's |c - phi|^2 term, so weights are comparable to plain SIGReg.
        Projections are drawn fresh every step; the queue stores tokens, not ECFs.
        """
        n, d = z_now.shape
        a = torch.randn(d, self.sigreg_num_projections, device=z_now.device)
        a = a / a.norm(p=2, dim=0, keepdim=True)
        t, phi = self.sigreg_t, self.sigreg_phi

        x_now = (z_now @ a).unsqueeze(-1) * t  # [cells, projections, knots]
        cos_now, sin_now = x_now.cos().mean(0), x_now.sin().mean(0)

        with torch.no_grad():
            cos_q = torch.zeros_like(cos_now)
            sin_q = torch.zeros_like(sin_now)
            num_q = 0
            for z_q in queue:
                x_q = (z_q.float() @ a).unsqueeze(-1) * t
                cos_q += x_q.cos().sum(0)
                sin_q += x_q.sin().sum(0)
                num_q += z_q.shape[0]
            cos_q /= num_q
            sin_q /= num_q

        cross = (cos_now - phi) * (cos_q - phi) + sin_now * sin_q
        return 2.0 * n * (cross @ self.sigreg_quad_weights).mean()


def kurtosis_penalty(z: torch.Tensor) -> torch.Tensor:
    """
    Penalty on heavy tails of the latent: relu of the excess kurtosis of all values (over cell
    tokens and channels) per sample, averaged over samples. Zero for a normal (or lighter-tailed)
    distribution, large when single values or channels are far out.

    Args:
        z: [samples, cells, channels], aux tokens removed.
    """
    zc = z - z.mean(dim=(1, 2), keepdim=True)
    m2 = zc.pow(2).mean(dim=(1, 2))
    m4 = zc.pow(4).mean(dim=(1, 2))
    excess = m4 / m2.pow(2).clamp_min(1e-12) - 3.0
    return F.relu(excess).mean()


@torch.no_grad()
def latent_diagnostics(z: torch.Tensor, sample_size: int = 16384) -> dict[str, float]:
    """
    Diagnostics of the latent: excess kurtosis of all values (as in kurtosis_penalty), the
    largest per-token share of a single channel, the largest absolute channel mean, and the
    ratio of the largest to the median channel standard deviation. (No channel index: values
    are averaged over DDP ranks for logging.)

    Args:
        z: [samples, cells, channels], aux tokens removed.
    """
    zc = z - z.mean(dim=(1, 2), keepdim=True)
    excess = zc.pow(4).mean(dim=(1, 2)) / zc.pow(2).mean(dim=(1, 2)).pow(2).clamp_min(1e-12) - 3.0

    z = z.reshape(-1, z.shape[-1])
    if z.shape[0] > sample_size:
        z = z[torch.randperm(z.shape[0], device=z.device)[:sample_size]]
    sq = z.pow(2)
    share = (sq / sq.sum(dim=-1, keepdim=True).clamp_min(1e-12)).mean(dim=0)
    channel_std = z.std(dim=0)
    return {
        "excess_kurtosis": excess.mean().item(),
        "max_channel_share": share.max().item(),
        "max_abs_channel_mean": z.mean(dim=0).abs().max().item(),
        "channel_std_max_over_median": (
            channel_std.max() / channel_std.median().clamp_min(1e-12)
        ).item(),
    }
