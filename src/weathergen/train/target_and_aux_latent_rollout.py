# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import copy

import torch

from weathergen.datasets.batch import BatchSamples
from weathergen.model.model import ModelParams
from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase, TargetAuxOutput

# per-input-step members of StreamData
_STEP_ATTRS = (
    "source_tokens_cells",
    "source_tokens_lens",
    "source_raw",
    "source_idxs_embed",
    "source_idxs_embed_pe",
    "source_is_spoof",
    "latent_rollout_is_spoof",
)


def _select_input_step(batch: BatchSamples, step: int) -> BatchSamples:
    """View of the batch with only one input step (no copies)"""
    view = copy.copy(batch)
    view.samples = []
    for sample in batch.samples:
        sample_step = copy.copy(sample)
        sample_step.streams_data = {}
        for name, stream_data in sample.streams_data.items():
            sd_step = copy.copy(stream_data)
            sd_step.input_steps = 1
            for attr in _STEP_ATTRS:
                setattr(sd_step, attr, [getattr(stream_data, attr)[step]])
            sample_step.streams_data[name] = sd_step
        view.samples.append(sample_step)
    view.tokens_lens = batch.tokens_lens[step : step + 1]
    return view


class LatentRolloutTargets(TargetAndAuxModuleBase):
    """
    Latent targets for LossLatentRollout: encoder of the model applied to the source data of
    each forecast step, without gradients and in eval mode.
    """

    def __init__(self, cf, model, **kwargs):
        self.cf = cf

    def compute(
        self,
        istep: int,
        batch: BatchSamples,
        model_params: ModelParams,
        model: torch.nn.Module,
        *args,
        **kwargs,
    ) -> TargetAuxOutput:
        output_idxs = batch.get_output_idxs()
        assert batch.get_num_steps() == len(output_idxs), (
            f"Target samples hold {batch.get_num_steps()} input steps but there are "
            f"{len(output_idxs)} forecast steps."
        )

        # unwrap DDP (FSDP2 keeps module attributes)
        encoder = getattr(model, "module", model).encoder

        # Encode in eval mode and one step at a time (all steps at once crashes flash attention
        # for large streams), then restore the train/eval mode of each module. The input of each
        # step goes to the GPU only for its encoding (the trainer keeps it on the CPU).
        device = next(encoder.parameters()).device
        modes = [(module, module.training) for module in encoder.modules()]
        encoder.eval()
        with torch.no_grad():
            tokens = []
            for i in range(len(output_idxs)):
                batch_step = _select_input_step(batch, i).to_device(
                    device,
                    target_steps=[],
                    include_target_coords=False,
                    include_target_tokens=False,
                )
                # bf16 halves the memory of the targets kept until the backward pass
                tokens.append(encoder(model_params, batch_step)[0].to(torch.bfloat16))
                del batch_step
        for module, mode in modes:
            module.training = mode

        targets = TargetAuxOutput(batch.get_output_len(), output_idxs)
        is_spoof = {}
        for i, step in enumerate(output_idxs):
            targets.add_latent_target(step, "latent_rollout", tokens[i])
            # skip step only if all input streams are spoofed (missing streams are spoofed as
            # for the model input)
            streams_with_input = [
                stream_data
                for sample in batch.get_samples()
                for stream_data in sample.streams_data.values()
                if stream_data.source_tokens_cells[i] is not None
            ]
            is_spoof[step] = all(sd.latent_rollout_is_spoof[i] for sd in streams_with_input)
        targets.aux_outputs = {"is_spoof": is_spoof}

        return targets
