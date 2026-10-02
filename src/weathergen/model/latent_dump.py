# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Optional dump of latent tokens to disk, used to analyse latent space distributions.

The dump is switched on with the config option ``save_latents=True`` (e.g. through
``--options save_latents=True``) and is only active when the model is in eval mode.

Options
-------
save_latents : bool
    Switch the dump on. Default: False.
save_latents_dir : str
    Output directory. Default: ``<results>/<run_id>/latents``.
save_latents_dtype : str
    ``float32`` (default) or ``float16``.

Files
-----
encoder_sample<idx>_rank<rank>.pt
    Encoder output of one sample, shape (num_input_steps, num_tokens, dim). Index 0 along the
    first dimension is the most recent input step.
forecast_sample<idx>_step<step>_rank<rank>.pt
    Output of the deterministic forecast engine for one forecast step,
    shape (1, num_tokens, dim).
diffusion_sample<idx>_step<step>_rank<rank>.pt
    Final denoised state of the diffusion forecast engine for one forecast step,
    shape (num_members, num_tokens, dim).

Each file holds a dict with the key ``tokens`` and metadata. The metadata is also appended, one
line per file, to ``index_rank<rank>.jsonl``. ``num_tokens`` includes register and class tokens,
which come first; their number is stored as ``num_aux_tokens``.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from weathergen.common.config import Config, get_path_run

logger = logging.getLogger(__name__)

KIND_ENCODER = "encoder"
KIND_FORECAST = "forecast"
KIND_DIFFUSION = "diffusion"


class LatentDumper:
    """Writes latent tokens to disk when ``save_latents`` is set in the config."""

    def __init__(self, cf: Config) -> None:
        self.cf = cf
        self.enabled = bool(cf.get("save_latents", False))
        self._dir: Path | None = None

        dtype = cf.get("save_latents_dtype", "float32")
        assert dtype in ("float32", "float16"), (
            f"save_latents_dtype must be 'float32' or 'float16', got '{dtype}'."
        )
        self.dtype = torch.float16 if dtype == "float16" else torch.float32

    def _get_dir(self) -> Path:
        """Output directory; resolved on first use so that the run id is final."""
        if self._dir is None:
            path = self.cf.get("save_latents_dir", None)
            self._dir = Path(path) if path else get_path_run(self.cf) / "latents"
            self._dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing latent tokens to {self._dir}.")
        return self._dir

    @staticmethod
    def _sample_info(sample) -> tuple[int, list[str | None]]:
        """
        Dataset index of the sample and, per input step (most recent first), the earliest
        time stamp of the source data over all streams.
        """
        sample_idx, source_times = -1, []
        try:
            streams_data = list(sample.streams_data.values())
            sample_idx = int(streams_data[0].sample_idx)
            num_steps = max(len(sd.source_raw) for sd in streams_data)
            for i_step in range(num_steps):
                t_min = None
                for sd in streams_data:
                    if i_step >= len(sd.source_raw) or sd.source_raw[i_step] is None:
                        continue
                    datetimes = np.asarray(sd.source_raw[i_step].datetimes)
                    if datetimes.size > 0:
                        t = datetimes.min()
                        t_min = t if t_min is None or t < t_min else t_min
                source_times.append(None if t_min is None else str(t_min.astype("datetime64[s]")))
        except Exception as e:  # metadata is a convenience, never fail the forward pass on it
            logger.warning(f"Could not collect sample metadata for latent dump: {e!r}")
        return sample_idx, source_times

    def _write(self, kind: str, tokens: torch.Tensor, sample, step: int | None) -> None:
        sample_idx, source_times = self._sample_info(sample)
        rank = int(self.cf.get("rank", 0))

        name = f"{kind}_sample{sample_idx:06d}"
        if step is not None:
            name += f"_step{step:03d}"
        name += f"_rank{rank:04d}.pt"

        meta = {
            "kind": kind,
            "file": name,
            "sample_idx": sample_idx,
            "forecast_step": step,
            "source_times": source_times,
            "shape": list(tokens.shape),
            "dtype": str(self.dtype).replace("torch.", ""),
            "num_aux_tokens": int(self.cf.num_register_tokens + self.cf.num_class_tokens),
            "run_id": str(self.cf.get("general", {}).get("run_id", "")),
            "from_run_id": str(self.cf.get("from_run_id", "")),
        }

        out_dir = self._get_dir()
        tokens_cpu = tokens.detach().to(device="cpu", dtype=self.dtype)
        torch.save({"tokens": tokens_cpu, **meta}, out_dir / name)
        with open(out_dir / f"index_rank{rank:04d}.jsonl", "a") as f:
            f.write(json.dumps(meta) + "\n")

    def dump_encoder(self, tokens: torch.Tensor, source_samples) -> None:
        """
        Write the encoder output.

        Args:
            tokens : shape (batch_size, num_input_steps, num_tokens, dim)
            source_samples : source samples of the batch
        """
        for i_b, sample in enumerate(source_samples.samples):
            self._write(KIND_ENCODER, tokens[i_b], sample, None)

    def dump_forecast(self, kind: str, tokens: torch.Tensor, source_samples, step: int) -> None:
        """
        Write the output of the forecast engine for one forecast step.

        Args:
            kind : KIND_FORECAST for the deterministic engine, KIND_DIFFUSION for the final
                denoised state of the diffusion engine
            tokens : shape (batch_size, num_tokens, dim); for diffusion the leading dimension
                is the ensemble member and the batch holds a single sample
            source_samples : source samples of the batch
            step : global forecast step
        """
        samples = source_samples.samples
        if kind == KIND_DIFFUSION or tokens.shape[0] != len(samples):
            self._write(kind, tokens, samples[0], step)
        else:
            for i_b, sample in enumerate(samples):
                self._write(kind, tokens[i_b : i_b + 1], sample, step)
