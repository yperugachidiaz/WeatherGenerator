# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CPU tests of DataLoader arguments: data_loading.prefetch_factor reaches the loader."""

import torch
from omegaconf import OmegaConf

from weathergen.train.trainer import loader_kwargs


class _Steps(torch.utils.data.IterableDataset):
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id
        for i in range(4):
            yield torch.full((2,), float(10 * wid + i))


def test_default_prefetch_is_torch_default():
    # no prefetch_factor in config thus 2 as before
    kw = loader_kwargs(OmegaConf.create({"num_workers": 3}), 3)
    assert kw["prefetch_factor"] == 2
    loader = torch.utils.data.DataLoader(_Steps(), **kw)
    assert loader.prefetch_factor == 2


def test_prefetch_from_config():
    kw = loader_kwargs(OmegaConf.create({"num_workers": 3, "prefetch_factor": 1}), 3)
    loader = torch.utils.data.DataLoader(_Steps(), **kw)
    assert loader.prefetch_factor == 1
    assert loader.num_workers == 3


def test_no_workers_no_prefetch():
    # torch raises for prefetch_factor with num_workers 0
    kw = loader_kwargs(OmegaConf.create({"num_workers": 0, "prefetch_factor": 1}), 0)
    assert "prefetch_factor" not in kw
    out = [b.tolist() for b in torch.utils.data.DataLoader(_Steps(), **kw)]
    assert out == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]


def test_prefetch_keeps_batch_order():
    # prefetch only changes how many batches wait in /dev/shm, not which batches come in which order
    cfg1 = OmegaConf.create({"num_workers": 2, "prefetch_factor": 1})
    cfg2 = OmegaConf.create({"num_workers": 2, "prefetch_factor": 2})
    out1 = [b[0].item() for b in torch.utils.data.DataLoader(_Steps(), **loader_kwargs(cfg1, 2))]
    out2 = [b[0].item() for b in torch.utils.data.DataLoader(_Steps(), **loader_kwargs(cfg2, 2))]
    assert out1 == out2
