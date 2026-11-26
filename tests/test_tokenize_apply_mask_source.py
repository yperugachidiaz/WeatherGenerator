"""Tests for the faster source tokenization: padding rows of the local coords once per cell and
stacked padded tokens. Values must be bitwise equal to the per row computation."""

import numpy as np
import pytest
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.tokenizer_utils import (
    _token_mask_info,
    encode_times_source,
    get_source_coords_local,
    get_source_coords_local_padded,
    tokenize_apply_mask_source,
    tokenize_space,
)
from weathergen.datasets.utils import vecs_to_rots


def _random_rdata(num_points, num_channels, seed, nan_rows=0):
    rng = np.random.default_rng(seed)
    lats = rng.uniform(-90.0, 90.0, num_points).astype(np.float32)
    lons = rng.uniform(-180.0, 180.0, num_points).astype(np.float32)
    coords = torch.from_numpy(np.stack([lats, lons], axis=1))
    geoinfos = torch.from_numpy(rng.standard_normal((num_points, 3)).astype(np.float32))
    data = torch.from_numpy(rng.standard_normal((num_points, num_channels)).astype(np.float32))
    if nan_rows > 0:
        data[rng.choice(num_points, nan_rows, replace=False), 0] = torch.nan
    t0 = np.datetime64("2020-01-01T00:00")
    datetimes = t0 + rng.integers(0, 6 * 3600, num_points).astype("timedelta64[s]")
    return IOReaderData(coords, geoinfos, data, datetimes)


def _rots(num_cells, seed):
    rng = np.random.default_rng(seed)
    vecs = torch.from_numpy(rng.standard_normal((num_cells, 3)))
    vecs = vecs / torch.linalg.norm(vecs, dim=1, keepdim=True)
    return vecs_to_rots(vecs).to(torch.float32)


def _bits(x):
    return x.contiguous().view(torch.int32)


@pytest.mark.parametrize(
    "num_points,hl,token_size,keep", [(500, 2, 8, 1.0), (3000, 2, 16, 0.5), (40, 3, 32, 0.7)]
)
def test_coords_local_padded_matches_per_row(num_points, hl, token_size, keep):
    """padding rows once per cell gives the same bits as get_source_coords_local on all rows"""
    rdata = _random_rdata(num_points, 2, seed=num_points)
    num_cells = 12 * 4**hl
    idxs_cells, idxs_cells_lens = tokenize_space(rdata, token_size, hl, pad_tokens=True)
    num_tokens = sum(len(c) for c in idxs_cells_lens)
    mask = np.random.default_rng(1).uniform(size=num_tokens) < keep
    mask[0] = True
    _, _, _, points_per_cell, idxs_data = _token_mask_info(idxs_cells, idxs_cells_lens, mask)
    counts = torch.from_numpy(points_per_cell.astype(np.int32))
    coords_padded = torch.cat([torch.zeros_like(rdata.coords[0]).unsqueeze(0), rdata.coords])
    rots = _rots(num_cells, seed=hl)

    ref = get_source_coords_local(coords_padded[idxs_data], rots, counts)
    out = get_source_coords_local_padded(coords_padded, idxs_data, rots, counts)

    assert (idxs_data == 0).any(), "test needs padding rows"
    assert out.dtype == ref.dtype and out.shape == ref.shape
    assert torch.equal(_bits(out), _bits(ref))


def test_tokenize_apply_mask_source_columns():
    """stacked padded tokens hold stream id, times, local coords, geoinfos and data per row"""
    hl, token_size, stream_id = 2, 8, 7
    rdata = _random_rdata(800, 3, seed=3, nan_rows=5)
    num_cells = 12 * 4**hl
    idxs_cells, idxs_cells_lens = tokenize_space(rdata, token_size, hl, pad_tokens=True)
    num_tokens = sum(len(c) for c in idxs_cells_lens)
    mask = np.random.default_rng(2).uniform(size=num_tokens) < 0.6
    mask[0] = True
    rots = _rots(num_cells, seed=5)
    time_win = (np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-01T06:00"))

    tokens_cells, tokens_per_cell = tokenize_apply_mask_source(
        idxs_cells,
        idxs_cells_lens,
        mask,
        None,
        stream_id,
        rdata,
        time_win,
        rots,
        encode_times_source,
    )
    _, _, ref_tokens_per_cell, points_per_cell, idxs_data = _token_mask_info(
        idxs_cells, idxs_cells_lens, mask
    )

    # padded tokens come back stacked: num kept tokens x token_size x channels
    assert torch.is_tensor(tokens_cells)
    assert tokens_cells.shape[:2] == (int(mask.sum()), token_size)
    assert torch.equal(tokens_per_cell, torch.from_numpy(ref_tokens_per_cell))

    # reference per row, padding row 0 has zeros apart from the stream id
    def padded(x):
        return torch.cat([torch.zeros_like(x[0]).unsqueeze(0), x])

    times = padded(encode_times_source(rdata.datetimes, time_win))[idxs_data]
    coords_padded = padded(rdata.coords)
    counts = torch.from_numpy(points_per_cell.astype(np.int32))
    coords_local = get_source_coords_local(coords_padded[idxs_data], rots, counts)
    ref = torch.cat(
        (
            torch.full([len(idxs_data), 1], stream_id, dtype=torch.float32),
            times,
            coords_local,
            padded(rdata.geoinfos)[idxs_data],
            padded(rdata.data)[idxs_data],
        ),
        1,
    )
    flat = tokens_cells.reshape(-1, tokens_cells.shape[-1])
    # NaN in the data stay NaN here, StreamData.add_source replaces them later
    assert torch.equal(torch.isnan(flat), torch.isnan(ref))
    assert torch.equal(_bits(torch.nan_to_num(flat)), _bits(torch.nan_to_num(ref)))
