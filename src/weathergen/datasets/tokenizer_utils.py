from itertools import chain, compress

import numpy as np
import pandas as pd
import torch
from astropy_healpix.healpy import ang2pix
from torch import Tensor

from weathergen.common.io import IOReaderData
from weathergen.datasets.utils import (
    r3tos2,
    s2tor3,
    vecs_to_rots,
)

# on some clusters our numpy version is pinned to be 1.x.x where the np.argsort does not
# the stable=True argument
numpy_argsort_args = {"stable": True} if int(np.__version__.split(".")[0]) >= 2 else {}


def theta_phi_to_standard_coords(coords):
    thetas = ((90.0 - coords[:, 0]) / 180.0) * np.pi
    phis = ((coords[:, 1] + 180.0) / 360.0) * 2.0 * np.pi

    return thetas, phis


def precompute_cell_ids(
    input_data: list,
    healpix_level: int,
) -> list[np.typing.NDArray | None]:
    """Precompute HEALPix cell IDs for each rdata's coordinates.

    Returns a list parallel to ``input_data``, with ``None`` for empty entries.
    """
    nside = 2**healpix_level
    cell_ids_list = []
    for rdata in input_data:
        if rdata.is_empty():
            cell_ids_list.append(None)
        else:
            coords_np = (
                rdata.coords.numpy() if isinstance(rdata.coords, torch.Tensor) else rdata.coords
            )
            thetas, phis = theta_phi_to_standard_coords(coords_np)
            cell_ids_list.append(ang2pix(nside, thetas, phis, nest=True))
    return cell_ids_list


def encode_times_source(times, time_win) -> torch.tensor:
    """Encode times in the format used for source

    Return:
        len(times) x 5
    """
    # assemble tensor as fed to the network, combining geoinfo and data
    fp32 = torch.float32
    dt = pd.to_datetime(times)
    dt_win = pd.to_datetime(time_win)
    dt_delta = dt - dt_win[0]
    year = np.atleast_1d(dt.year)
    dayofyear = np.atleast_1d(dt.dayofyear)
    minutes = np.atleast_1d(dt.hour * 60 + dt.minute)
    delta_seconds = np.atleast_1d(dt_delta.seconds)
    time_tensor = torch.cat(
        (
            torch.tensor(year, dtype=fp32).unsqueeze(1),
            torch.tensor(dayofyear, dtype=fp32).unsqueeze(1),
            torch.tensor(minutes, dtype=fp32).unsqueeze(1),
            torch.tensor(delta_seconds, dtype=fp32).unsqueeze(1),
            torch.tensor(delta_seconds, dtype=fp32).unsqueeze(1),
        ),
        1,
    )

    # normalize
    time_tensor[..., 0] /= 2100.0
    time_tensor[..., 1] = time_tensor[..., 1] / 365.0
    time_tensor[..., 2] = time_tensor[..., 2] / 1440.0
    time_tensor[..., 3] = np.sin(time_tensor[..., 3] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 4] = np.cos(time_tensor[..., 4] / (12.0 * 3600.0) * 2.0 * np.pi)

    return time_tensor


def encode_times_target(times, time_win) -> torch.tensor:
    """Encode times in the format used for target (relative time in window)

    Return:
        len(times) x 5
    """
    dt = pd.to_datetime(times)
    dt_win = pd.to_datetime(time_win)
    # for target only provide local time
    dt_delta = torch.tensor(np.atleast_1d((dt - dt_win[0]).seconds), dtype=torch.float32).unsqueeze(
        1
    )
    time_tensor = torch.cat(
        (
            dt_delta,
            dt_delta,
            dt_delta,
            dt_delta,
            dt_delta,
        ),
        1,
    )

    # normalize
    time_tensor[..., 0] = np.sin(time_tensor[..., 0] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 1] = np.cos(time_tensor[..., 1] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 2] = np.sin(time_tensor[..., 2] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 3] = np.cos(time_tensor[..., 3] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 4] = np.sin(time_tensor[..., 4] / (12.0 * 3600.0) * 2.0 * np.pi)

    # We add + 0.5 as for datasets with regular time steps we otherwise very often get 0 as the
    # first time and to prevent too many zeros in the input
    return time_tensor + 0.5


def hpy_cell_splits(coords: torch.tensor, hl: int):
    """Compute nest HEALPix cell ids and a packed point order for level ``hl``.

    Points grouped by cell and ordered by theta within each cell, as
    ``np.lexsort((theta, cell_id))``. Done as stable sort by theta then stable
    sort by cell: same order but faster (radix sort of cell ids).

    Returns
      idxs_ord : indices into thetas,phis,posr3, grouped by ascending healpix cell and
        ordered by theta within each cell
      counts : number of points in each of the 12 * 4**hl cells, so that
        np.split(idxs_ord, np.cumsum(counts)) recovers the per cell indices
    """
    thetas, phis = theta_phi_to_standard_coords(coords)
    hpy_idxs = ang2pix(2**hl, thetas, phis, nest=True)

    thetas_np = thetas.numpy() if isinstance(thetas, torch.Tensor) else np.asarray(thetas)
    num_cells = 12 * 4**hl
    idxs_theta = np.argsort(thetas_np, kind="stable")
    cells_sorted = hpy_idxs[idxs_theta]
    # numpy uses a radix sort for stable sorts of 16 bit ints
    if num_cells <= 2**16:
        cells_sorted = cells_sorted.astype(np.uint16)
    idxs_ord = idxs_theta[np.argsort(cells_sorted, kind="stable")]
    counts = np.bincount(hpy_idxs, minlength=num_cells)

    return idxs_ord, counts


def hpy_splits(
    coords: torch.Tensor, hl: int, token_size: int, pad_tokens: bool, offset_step: int = 0
) -> tuple[list[list[torch.Tensor]], list[list[int]]]:
    """Group points into nest HEALPix cells and split each cell into tokens.

    Cells are the ``12 * 4**hl`` nest cells at level ``hl``. Points in a cell
    are ordered by increasing theta (north first). Occupied cells are cut into
    chunks of ``token_size``; empty cells stay ``[]``.

    If ``pad_tokens``, stored indices are ``index + 1 + offset_step`` (``0`` is
    the padding row the caller prepends) and every token has length
    ``token_size``. Otherwise indices are ``index + offset_step`` and the last
    token of a cell may be shorter. ``offset_step`` shifts later time steps onto
    the full stream.

    Returns:
        idxs_ord: per cell, a list of index tensors (one per token)
        idxs_ord_lens: per cell, the length of each token
    """

    # TODO: split by hierarchically traversing healpix scheme
    idxs_ord_flat, counts = hpy_cell_splits(coords, hl)

    num_cells = len(counts)
    if len(idxs_ord_flat) == 0:
        return [[] for _ in range(num_cells)], [[] for _ in range(num_cells)]

    offset = (1 if pad_tokens else 0) + offset_step

    tokens_per_cell = -(-counts // token_size)
    slots_per_cell = tokens_per_cell * token_size if pad_tokens else counts

    # Scatter the sorted indices into one flat buffer instead of concatenating and splitting
    # each cell separately: the destination of every point is its cell's slot offset plus its
    # rank within the cell.
    point_starts = np.cumsum(counts) - counts
    slot_starts = np.cumsum(slots_per_cell) - slots_per_cell
    dest = (
        np.arange(len(idxs_ord_flat))
        - np.repeat(point_starts, counts)
        + np.repeat(slot_starts, counts)
    )
    flat = np.zeros(int(slots_per_cell.sum()), dtype=np.int64)
    flat[dest] = idxs_ord_flat + offset

    # One split covers every token of every cell; the tokens of a cell are then a contiguous run
    if pad_tokens:
        tokens = torch.split(torch.from_numpy(flat), token_size)
    else:
        token_lens = np.full(int(tokens_per_cell.sum()), token_size, dtype=np.int64)
        filled = counts > 0
        token_starts_filled = (np.cumsum(tokens_per_cell) - tokens_per_cell)[filled]
        last = token_starts_filled + tokens_per_cell[filled] - 1
        token_lens[last] = counts[filled] - (tokens_per_cell[filled] - 1) * token_size
        tokens = torch.split(torch.from_numpy(flat), token_lens.tolist())

    # token lens from numpy instead of len() per token tensor
    lens = [token_size] * len(tokens) if pad_tokens else token_lens.tolist()
    token_starts = (np.cumsum(tokens_per_cell) - tokens_per_cell).tolist()
    starts_nums = list(zip(token_starts, tokens_per_cell.tolist(), strict=True))
    idxs_ord = [list(tokens[start : start + num]) for start, num in starts_nums]
    idxs_ord_lens = [lens[start : start + num] for start, num in starts_nums]

    return idxs_ord, idxs_ord_lens


def tokenize_space(
    rdata,
    token_size,
    hl,
    pad_tokens=True,
    offset_step=0,
):
    """Process one window into tokens"""

    # idx_ord_lens is length is number of tokens per healpix cell
    idxs_ord, idxs_ord_lens = hpy_splits(rdata.coords, hl, token_size, pad_tokens, offset_step)

    return idxs_ord, idxs_ord_lens


def tokenize_spacetime(
    rdata,
    token_size,
    hl,
    pad_tokens=True,
):
    """Tokenize respecting an intrinsic time step in the data, i.e. each time step is tokenized
    separately
    """

    num_healpix_cells = 12 * 4**hl
    idxs_cells = [[] for _ in range(num_healpix_cells)]
    idxs_cells_lens = [[] for _ in range(num_healpix_cells)]

    offset_step = 0
    t_unique = np.unique(rdata.datetimes)
    for _, t in enumerate(t_unique):
        # data for current time step
        mask = t == rdata.datetimes
        rdata_cur = IOReaderData(
            rdata.coords[mask], rdata.geoinfos[mask], rdata.data[mask], rdata.datetimes[mask]
        )
        idxs_cur, idxs_cur_lens = tokenize_space(rdata_cur, token_size, hl, pad_tokens, offset_step)

        # collect data for all time steps
        idxs_cells = [t + tc for t, tc in zip(idxs_cells, idxs_cur, strict=True)]
        idxs_cells_lens = [t + tc_l for t, tc_l in zip(idxs_cells_lens, idxs_cur_lens, strict=True)]
        offset_step += mask.sum()

    return idxs_cells, idxs_cells_lens


def _take_rows(x, idxs):
    """x[idxs] along the first dim, index_select is much faster than advanced indexing"""
    return torch.index_select(x, 0, idxs) if torch.is_tensor(x) else x[idxs]


def _token_mask_info(idxs_cells, idxs_cells_lens, mask_tokens):
    """Per token mask info, without python loops over tensors.

    Returns
      mask : bool per token
      idxs_lens : length of each token
      tokens_per_cell : number of kept tokens per cell (int64)
      points_per_cell : number of kept points per cell (int64)
      idxs_data : indices of all kept points (torch, concatenated) or None if nothing is kept
    """
    num_cells = len(idxs_cells_lens)
    num_tokens_per_cell = np.fromiter(map(len, idxs_cells_lens), dtype=np.int64, count=num_cells)
    num_tokens = int(num_tokens_per_cell.sum())
    idxs_lens = np.fromiter(chain.from_iterable(idxs_cells_lens), dtype=np.int64, count=num_tokens)
    mask = np.asarray(mask_tokens, dtype=bool)
    assert mask.shape == (num_tokens,), "mask_tokens has to have one entry per token"

    # tokens of a cell are contiguous thus per cell sums = cumsum differences
    cell_ends = np.cumsum(num_tokens_per_cell)
    cell_starts = cell_ends - num_tokens_per_cell
    csum_tokens = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    csum_points = np.concatenate(([0], np.cumsum(np.where(mask, idxs_lens, 0))))
    tokens_per_cell = csum_tokens[cell_ends] - csum_tokens[cell_starts]
    points_per_cell = csum_points[cell_ends] - csum_points[cell_starts]

    idxs_data = list(compress(chain.from_iterable(idxs_cells), mask))
    idxs_data = torch.cat(idxs_data) if len(idxs_data) > 0 else None

    return mask, idxs_lens, tokens_per_cell, points_per_cell, idxs_data


def tokenize_apply_mask_source(
    idxs_cells,
    idxs_cells_lens,
    mask_tokens,
    mask_channels,
    stream_id,
    rdata,
    time_win,
    hpy_verts_rots,
    enc_time,
):
    """
    Apply masking to the data.

    Conceptually, the data is a matrix with the rows corresponding to data points / tokens and
    the cols the channels. Thereby mask_tokens acts on the rows, grouped according to the tokens as
    specified in idxs_cells and mask_channels acts on the columns.

    """

    def return_empty(rdata, idxs_cells_lens):
        tokens_cells = [torch.tensor([])]
        tokens_per_cell = torch.zeros(len(idxs_cells_lens), dtype=torch.int32)
        return tokens_cells, tokens_per_cell

    # apply spatial masking on a per token level
    if mask_tokens is None:
        return return_empty(rdata, idxs_cells_lens)

    # filter tokens using mask to obtain flat per data point index list
    mask, idxs_lens, tokens_per_cell, points_per_cell, idxs_data = _token_mask_info(
        idxs_cells, idxs_cells_lens, mask_tokens
    )

    if idxs_data is None:
        return return_empty(rdata, idxs_cells_lens)

    # filter list of token lens using mask and obtain flat list for splitting
    idxs_data_lens = idxs_lens[mask]

    # pad with zero at the begining of the conceptual 2D data tensor:
    # idxs_cells -> idxs_tokens -> idxs_data has been prepared so
    # that the zero-index is used to add the padding to the tokens to ensure fixed size
    def pad_row0(x):
        return torch.cat([torch.zeros_like(x[:1]), x])

    times_enc = enc_time(rdata.datetimes, time_win)
    coords_padded = pad_row0(rdata.coords)

    if mask_channels is not None:
        assert False, "to be implemented"
        # data = data_padded[ : channel_mask]

    # local coords, padding rows computed once per cell
    tokens_per_cell = torch.from_numpy(tokens_per_cell)
    masked_points_per_cell = torch.from_numpy(points_per_cell.astype(np.int32))
    coords_local = get_source_coords_local_padded(
        coords_padded, idxs_data, hpy_verts_rots, masked_points_per_cell
    )

    # all data: stream_id | datetimes | coords_local | geoinfos | data
    # columns joined per point first (row 0 = padding) then gathered once, coords_local set after
    num_points = times_enc.shape[0]
    stream_ids = torch.full([num_points + 1, 1], stream_id, dtype=torch.float32)
    coords_local_zeros = torch.zeros([num_points + 1, coords_local.shape[1]], dtype=torch.float32)
    per_point = torch.cat(
        (
            stream_ids,
            pad_row0(times_enc),
            coords_local_zeros,
            pad_row0(rdata.geoinfos),
            pad_row0(rdata.data),
        ),
        1,
    )
    tokens = torch.index_select(per_point, 0, idxs_data)
    col_coords = 1 + times_enc.shape[1]
    tokens[:, col_coords : col_coords + coords_local.shape[1]] = coords_local

    # split up tensor into tokens
    # TODO: idxs_data_lens is currently only defined when mask_tokens is not None
    if (idxs_data_lens == idxs_data_lens[0]).all():
        # padded tokens all have same length thus one stacked view (no copy), indexing and len()
        # work as for the list and StreamData.add_source skips torch.stack
        tokens_cells = tokens.view(len(idxs_data_lens), int(idxs_data_lens[0]), tokens.shape[1])
    else:
        tokens_cells = torch.split(tokens, idxs_data_lens.tolist())

    return tokens_cells, tokens_per_cell


def tokenize_apply_mask_target(
    stream_id,
    hl,
    idxs_cells,
    idxs_cells_lens,
    mask_tokens,
    mask_channels,
    rdata,
    time_win,
    hpy_verts_rots,
    hpy_verts_local,
    hpy_nctrs,
    enc_time,
    compute_coords_local=True,
):
    """
    Apply masking to the data.

    Conceptually, the data is a matrix with the rows corresponding to data points / tokens and
    the cols the channels. Thereby mask_tokens acts on the rows, grouped according to the tokens as
    specified in idxs_cells and mask_channels acts on the columns.

    compute_coords_local=False skips target coords encoding (coords_local empty), for when only
    target values are needed.
    """

    def return_empty(rdata, idxs_cells_lens):
        do = torch.zeros([0, rdata.data.shape[-1]])
        coords = torch.zeros([0, rdata.coords.shape[-1]])
        dt = np.array([], dtype=np.datetime64)
        masked_points_per_cell = torch.zeros(len(idxs_cells_lens), dtype=torch.int32)
        # data, datetimes, coords, coords_local, masked_points_per_cell
        return do, dt, coords, coords, masked_points_per_cell

    # apply spatial masking on a per token level
    if mask_tokens is None:
        return return_empty(rdata, idxs_cells_lens)

    # filter tokens using mask to obtain flat per data point index list
    _, idxs_lens, _, points_per_cell, idxs_data = _token_mask_info(
        idxs_cells, idxs_cells_lens, mask_tokens
    )

    if idxs_data is None:
        return return_empty(rdata, idxs_cells_lens)

    # apply mask
    datetimes = np.atleast_1d(rdata.datetimes[idxs_data])
    geoinfos = _take_rows(rdata.geoinfos, idxs_data)
    coords = _take_rows(rdata.coords, idxs_data)
    data = _take_rows(rdata.data, idxs_data)

    if mask_channels is not None:
        assert False, "to be implemented"
        # data = data_padded[ : channel_mask]

    masked_points_per_cell = torch.from_numpy(points_per_cell.astype(np.int32))

    # compute encoding of target coordinates used in prediction network
    if compute_coords_local and idxs_lens.sum() > 0:
        datetimes_enc = enc_time(datetimes, time_win)
        coords_local = get_target_coords_local(
            stream_id,
            hl,
            masked_points_per_cell,
            coords,
            geoinfos,
            datetimes_enc,
            hpy_verts_rots,
            hpy_verts_local,
            hpy_nctrs,
        )
        coords_local.requires_grad = False
    else:
        coords_local = torch.tensor([])

    return data, datetimes, coords, coords_local, masked_points_per_cell


def get_source_coords_local(
    coords: Tensor,
    hpy_verts_rots: Tensor,
    masked_points_per_cell,
) -> list[Tensor]:
    """Compute simple local coordinates for a set of 3D positions on the unit sphere."""

    # remove padding from coords
    posr3 = s2tor3(*theta_phi_to_standard_coords(coords))
    posr3[0, 0] = 0.0
    posr3[0, 1] = 0.0
    posr3[0, 2] = 0.0

    rots = torch.repeat_interleave(hpy_verts_rots, masked_points_per_cell, dim=0)
    # BMM only works for b x n x m and b x m x 1
    # adding a dummy dimension to posr3
    vec_rot = torch.bmm(rots, posr3.unsqueeze(-1)).squeeze(-1)
    vec_scaled = r3tos2(vec_rot).to(torch.float32)

    # TODO: vec_scaled are small -> should they be normalized/rescaled?

    return vec_scaled


def _source_coords_local_rows(coords: Tensor, rots: Tensor) -> Tensor:
    """get_source_coords_local for given rows and per row rotations, without the row 0 reset"""
    posr3 = s2tor3(*theta_phi_to_standard_coords(coords))
    vec_rot = torch.bmm(rots, posr3.unsqueeze(-1)).squeeze(-1)
    return r3tos2(vec_rot).to(torch.float32)


def get_source_coords_local_padded(
    coords_padded: Tensor,
    idxs_data: Tensor,
    hpy_verts_rots: Tensor,
    masked_points_per_cell: Tensor,
) -> Tensor:
    """Same values as get_source_coords_local(coords_padded[idxs_data], ...).

    A padding row (index 0) only depends on its cell thus it is computed once per cell, per point
    work only for real points.
    """
    num_cells = hpy_verts_rots.shape[0]
    cell_of_row = torch.repeat_interleave(
        torch.arange(num_cells), masked_points_per_cell.to(torch.int64)
    )

    # row 0 is done on its own below
    is_pad = idxs_data == 0
    is_real = ~is_pad
    is_pad[0] = is_real[0] = False
    rows_real = torch.nonzero(is_real).squeeze(1)
    rows_pad = torch.nonzero(is_pad).squeeze(1)

    out = torch.empty((idxs_data.shape[0], 2), dtype=torch.float32)

    # real points
    coords = torch.index_select(coords_padded, 0, torch.index_select(idxs_data, 0, rows_real))
    rots = torch.index_select(hpy_verts_rots, 0, torch.index_select(cell_of_row, 0, rows_real))
    out[rows_real] = _source_coords_local_rows(coords, rots)

    # padding rows: one value per cell
    if rows_pad.numel() > 0:
        pad_vals = _source_coords_local_rows(
            coords_padded[0:1].expand(num_cells, -1), hpy_verts_rots
        )
        out[rows_pad] = torch.index_select(
            pad_vals, 0, torch.index_select(cell_of_row, 0, rows_pad)
        )

    # row 0 with zero position vector as in get_source_coords_local
    rot0 = hpy_verts_rots[cell_of_row[0]].unsqueeze(0)
    posr3_0 = torch.zeros((1, 3, 1), dtype=coords_padded.dtype)
    out[0] = r3tos2(torch.bmm(rot0, posr3_0).squeeze(-1)).to(torch.float32)[0]

    return out


def _rotate_points_per_cell(cell_rots, points, points_per_cell):
    """Apply each healpix cell's 3x3 rotation to the points that belong to it.

    ``points`` is already concatenated; ``points_per_cell[i]`` is the number of rows
    that belong to cell ``i`` (zero for empty cells). Equivalent to concatenating a
    per-cell list and then ``locs_to_cell_coords_ctrs``, but without building that list.
    """
    cell_index = torch.repeat_interleave(
        torch.arange(points_per_cell.shape[0], device=points.device),
        points_per_cell.to(device=points.device, dtype=torch.int64),
    )
    return torch.bmm(cell_rots[cell_index], points.unsqueeze(-1)).squeeze(-1)


def get_target_coords_local(
    stream_id,
    hlc,
    masked_points_per_cell,
    coords,
    target_geoinfos,
    target_times,
    verts_rots,
    verts_local,
    nctrs,
):
    """Generate local coordinates for target coords w.r.t healpix cell vertices and
    and for healpix cell vertices themselves
    """

    target_coords = s2tor3(*theta_phi_to_standard_coords(coords))

    if target_coords.shape[0] == 0:
        return torch.tensor([])

    verts00_rots, verts10_rots, verts11_rots, verts01_rots, vertsmm_rots = verts_rots

    a = torch.zeros(
        [
            *target_coords.shape[:-1],
            1 + target_geoinfos.shape[1] + target_times.shape[1] + 5 * (3 * 5) + 3 * 8,
        ]
    )
    a[0] = stream_id
    geoinfo_offset = 1
    a[..., geoinfo_offset : geoinfo_offset + target_times.shape[1]] = target_times
    geoinfo_offset += target_times.shape[1]
    a[..., geoinfo_offset : geoinfo_offset + target_geoinfos.shape[1]] = target_geoinfos
    geoinfo_offset += target_geoinfos.shape[1]

    ref = torch.tensor([1.0, 0.0, 0.0])
    counts = masked_points_per_cell

    occupied = counts > 0
    vls = torch.repeat_interleave(verts_local[occupied], counts[occupied], dim=0).transpose(0, 1)

    zi = 0
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - _rotate_points_per_cell(
        verts00_rots, target_coords, counts
    )

    zi = 3
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[0]

    zi = 15
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - _rotate_points_per_cell(
        verts10_rots, target_coords, counts
    )

    zi = 18
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[1]

    zi = 30
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - _rotate_points_per_cell(
        verts11_rots, target_coords, counts
    )

    zi = 33
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[2]

    zi = 45
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - _rotate_points_per_cell(
        verts01_rots, target_coords, counts
    )

    zi = 48
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[3]

    zi = 60
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - _rotate_points_per_cell(
        vertsmm_rots, target_coords, counts
    )

    zi = 63
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[4]

    # Eight neighbor-center frames: one vecs_to_rots + batched matmul instead of
    # eight list-split round trips through locs_to_ctr_coords.
    n_nbors, n_cells, _ = nctrs.shape
    nbor_rots = vecs_to_rots(nctrs.reshape(-1, 3)).to(torch.float32).reshape(n_nbors, n_cells, 3, 3)
    cell_index = torch.repeat_interleave(
        torch.arange(n_cells, device=target_coords.device),
        counts.to(device=target_coords.device, dtype=torch.int64),
    )
    rotated_nbors = torch.matmul(
        nbor_rots[:, cell_index],
        target_coords.unsqueeze(0).unsqueeze(-1).expand(n_nbors, -1, -1, -1),
    ).squeeze(-1)
    tcs_ctrs = (ref - rotated_nbors).permute(1, 0, 2).reshape(target_coords.shape[0], -1)
    zi = 75
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + (3 * 8))] = tcs_ctrs

    # remaining geoinfos (zenith angle etc)
    zi = 99
    a[..., (geoinfo_offset + zi) :] = target_coords[..., (geoinfo_offset + 2) :]

    # Careful when merging develop-ssl into develop.
    # This is not to be merged in to develop.
    a[..., 98] = np.sin(coords[:, 0])
    a[..., 97] = np.cos(coords[:, 0])
    a[..., 96] = np.sin(coords[:, 1])
    a[..., 95] = np.cos(coords[:, 1])

    return a
