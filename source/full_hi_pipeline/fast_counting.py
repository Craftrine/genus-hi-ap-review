"""Exact periodic genus counting using one threshold-rank pass per voxel.

For sorted thresholds ``t``, let ``rank(v) = searchsorted(t, v, 'left')``.
Then ``v > t[i]`` iff ``rank(v) > i``. Because rank is monotone,
``rank(min(v1, ..., vn)) == min(rank(v1), ..., rank(vn))``, including ties
and repeated thresholds. Thus the cubical-complex Euler characteristic can
be counted with integer neighbour minima after a single floating comparison
pass. This changes the implementation, not the statistic or its thresholds.

No fast-math, approximate binning, threshold interpolation, or floating-point
Euler accumulation is used. The frozen reference implementation is untouched.
"""
from __future__ import annotations

from math import prod

import numpy as np
from numba import njit, prange


def _rank_dtype_for_threshold_count(count: int) -> np.dtype:
    """Ranks range from zero THROUGH count, so count=256 needs uint16."""
    if not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("threshold count must be a non-negative integer")
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if count <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError("threshold count exceeds uint64 rank capacity")


@njit(inline="always")
def _left_rank(thresholds: np.ndarray, value: float) -> int:
    lo = 0
    hi = thresholds.size
    while lo < hi:
        middle = (lo + hi) // 2
        if thresholds[middle] < value:
            lo = middle + 1
        else:
            hi = middle
    return lo


@njit(parallel=True, cache=True)
def _encode_ranks(
    field: np.ndarray, thresholds: np.ndarray, ranks: np.ndarray
) -> int:
    """Fuse finite-value validation into rank encoding; caller rejects failures."""
    nx, ny, nz = field.shape
    invalid_by_x = np.zeros(nx, dtype=np.int64)
    for i in prange(nx):
        invalid = 0
        for j in range(ny):
            for k in range(nz):
                # Never down-cast the float64 thresholds to field precision.
                value = np.float64(field[i, j, k])
                if not np.isfinite(value):
                    invalid += 1
                    ranks[i, j, k] = 0
                else:
                    ranks[i, j, k] = _left_rank(thresholds, value)
        invalid_by_x[i] = invalid
    return invalid_by_x.sum()


@njit(inline="always")
def _minimum_four(a: int, b: int, c: int, d: int) -> int:
    value = a if a < b else b
    value = value if value < c else c
    return value if value < d else d


@njit(parallel=True, cache=True)
def _periodic_rank_histograms(
    ranks: np.ndarray, threshold_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-x histograms avoid atomics and races between parallel iterations.

    The signed histogram stores genus numerators directly:
    -vertex + three edges - three faces + cube. All increments are int64.
    """
    nx, ny, nz = ranks.shape
    signed = np.zeros((nx, threshold_count + 1), dtype=np.int64)
    vertices = np.zeros((nx, threshold_count + 1), dtype=np.int64)
    for i in prange(nx):
        ip = i + 1
        if ip == nx:
            ip = 0
        local_signed = signed[i]
        local_vertices = vertices[i]
        for j in range(ny):
            jp = j + 1
            if jp == ny:
                jp = 0
            for k in range(nz):
                kp = k + 1
                if kp == nz:
                    kp = 0

                v000 = ranks[i, j, k]
                v100 = ranks[ip, j, k]
                v010 = ranks[i, jp, k]
                v001 = ranks[i, j, kp]
                v110 = ranks[ip, jp, k]
                v101 = ranks[ip, j, kp]
                v011 = ranks[i, jp, kp]
                v111 = ranks[ip, jp, kp]

                local_vertices[v000] += 1
                local_signed[v000] -= 1
                local_signed[v000 if v000 < v100 else v100] += 1
                local_signed[v000 if v000 < v010 else v010] += 1
                local_signed[v000 if v000 < v001 else v001] += 1
                local_signed[_minimum_four(v000, v100, v010, v110)] -= 1
                local_signed[_minimum_four(v000, v100, v001, v101)] -= 1
                local_signed[_minimum_four(v000, v010, v001, v011)] -= 1
                front = _minimum_four(v000, v100, v010, v001)
                back = _minimum_four(v110, v101, v011, v111)
                local_signed[front if front < back else back] += 1
    return signed, vertices


@njit(cache=True)
def _finish_histograms(
    signed: np.ndarray, vertices: np.ndarray, volume: float, cell_count: int
) -> tuple[np.ndarray, np.ndarray]:
    nx, nbins = signed.shape
    signed_bins = np.zeros(nbins, dtype=np.int64)
    vertex_bins = np.zeros(nbins, dtype=np.int64)
    for i in range(nx):
        for b in range(nbins):
            signed_bins[b] += signed[i, b]
            vertex_bins[b] += vertices[i, b]
    curve = np.empty(nbins - 1, dtype=np.float64)
    fractions = np.empty(nbins - 1, dtype=np.float64)
    signed_suffix = np.int64(0)
    vertex_suffix = np.int64(0)
    for b in range(nbins - 1, 0, -1):
        signed_suffix += signed_bins[b]
        vertex_suffix += vertex_bins[b]
        # Match the frozen ``-float(Euler) / volume`` even for signed zero.
        curve[b - 1] = -float(-signed_suffix) / volume
        fractions[b - 1] = float(vertex_suffix) / float(cell_count)
    return curve, fractions


def compute_curve_and_fractions_fast(
    field: np.ndarray,
    thresholds: np.ndarray,
    volume: float,
    periodic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact periodic genus density and strict-above volume fractions.

    Parameters
    ----------
    field:
        Non-empty real float32 or float64 3D array. Read-only and strided arrays
        are accepted without changing their values or writeability. The frozen
        production measurement uses float32; this function never silently
        rounds float64 values or float64 thresholds to float32.
    thresholds:
        Finite, non-decreasing one-dimensional thresholds. Repeats and an empty
        threshold list are supported. Comparison is strictly ``field > t``.
    volume:
        Finite, positive physical volume used only to normalize the genus.
    periodic:
        Only periodic cubical complexes are implemented here. False is rejected
        explicitly rather than silently applying different boundary conditions.

    Notes
    -----
    A uint8/uint16/uint32/uint64 rank grid is selected from the threshold count.
    At 41 thresholds it costs one byte per voxel; at 641 it costs two. Two
    int64 per-x histograms replace the reference's eight per-x histograms.
    Floating-point inputs and threshold coordinates are not modified.
    """
    if not periodic:
        raise NotImplementedError(
            "fast counting implements periodic=True only; use the reference "
            "counter for open boundaries"
        )
    values = np.asarray(field)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError("field must be a non-empty 3D array")
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("field must have real float32 or float64 dtype")
    if np.iscomplexobj(thresholds):
        raise TypeError("thresholds must be real")
    thr = np.asarray(thresholds, dtype=np.float64)
    if thr.ndim != 1:
        raise ValueError("thresholds must be one-dimensional")
    if not np.isfinite(thr).all() or np.any(thr[1:] < thr[:-1]):
        raise ValueError("thresholds must be finite and sorted non-decreasing")
    physical_volume = float(volume)
    if not np.isfinite(physical_volume) or physical_volume <= 0:
        raise ValueError("volume must be positive and finite")
    cell_count = prod(values.shape)
    if cell_count > np.iinfo(np.int64).max // 8:
        raise ValueError("field is too large for safe int64 Euler accumulation")
    if thr.size == 0:
        if not np.isfinite(values).all():
            raise ValueError("field must contain only finite values")
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    values = np.ascontiguousarray(values)
    thr = np.ascontiguousarray(thr)
    ranks = np.empty(values.shape, dtype=_rank_dtype_for_threshold_count(thr.size))
    invalid = _encode_ranks(values, thr, ranks)
    if invalid:
        raise ValueError("field must contain only finite values")
    signed, vertices = _periodic_rank_histograms(ranks, thr.size)
    return _finish_histograms(signed, vertices, physical_volume, cell_count)


__all__ = ["compute_curve_and_fractions_fast"]
