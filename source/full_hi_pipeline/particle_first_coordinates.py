"""Explicit object-coordinate steps: real positions -> fiducial RSD -> AP.

Positions are in h^-1 Mpc and peculiar velocities in physical km/s. The
reference and trial cosmologies share h; only Omega_m and constant w vary.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.integrate import quad

from .stream_density import BOX_SIZE_MPC_H, OMEGA_M, W_VALUE


def expansion_e(z: float, omega_m: float, w: float) -> float:
    if not np.isfinite([z, omega_m, w]).all() or z < 0 or not 0 < omega_m <= 1:
        raise ValueError('Require finite z>=0, 0<Omega_m<=1 and finite w')
    with np.errstate(over='raise', invalid='raise'):
        value = np.sqrt(omega_m * (1. + z)**3 + (1 - omega_m) * np.power(1. + z, 3 * (1 + w)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Invalid expansion rate')
    return float(value)


def distance_integral(z: float, omega_m: float, w: float) -> float:
    expansion_e(z, omega_m, w)
    return float(quad(lambda zz: 1 / expansion_e(zz, omega_m, w), 0, z,
                      epsabs=1e-10, epsrel=1e-9)[0])


def ap_factors(z: float, omega_m: float, w: float, *, los_axis: int = 2,
               reference_omega_m: float = OMEGA_M, reference_w: float = W_VALUE) -> np.ndarray:
    """Step 3 coefficients, ordered along the three Cartesian axes."""
    if los_axis not in (0, 1, 2):
        raise ValueError('los_axis must be 0, 1 or 2')
    efid = expansion_e(z, reference_omega_m, reference_w)
    etrial = expansion_e(z, omega_m, w)
    if omega_m == reference_omega_m and w == reference_w:
        return np.ones(3)
    transverse = (distance_integral(z, omega_m, w) /
                  distance_integral(z, reference_omega_m, reference_w)) if z > 0 else 1.
    factors = np.full(3, transverse, dtype=np.float64)
    factors[los_axis] = efid / etrial
    return factors


def _positions(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError('Positions/velocities must be finite (N,3) arrays')
    return result


def add_fiducial_rsd(positions: np.ndarray, velocities: np.ndarray, z: float, *,
                     los_axis: int = 2, reference_omega_m: float = OMEGA_M,
                     reference_w: float = W_VALUE) -> np.ndarray:
    """Step 2: s_fid = x + v_LOS*(1+z)/(100 E_fid); leave input untouched.

    Return unwrapped coordinates. Periodic wrapping happens inside CIC after
    AP, in the transformed cuboid. No trial cosmology is used in this step.
    """
    if los_axis not in (0, 1, 2):
        raise ValueError('los_axis must be 0, 1 or 2')
    real, velocity = _positions(positions), _positions(velocities)
    if real.shape != velocity.shape:
        raise ValueError('Position and velocity shapes differ')
    rsd = real.copy()
    rsd[:, los_axis] += velocity[:, los_axis] * (1 + z) / (
        100 * expansion_e(z, reference_omega_m, reference_w))
    return rsd


def apply_ap_to_objects(rsd_positions: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Step 3: scale the COMPLETE RSD positions, including their velocity term."""
    positions = _positions(rsd_positions)
    factors = np.asarray(factors, dtype=np.float64)
    if factors.shape != (3,) or not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError('AP factors must be three positive finite numbers')
    return np.ascontiguousarray(positions * factors)


def choose_grid(factors: np.ndarray, *, grid_size: int = 512,
                cell_size_mpc_h: float | None = None,
                box_size_mpc_h: float = BOX_SIZE_MPC_H) -> dict:
    """Choose a NEW trial cuboid grid, before depositing any objects.

    fixed-count: N^3 cells; physical cell lengths change with AP.
    fixed-cell: ceil(L_i / requested_spacing) cells per axis, so actual
    spacing never exceeds the request. Integer cell counts make it approximate.
    """
    factors = np.asarray(factors, dtype=float)
    if factors.shape != (3,) or not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError('Invalid AP factors')
    if not np.isfinite(box_size_mpc_h) or box_size_mpc_h <= 0:
        raise ValueError('Invalid box size')
    lengths = float(box_size_mpc_h) * factors
    if cell_size_mpc_h is None:
        if isinstance(grid_size, bool) or int(grid_size) != grid_size or grid_size < 8:
            raise ValueError('grid_size must be an integer >=8')
        shape = [int(grid_size)] * 3
        policy = 'fixed_count'
    else:
        if not np.isfinite(cell_size_mpc_h) or cell_size_mpc_h <= 0:
            raise ValueError('cell_size_mpc_h must be positive and finite')
        shape = [math.ceil(float(length) / cell_size_mpc_h) for length in lengths]
        if min(shape) < 8:
            raise ValueError('Requested spacing yields fewer than 8 cells on an axis')
        policy = 'fixed_maximum_physical_spacing'
    return {'policy': policy, 'shape': shape, 'trial_lengths_mpc_h': lengths.tolist(),
            'actual_cell_lengths_mpc_h': (lengths / shape).tolist(),
            'requested_cell_size_mpc_h': cell_size_mpc_h,
            'requested_grid_size': int(grid_size) if cell_size_mpc_h is None else None}
