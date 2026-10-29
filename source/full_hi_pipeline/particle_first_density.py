"""Stream ALL positive-HI objects through RSD -> AP -> a newly deposited CIC.

Every distinct trial reads object records and creates a new mesh. A saved
mesh can only be reused for the identical signed trial/source configuration.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import h5py
import numba
from numba import njit, prange, set_num_threads
import numpy as np

from . import particle_first_coordinates as coordinates
from . import particle_first_catalog as catalog
from .genus_grid import _atomic_json, _canonical_hash, sha256_file
from .stream_density import (BOX_SIZE_MPC_H, HUBBLE_H, JIUTIAN_DTYPE, OMEGA_M,
                             REDSHIFT_BY_SNAPSHOT, W_VALUE)

DENSITY_SCHEMA = 'jiutian_particle_first_RSD_AP_CIC_v1'
ORDER = ['read_all_valid_positive_HI_objects', 'add_fiducial_RSD_to_objects',
         'apply_AP_to_complete_RSD_object_positions', 'periodic_CIC_in_trial_cuboid',
         'normalize_trial_HI_density_contrast']


@contextmanager
def output_lock(path: Path):
    """POSIX nonblocking lock; a completed lock file is harmless on restart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + '.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process owns {path}') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def inspect_source(path: Path, max_records: int | None, *, hash_content: bool = True) -> dict:
    """Full SHA-256 of the records actually used, not a sparse source sample."""
    path = Path(path).expanduser().resolve()
    before = path.stat()
    if before.st_size == 0 or before.st_size % JIUTIAN_DTYPE.itemsize:
        raise ValueError('Input must contain complete 40-byte records')
    if max_records is not None and (isinstance(max_records, bool) or int(max_records) != max_records or max_records < 1):
        raise ValueError('max_records must be a positive integer or None')
    total = before.st_size // JIUTIAN_DTYPE.itemsize
    processed = total if max_records is None else min(total, int(max_records))
    digest = hashlib.sha256()
    if hash_content:
        remaining = processed * JIUTIAN_DTYPE.itemsize
        with path.open('rb') as handle:
            while remaining:
                chunk = handle.read(min(8 * 1024**2, remaining))
                if not chunk:
                    raise RuntimeError('Source truncated during hashing')
                digest.update(chunk)
                remaining -= len(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Source changed during hashing')
    return {'path': str(path), 'size_bytes': before.st_size, 'mtime_ns': before.st_mtime_ns,
            'source_total_records': total, 'records_to_process': processed,
            'max_records_requested': max_records, 'complete_catalog': processed == total,
            'processed_records_sha256': digest.hexdigest() if hash_content else None,
            'hash_scope': 'all processed raw record bytes, including rejected records'}


def make_density_config(source_path: Path, *, snapshot: int, omega_m: float, w: float,
                         grid_size: int = 512, cell_size_mpc_h: float | None = None,
                         workers: int = 2, chunk_records: int = 2_000_000,
                         max_records: int | None = None, max_memory_gib: float = 12.,
                         hash_source: bool = True, source_kind: str = 'provided_compact_catalogue',
                         catalog_backend: str = 'numpy') -> dict:
    if snapshot not in REDSHIFT_BY_SNAPSHOT:
        raise ValueError('Unknown snapshot')
    if source_kind not in ('provided_compact_catalogue', 'synthetic_demo'):
        raise ValueError('Unknown source_kind')
    if catalog_backend not in ('numpy', 'numba'):
        raise ValueError('catalog_backend must be numpy or numba')
    for value, name in [(workers, 'workers'), (chunk_records, 'chunk_records')]:
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(name + ' must be a positive integer')
    if workers > numba.config.NUMBA_NUM_THREADS:
        raise ValueError('workers exceeds available Numba thread capacity')
    if not np.isfinite(max_memory_gib) or max_memory_gib <= 0:
        raise ValueError('max_memory_gib must be positive')
    z = REDSHIFT_BY_SNAPSHOT[snapshot]
    factors = coordinates.ap_factors(z, omega_m, w)
    grid = coordinates.choose_grid(factors, grid_size=grid_size, cell_size_mpc_h=cell_size_mpc_h)
    # Conservative allocation budget, including float64 worker grids and coordinates.
    cells = math.prod(grid['shape'])
    estimate = (cells * (8 * workers + 32) + 160 * chunk_records + 64 * 1024**2) / 1024**3
    if estimate > max_memory_gib:
        raise MemoryError(f'Estimated {estimate:.2f} GiB exceeds {max_memory_gib:g} GiB; '
                          'reduce grid size/workers/chunk size or explicitly raise the budget')
    source = inspect_source(source_path, max_records, hash_content=hash_source)
    return {'schema': DENSITY_SCHEMA, 'snapshot': int(snapshot), 'redshift': z,
            'trial_cosmology': {'omega_m': float(omega_m), 'w': float(w), 'h': HUBBLE_H},
            'reference_cosmology': {'omega_m': OMEGA_M, 'w': W_VALUE, 'h': HUBBLE_H},
            'source': source, 'source_kind': source_kind, 'grid': grid, 'ap_factors_xyz': factors.tolist(),
            'reference_box_size_mpc_h': BOX_SIZE_MPC_H, 'los_axis': 2,
            'rsd_displacement_mpc_h_per_km_s': (1 + z) / (100 * coordinates.expansion_e(z, OMEGA_M, W_VALUE)),
            'velocity_unit_assumption': 'physical km/s', 'position_unit': 'h^-1 Mpc',
            'selection': 'all_finite_Pos_Vel_HImass_and_HImass_positive', 'weighting': 'native_HImass',
            'record_schema': JIUTIAN_DTYPE.descr, 'record_itemsize_bytes': JIUTIAN_DTYPE.itemsize,
            'operation_order': ORDER, 'workers': int(workers), 'chunk_records': int(chunk_records),
            'catalog_backend': catalog_backend,
            'accumulator_dtype': 'float64', 'density_dtype': 'float32',
            'estimated_memory_gib': estimate, 'conservation_tolerance': 1e-10,
            'source_code_sha256': {p.name: sha256_file(p) for p in
                                   [Path(__file__), Path(coordinates.__file__), Path(catalog.__file__), Path(__file__).with_name('stream_density.py')]}}


def valid_objects(records: np.ndarray, offset: int = 0, *, backend='numpy'):
    """Step 1: apply exactly the same finite/positive HI selection to all cases."""
    if backend == 'numba':
        return catalog.extract_valid_objects(records, offset)
    if backend != 'numpy':
        raise ValueError('Unknown catalog backend')
    finite = (np.isfinite(records['Pos']).all(axis=1) & np.isfinite(records['Vel']).all(axis=1)
              & np.isfinite(records['HImass']))
    positive = records['HImass'] > 0
    keep = finite & positive
    pos = np.ascontiguousarray(records['Pos'][keep], dtype=np.float64)
    if np.any(pos < -1e-4) or np.any(pos > BOX_SIZE_MPC_H + 1e-4):
        raise ValueError('Real positions outside the declared original box; check units/schema')
    return (pos, np.ascontiguousarray(records['Vel'][keep], dtype=np.float64),
            np.ascontiguousarray(records['HImass'][keep], dtype=np.float64),
            np.flatnonzero(keep) + offset,
            {'nonfinite': int((~finite).sum()), 'nonpositive': int((finite & ~positive).sum())})


@njit(parallel=True, cache=True)
def _deposit_transformed_objects(positions, weights, lengths, worker_grids):
    """Step 4: CIC takes ALREADY transformed positions; no RSD or AP here."""
    workers, nx, ny, nz = worker_grids.shape
    for worker in prange(workers):
        mesh = worker_grids[worker]
        for index in range(len(weights) * worker // workers, len(weights) * (worker + 1) // workers):
            x = positions[index, 0] % lengths[0] * nx / lengths[0]
            y = positions[index, 1] % lengths[1] * ny / lengths[1]
            z = positions[index, 2] % lengths[2] * nz / lengths[2]
            ix, iy, iz = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
            fx, fy, fz = x - ix, y - iy, z - iz
            for dx in range(2):
                wx = fx if dx else 1 - fx
                for dy in range(2):
                    wy = fy if dy else 1 - fy
                    for dz in range(2):
                        wz = fz if dz else 1 - fz
                        mesh[(ix+dx) % nx, (iy+dy) % ny, (iz+dz) % nz] += weights[index] * wx * wy * wz


def cic_deposit(positions, weights, lengths, worker_grids):
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    if positions.shape != (len(weights), 3) or not np.isfinite(positions).all():
        raise ValueError('Invalid transformed positions')
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('HI weights must be finite and positive')
    if lengths.shape != (3,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Invalid physical cuboid lengths')
    if worker_grids.ndim != 4 or worker_grids.dtype != np.float64 or not worker_grids.flags.c_contiguous:
        raise ValueError('CIC accumulators must be contiguous float64 (workers,Nx,Ny,Nz)')
    _deposit_transformed_objects(positions, weights, lengths, worker_grids)


def to_density_contrast(mass_mesh: np.ndarray, input_mass: float, *, tolerance=1e-10):
    """Step 5: rho/mean(rho)-1; equal trial cell volume cancels in this ratio."""
    deposited = float(np.sum(mass_mesh, dtype=np.float64))
    if not np.isfinite(input_mass) or input_mass <= 0:
        raise ValueError('No valid positive HI mass')
    relative = abs(deposited - input_mass) / input_mass
    if relative > tolerance:
        raise RuntimeError(f'CIC mass conservation failed: {relative}')
    mean = deposited / mass_mesh.size
    mass_mesh /= mean
    mass_mesh -= 1
    delta = np.asarray(mass_mesh, dtype=np.float32)
    if not np.isfinite(delta).all() or abs(np.mean(delta, dtype=np.float64)) > 2e-6:
        raise RuntimeError('Invalid density contrast')
    return delta, {'input_HI_mass_native': input_mass, 'deposited_HI_mass_native': deposited,
                   'relative_conservation_error': relative, 'mean_HI_mass_per_trial_cell_native': mean,
                   'delta_mean': float(np.mean(delta, dtype=np.float64)),
                   'delta_min': float(delta.min()), 'delta_max': float(delta.max())}


def validate_density(path: Path, expected_config: dict | None = None) -> dict:
    with h5py.File(path, 'r') as handle:
        config = json.loads(handle.attrs['config_json'])
        signature = _canonical_hash(config)
        if config.get('schema') != DENSITY_SCHEMA or signature != handle.attrs['config_sha256']:
            raise ValueError('Wrong or corrupt particle-first density contract')
        if expected_config is not None and signature != _canonical_hash(expected_config):
            raise ValueError('Density cache request mismatch (source/grid/AP/RSD/selection/config); use a new output path')
        field = handle['density_contrast_trial']
        if list(field.shape) != config['grid']['shape'] or field.dtype != np.float32:
            raise ValueError('Density dataset shape or dtype disagrees with contract')
        sha = hashlib.sha256()
        for start in range(0, field.shape[0], 8):
            slab = np.ascontiguousarray(field[start:start+8])
            if not np.isfinite(slab).all():
                raise ValueError('Nonfinite density dataset')
            sha.update(slab.tobytes())
        if sha.hexdigest() != handle.attrs['delta_sha256']:
            raise ValueError('Density content checksum mismatch')
        quality = json.loads(handle.attrs['quality_json'])
        if quality['relative_conservation_error'] > config['conservation_tolerance']:
            raise ValueError('Stored conservation check failed')
        return {'status': 'PASS', 'path': str(Path(path).resolve()), 'config': config,
                'config_sha256': signature, 'delta_sha256': sha.hexdigest(), 'quality': quality,
                'coordinate_trace': json.loads(handle.attrs['coordinate_trace_json'])}


def build_particle_first_density(source_path: Path, output_path: Path, **options) -> dict:
    """Read objects anew for this trial and execute each coordinate step literally."""
    output_path = Path(output_path).expanduser().resolve()
    source_path = Path(source_path).expanduser().resolve()
    if output_path == source_path:
        raise ValueError('Output may not replace the source catalog')
    config = make_density_config(source_path, **options)
    if config['source']['processed_records_sha256'] is None:
        raise ValueError('A dry-run plan cannot be used to build a density field')
    with output_lock(output_path):
        if output_path.exists():
            result = validate_density(output_path, config)
            result['cache_reused'] = True
            return result
        workers, chunk = config['workers'], config['chunk_records']
        set_num_threads(workers)
        shape = tuple(config['grid']['shape'])
        lengths = np.array(config['grid']['trial_lengths_mpc_h'])
        factors = np.array(config['ap_factors_xyz'])
        local = np.zeros((workers, *shape), dtype=np.float64)
        digest, processed, kept, input_mass = hashlib.sha256(), 0, 0, 0.
        rejected, trace = {'nonfinite': 0, 'nonpositive': 0}, []
        total, started = config['source']['records_to_process'], time.monotonic()
        with source_path.open('rb') as handle:
            while processed < total:
                records = np.fromfile(handle, dtype=JIUTIAN_DTYPE, count=min(chunk, total - processed))
                if len(records) == 0:
                    raise RuntimeError('Source truncated during deposition')
                digest.update(memoryview(records).cast('B'))
                real, velocity, weights, ids, bad = valid_objects(records, processed, backend=config['catalog_backend'])  # Step 1
                rsd = coordinates.add_fiducial_rsd(real, velocity, config['redshift'])  # Step 2
                trial = coordinates.apply_ap_to_objects(rsd, factors)                 # Step 3
                cic_deposit(trial, weights, lengths, local)                          # Step 4
                for i in range(min(8 - len(trace), len(weights))):
                    trace.append({'source_record_index': int(ids[i]), 'real_position': real[i].tolist(),
                                  'velocity_km_s': velocity[i].tolist(), 'RSD_position_unwrapped': rsd[i].tolist(),
                                  'AP_position_unwrapped': trial[i].tolist(),
                                  'AP_position_periodic': np.mod(trial[i], lengths).tolist(), 'HI_mass_native': float(weights[i])})
                kept += len(weights)
                input_mass += float(np.sum(weights, dtype=np.float64))
                for key in rejected:
                    rejected[key] += bad[key]
                processed += len(records)
                if processed == total or processed <= chunk or processed % (10 * chunk) == 0:
                    print(f'[particle-first] snap{config["snapshot"]} {processed:,}/{total:,} records; '
                          f'RSD -> AP -> CIC; {time.monotonic()-started:.1f}s', flush=True)
        stat = source_path.stat()
        if digest.hexdigest() != config['source']['processed_records_sha256'] or (
            stat.st_size, stat.st_mtime_ns) != (config['source']['size_bytes'], config['source']['mtime_ns']):
            raise RuntimeError('Source identity changed while building this trial')
        if kept + sum(rejected.values()) != processed:
            raise RuntimeError('Object selection accounting does not close')
        mass_mesh = np.sum(local, axis=0, dtype=np.float64)
        del local
        delta, quality = to_density_contrast(mass_mesh, input_mass)  # Step 5
        del mass_mesh
        quality.update(records_processed=processed, positive_HI_records=kept, rejected=rejected,
                       complete_catalog=config['source']['complete_catalog'], source_kind=config['source_kind'],
                       elapsed_seconds=time.monotonic()-started)
        temporary = output_path.with_name('.' + output_path.name + f'.tmp-{os.getpid()}')
        try:
            with h5py.File(temporary, 'w') as handle:
                handle.attrs.update(schema=DENSITY_SCHEMA, config_json=json.dumps(config, sort_keys=True),
                                    config_sha256=_canonical_hash(config), quality_json=json.dumps(quality),
                                    coordinate_trace_json=json.dumps(trace), delta_sha256=hashlib.sha256(delta.tobytes()).hexdigest(),
                                    generated_utc=datetime.now(timezone.utc).isoformat(),
                                    HImass_unit_status='native_catalogue_unit_unconfirmed', absolute_brightness_temperature_calibrated=False)
                handle.create_dataset('density_contrast_trial', data=delta, chunks=tuple(min(32, n) for n in shape),
                                      compression='lzf', shuffle=True)
            temporary.replace(output_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        result = validate_density(output_path, config)
        result['cache_reused'] = False
        _atomic_json(output_path.with_suffix('.summary.json'), result)
        return result
