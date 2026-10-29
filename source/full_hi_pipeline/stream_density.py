"""Stream a Jiutian compact catalogue into full-HI real/RSD CIC meshes.

The compact source is never reshaped as an unstructured float array.  Records
are read with the audited 40-byte schema and processed in bounded chunks.  All
finite, positive-HImass objects contribute; no top-N selection is performed.

The output is deliberately a dimensionless, noiseless HI density contrast.
The native HImass normalization is retained as an unconfirmed catalogue unit.
A physical brightness temperature must not be derived until the data provider
confirms that normalization.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import h5py
import numba
from numba import njit, prange, set_num_threads
import numpy as np


DENSITY_SCHEMA = "jiutian_full_hi_streaming_density_v1"
BOX_SIZE_MPC_H = 1000.0
OMEGA_M = 0.3111
W_VALUE = -1.0
HUBBLE_H = 0.6766
LOS_AXIS = 2
SOURCE_FINGERPRINT_BLOCK_BYTES = 1024 * 1024

REDSHIFT_BY_SNAPSHOT = {
    55: 3.06042,
    60: 2.51859,
    66: 1.98989,
    71: 1.63027,
    76: 1.32982,
    83: 0.98871,
    88: 0.79075,
    94: 0.59339,
    99: 0.45658,
    100: 0.43183,
    105: 0.31970,
    112: 0.19095,
    118: 0.10235,
}

JIUTIAN_DTYPE = np.dtype(
    [
        ("Pos", "<f4", (3,)),
        ("Vel", "<f4", (3,)),
        ("Vmax", "<f4"),
        ("CentralMvir", "<f4"),
        ("StellarMass", "<f4"),
        ("HImass", "<f4"),
    ]
)


def e_z(z: float, omega_m: float = OMEGA_M, w_value: float = W_VALUE) -> float:
    """Dimensionless Hubble function for flat constant-w cosmology."""
    return float(
        np.sqrt(
            omega_m * (1.0 + z) ** 3
            + (1.0 - omega_m) * (1.0 + z) ** (3.0 * (1.0 + w_value))
        )
    )


def source_sample_sha256(path: Path) -> str:
    """Hash five fixed 1-MiB source blocks and the exact byte size."""
    path = Path(path)
    size = int(path.stat().st_size)
    block = min(SOURCE_FINGERPRINT_BLOCK_BYTES, size)
    maximum_offset = max(size - block, 0)
    offsets = sorted(
        {0, maximum_offset // 4, maximum_offset // 2, 3 * maximum_offset // 4, maximum_offset}
    )
    digest = hashlib.sha256()
    digest.update(f"size={size};block={block};".encode("ascii"))
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            payload = handle.read(block)
            digest.update(f"offset={offset};length={len(payload)};".encode("ascii"))
            digest.update(payload)
    return digest.hexdigest()


@njit(inline="always")
def _deposit_one(
    grid: np.ndarray,
    x: float,
    y: float,
    z: float,
    weight: float,
    box_size: float,
) -> None:
    n_grid = grid.shape[0]
    x -= np.floor(x / box_size) * box_size
    y -= np.floor(y / box_size) * box_size
    z -= np.floor(z / box_size) * box_size
    scale = n_grid / box_size
    gx = x * scale
    gy = y * scale
    gz = z * scale
    ix_floor = np.floor(gx)
    iy_floor = np.floor(gy)
    iz_floor = np.floor(gz)
    ix0 = int(ix_floor) % n_grid
    iy0 = int(iy_floor) % n_grid
    iz0 = int(iz_floor) % n_grid
    ix1 = (ix0 + 1) % n_grid
    iy1 = (iy0 + 1) % n_grid
    iz1 = (iz0 + 1) % n_grid
    fx = gx - ix_floor
    fy = gy - iy_floor
    fz = gz - iz_floor
    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    wz0 = 1.0 - fz
    grid[ix0, iy0, iz0] += weight * wx0 * wy0 * wz0
    grid[ix1, iy0, iz0] += weight * fx * wy0 * wz0
    grid[ix0, iy1, iz0] += weight * wx0 * fy * wz0
    grid[ix0, iy0, iz1] += weight * wx0 * wy0 * fz
    grid[ix1, iy1, iz0] += weight * fx * fy * wz0
    grid[ix1, iy0, iz1] += weight * fx * wy0 * fz
    grid[ix0, iy1, iz1] += weight * wx0 * fy * fz
    grid[ix1, iy1, iz1] += weight * fx * fy * fz


@njit(parallel=True, cache=True)
def _deposit_dual_chunk(
    positions: np.ndarray,
    velocities: np.ndarray,
    weights: np.ndarray,
    real_local: np.ndarray,
    rsd_local: np.ndarray,
    box_size: float,
    los_axis: int,
    rsd_shift_mpc_h_per_kms: float,
) -> None:
    """Deposit one chunk into race-free worker-local real and RSD meshes."""
    n_workers = real_local.shape[0]
    n_values = positions.shape[0]
    for worker in prange(n_workers):
        start = n_values * worker // n_workers
        stop = n_values * (worker + 1) // n_workers
        real_grid = real_local[worker]
        rsd_grid = rsd_local[worker]
        for index in range(start, stop):
            x = float(positions[index, 0])
            y = float(positions[index, 1])
            z = float(positions[index, 2])
            weight = float(weights[index])
            _deposit_one(real_grid, x, y, z, weight, box_size)
            if los_axis == 0:
                x += float(velocities[index, 0]) * rsd_shift_mpc_h_per_kms
            elif los_axis == 1:
                y += float(velocities[index, 1]) * rsd_shift_mpc_h_per_kms
            else:
                z += float(velocities[index, 2]) * rsd_shift_mpc_h_per_kms
            _deposit_one(rsd_grid, x, y, z, weight, box_size)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _field_from_locals(
    local: np.ndarray,
    input_weight_sum: float,
    *,
    conservation_tolerance: float,
) -> tuple[np.ndarray, dict[str, float]]:
    rho = np.sum(local, axis=0, dtype=np.float64)
    deposited = float(np.sum(rho, dtype=np.float64))
    relative_error = abs(deposited - input_weight_sum) / input_weight_sum
    if relative_error > conservation_tolerance:
        raise RuntimeError(
            f"CIC mass conservation failed: {relative_error:.3e} > {conservation_tolerance:.3e}"
        )
    mean = float(np.mean(rho, dtype=np.float64))
    if not np.isfinite(mean) or mean <= 0.0:
        raise RuntimeError("CIC field has a non-positive mean")
    delta = np.asarray(rho / mean - 1.0, dtype=np.float32)
    metadata = {
        "input_weight_sum_native": float(input_weight_sum),
        "deposited_weight_sum_native": deposited,
        "cic_relative_conservation_error": relative_error,
        "rho_mean_native_per_cell": mean,
        "delta_mean": float(np.mean(delta, dtype=np.float64)),
        "delta_std": float(np.std(delta, dtype=np.float64)),
        "delta_min": float(np.min(delta)),
        "delta_max": float(np.max(delta)),
    }
    del rho
    return delta, metadata


def validate_density_file(
    path: Path,
    *,
    expected_snapshot: int | None = None,
    require_complete_catalog: bool = True,
) -> dict[str, Any]:
    """Validate a completed full-HI density artifact without loading it fully."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema_version", "")) != DENSITY_SCHEMA:
            raise ValueError(f"{path}: wrong density schema")
        snapshot = int(handle.attrs["snapshot"])
        if expected_snapshot is not None and snapshot != int(expected_snapshot):
            raise ValueError(f"{path}: snapshot {snapshot} != {expected_snapshot}")
        if require_complete_catalog and not bool(handle.attrs["complete_catalog"]):
            raise ValueError(f"{path}: artifact is a partial-catalog smoke test")
        grid_size = int(handle.attrs["grid_size"])
        expected_shape = (grid_size, grid_size, grid_size)
        for name in ("density_contrast_real", "density_contrast_rsd"):
            if name not in handle or handle[name].shape != expected_shape:
                raise ValueError(f"{path}: missing or malformed {name}")
            if handle[name].dtype != np.dtype("float32"):
                raise ValueError(f"{path}: {name} is not float32")
        metadata = json.loads(str(handle.attrs["validation_json"]))
        for branch in ("real", "rsd"):
            if metadata[branch]["cic_relative_conservation_error"] > float(
                handle.attrs["conservation_tolerance"]
            ):
                raise ValueError(f"{path}: {branch} conservation check failed")
            if abs(metadata[branch]["delta_mean"]) > 2.0e-6:
                raise ValueError(f"{path}: {branch} density contrast is not mean zero")
        return {
            "status": "PASS",
            "path": str(path.resolve()),
            "snapshot": snapshot,
            "redshift": float(handle.attrs["redshift"]),
            "grid_size": grid_size,
            "source_records_processed": int(handle.attrs["source_records_processed"]),
            "positive_hi_records": int(handle.attrs["positive_hi_records"]),
            "complete_catalog": bool(handle.attrs["complete_catalog"]),
            "validation": metadata,
        }


def build_full_hi_density(
    source_path: Path,
    output_path: Path,
    *,
    snapshot: int,
    grid_size: int = 512,
    chunk_records: int = 2_000_000,
    workers: int = 4,
    max_records: int | None = None,
    conservation_tolerance: float = 3.0e-6,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one versioned full-HI real/RSD density file."""
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if snapshot not in REDSHIFT_BY_SNAPSHOT:
        raise ValueError(f"unknown snapshot {snapshot}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.stat().st_size % JIUTIAN_DTYPE.itemsize:
        raise ValueError(
            f"source byte size is not divisible by the audited {JIUTIAN_DTYPE.itemsize}-byte schema"
        )
    if grid_size < 8 or chunk_records < 1 or workers < 1:
        raise ValueError("grid_size>=8, chunk_records>=1, and workers>=1 are required")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive when provided")
    if not 0.0 < conservation_tolerance < 1.0e-3:
        raise ValueError("conservation_tolerance must lie in (0,1e-3)")
    if output_path.exists() and not overwrite:
        result = validate_density_file(
            output_path,
            expected_snapshot=snapshot,
            require_complete_catalog=max_records is None,
        )
        # A valid old file is not necessarily valid for THIS request. In
        # particular, changing a source or grid must never silently reuse it.
        stat = source_path.stat()
        total = stat.st_size // JIUTIAN_DTYPE.itemsize
        processed = total if max_records is None else min(total, max_records)
        redshift = REDSHIFT_BY_SNAPSHOT[snapshot]
        expected = {
            'grid_size': grid_size, 'snapshot': snapshot, 'redshift': redshift,
            'source_path': str(source_path), 'source_size_bytes': stat.st_size,
            'source_mtime_ns': stat.st_mtime_ns,
            'source_sample_sha256': source_sample_sha256(source_path),
            'source_total_records': total, 'source_records_processed': processed,
            'complete_catalog': processed == total,
            'record_itemsize_bytes': JIUTIAN_DTYPE.itemsize,
            'record_schema': 'Pos[3],Vel[3],Vmax,CentralMvir,StellarMass,HImass',
            'selection': 'all_finite_positive_HImass_records', 'weighting': 'native_HImass',
            'box_size_mpc_h': BOX_SIZE_MPC_H, 'los_axis': LOS_AXIS,
            'reference_omega_m': OMEGA_M, 'reference_w': W_VALUE, 'reference_h': HUBBLE_H,
            'velocity_unit_assumption': 'km/s',
            'rsd_shift_mpc_h_per_km_s': (1 + redshift) / (100 * e_z(redshift)),
        }
        with h5py.File(output_path, 'r') as handle:
            mismatches = [key for key, value in expected.items()
                          if key not in handle.attrs or handle.attrs[key] != value]
        if any(result['validation'][branch]['cic_relative_conservation_error'] > conservation_tolerance
               for branch in ('real', 'rsd')):
            mismatches.append('requested_conservation_tolerance')
        if mismatches:
            raise ValueError('Density cache request mismatch: ' + ', '.join(mismatches)
                             + '; use a new output path or explicitly rebuild')
        return result

    source_stat_before = source_path.stat()
    source_fingerprint_before = source_sample_sha256(source_path)
    catalogue = np.memmap(source_path, dtype=JIUTIAN_DTYPE, mode="r")
    total_records = int(catalogue.shape[0])
    records_to_process = total_records if max_records is None else min(total_records, max_records)
    complete_catalog = records_to_process == total_records
    redshift = float(REDSHIFT_BY_SNAPSHOT[snapshot])
    rsd_shift = (1.0 + redshift) / (100.0 * e_z(redshift))

    set_num_threads(min(workers, int(numba.config.NUMBA_DEFAULT_NUM_THREADS)))
    actual_workers = min(workers, int(numba.get_num_threads()))
    local_shape = (actual_workers, grid_size, grid_size, grid_size)
    estimated_local_gib = 2.0 * np.prod(local_shape) * np.dtype(np.float32).itemsize / 1024**3
    real_local = np.zeros(local_shape, dtype=np.float32)
    rsd_local = np.zeros(local_shape, dtype=np.float32)

    # Compile on tiny arrays without touching the production meshes.
    warm_pos = np.zeros((actual_workers, 3), dtype=np.float32)
    warm_vel = np.zeros_like(warm_pos)
    warm_weight = np.ones(actual_workers, dtype=np.float32)
    warm_real = np.zeros((actual_workers, 8, 8, 8), dtype=np.float32)
    warm_rsd = np.zeros_like(warm_real)
    _deposit_dual_chunk(
        warm_pos, warm_vel, warm_weight, warm_real, warm_rsd, BOX_SIZE_MPC_H, LOS_AXIS, rsd_shift
    )

    started = time.time()
    positive_hi_records = 0
    rejected_nonpositive_hi = 0
    rejected_nonfinite = 0
    input_weight_sum = 0.0
    position_min = np.full(3, np.inf)
    position_max = np.full(3, -np.inf)
    velocity_min = np.full(3, np.inf)
    velocity_max = np.full(3, -np.inf)
    chunks = 0

    for start in range(0, records_to_process, chunk_records):
        stop = min(start + chunk_records, records_to_process)
        records = catalogue[start:stop]
        positions = np.ascontiguousarray(records["Pos"], dtype=np.float32)
        velocities = np.ascontiguousarray(records["Vel"], dtype=np.float32)
        weights = np.ascontiguousarray(records["HImass"], dtype=np.float32)
        finite = (
            np.isfinite(weights)
            & np.all(np.isfinite(positions), axis=1)
            & np.all(np.isfinite(velocities), axis=1)
        )
        positive = weights > 0.0
        valid = finite & positive
        rejected_nonfinite += int(np.count_nonzero(~finite))
        rejected_nonpositive_hi += int(np.count_nonzero(finite & ~positive))
        if not np.any(valid):
            chunks += 1
            continue
        positions = np.ascontiguousarray(positions[valid])
        velocities = np.ascontiguousarray(velocities[valid])
        weights = np.ascontiguousarray(weights[valid])
        if np.any(positions < -1.0e-4) or np.any(positions > BOX_SIZE_MPC_H + 1.0e-4):
            raise ValueError(f"source positions leave the declared periodic box in records {start}:{stop}")
        positive_hi_records += int(weights.size)
        input_weight_sum += float(np.sum(weights, dtype=np.float64))
        position_min = np.minimum(position_min, np.min(positions, axis=0))
        position_max = np.maximum(position_max, np.max(positions, axis=0))
        velocity_min = np.minimum(velocity_min, np.min(velocities, axis=0))
        velocity_max = np.maximum(velocity_max, np.max(velocities, axis=0))
        _deposit_dual_chunk(
            positions,
            velocities,
            weights,
            real_local,
            rsd_local,
            BOX_SIZE_MPC_H,
            LOS_AXIS,
            rsd_shift,
        )
        chunks += 1
        if chunks == 1 or stop == records_to_process or chunks % 10 == 0:
            print(
                f"[full-hi-density] snap{snapshot} {stop:,}/{records_to_process:,} "
                f"positive={positive_hi_records:,} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
        del records, positions, velocities, weights, finite, positive, valid

    if positive_hi_records == 0 or input_weight_sum <= 0.0:
        raise RuntimeError("no positive finite HImass records were deposited")
    del catalogue

    real_delta, real_metadata = _field_from_locals(
        real_local, input_weight_sum, conservation_tolerance=conservation_tolerance
    )
    del real_local
    gc.collect()
    rsd_delta, rsd_metadata = _field_from_locals(
        rsd_local, input_weight_sum, conservation_tolerance=conservation_tolerance
    )
    del rsd_local
    gc.collect()

    source_stat_after = source_path.stat()
    source_fingerprint_after = source_sample_sha256(source_path)
    if (
        source_stat_before.st_size != source_stat_after.st_size
        or source_stat_before.st_mtime_ns != source_stat_after.st_mtime_ns
        or source_fingerprint_before != source_fingerprint_after
    ):
        raise RuntimeError("raw source changed while the density field was being built")

    validation = {"real": real_metadata, "rsd": rsd_metadata}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": DENSITY_SCHEMA,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "snapshot": int(snapshot),
                "redshift": redshift,
                "box_size_mpc_h": BOX_SIZE_MPC_H,
                "grid_size": int(grid_size),
                "los_axis": LOS_AXIS,
                "reference_omega_m": OMEGA_M,
                "reference_w": W_VALUE,
                "reference_h": HUBBLE_H,
                "record_schema": "Pos[3],Vel[3],Vmax,CentralMvir,StellarMass,HImass",
                "record_itemsize_bytes": JIUTIAN_DTYPE.itemsize,
                "source_path": str(source_path),
                "source_size_bytes": int(source_stat_after.st_size),
                "source_mtime_ns": int(source_stat_after.st_mtime_ns),
                "source_sample_sha256": source_fingerprint_after,
                "source_total_records": total_records,
                "source_records_processed": records_to_process,
                "complete_catalog": bool(complete_catalog),
                "selection": "all_finite_positive_HImass_records",
                "positive_hi_records": positive_hi_records,
                "rejected_nonpositive_hi_records": rejected_nonpositive_hi,
                "rejected_nonfinite_records": rejected_nonfinite,
                "weighting": "native_HImass",
                "HImass_unit_status": "native_catalogue_unit_unconfirmed",
                "absolute_Omega_HI_or_Tb_allowed": False,
                "velocity_unit_assumption": "km/s",
                "rsd_shift_mpc_h_per_km_s": rsd_shift,
                "chunk_records": int(chunk_records),
                "worker_local_meshes": int(actual_workers),
                "worker_local_mesh_memory_gib": float(estimated_local_gib),
                "conservation_tolerance": float(conservation_tolerance),
                "position_min_mpc_h": position_min,
                "position_max_mpc_h": position_max,
                "velocity_min_km_s": velocity_min,
                "velocity_max_km_s": velocity_max,
                "input_HImass_sum_native": input_weight_sum,
                "validation_json": json.dumps(validation, sort_keys=True),
                "runtime_python": platform.python_version(),
                "runtime_numpy": np.__version__,
                "runtime_numba": numba.__version__,
                "runtime_h5py": h5py.__version__,
                "elapsed_sec": float(time.time() - started),
            }
        )
        hdf5_chunk = tuple(min(64, int(size)) for size in real_delta.shape)
        for name, values in (
            ("density_contrast_real", real_delta),
            ("density_contrast_rsd", rsd_delta),
        ):
            handle.create_dataset(
                name,
                data=values,
                dtype=np.float32,
                chunks=hdf5_chunk,
                compression="lzf",
                shuffle=True,
            )
    temporary.replace(output_path)
    del real_delta, rsd_delta
    result = validate_density_file(
        output_path,
        expected_snapshot=snapshot,
        require_complete_catalog=complete_catalog,
    )
    _atomic_json(output_path.with_suffix(".summary.json"), result)
    return result


__all__ = [
    "BOX_SIZE_MPC_H",
    "DENSITY_SCHEMA",
    "JIUTIAN_DTYPE",
    "REDSHIFT_BY_SNAPSHOT",
    "build_full_hi_density",
    "e_z",
    "source_sample_sha256",
    "validate_density_file",
]
