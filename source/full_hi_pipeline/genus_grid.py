"""Measure idealized full-HI AP/RSD/wedge genus grids from streamed meshes.

This stage reuses the numerically validated Fourier, threshold, and cubical-
complex implementations from the frozen 2026-08-19 code package.  The real and
RSD observer meshes are deposited once.  Trial cosmologies change the physical
cuboid lengths, Gaussian kernel, and genus-volume normalization; this is exactly
equivalent to coordinate-first re-deposition for the periodic plane-parallel
geometry used here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
from typing import Any

import h5py
import numba
from numba import set_num_threads
import numpy as np
import scipy

from .stream_density import BOX_SIZE_MPC_H, DENSITY_SCHEMA, validate_density_file


GENUS_SCHEMA = "jiutian_full_hi_idealized_genus_grid_v1"
CHECKPOINT_SCHEMA = "jiutian_full_hi_idealized_genus_point_v1"
DEFAULT_CORE_DIR = Path(
    "/Users/LOCAL_USER/Documents/Codex/2026-08-19/"
    "HI_genus_3D_AP_code_package_20260819/pipeline_code"
)
COORDINATES = ("real", "rsd")
FOREGROUNDS = ("nofg", "wedge")
BRANCHES = tuple(f"{coordinate}_{foreground}" for coordinate in COORDINATES for foreground in FOREGROUNDS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_core(core_dir: Path, *, measurement_backend: str = "reference") -> SimpleNamespace:
    if measurement_backend not in ("reference", "fast"):
        raise ValueError("measurement_backend must be reference or fast")
    core_dir = Path(core_dir).expanduser().resolve()
    required = (
        "cosmo_genus.py",
        "factorial_v2/measure.py",
        "genus_amplitude_contract.py",
        "genus_counting_numba.py",
        "run_direct_trial_cic_grid.py",
        "spectral_smoothing.py",
        "volume_fraction_numba.py",
    )
    missing = [name for name in required if not (core_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"validated core is incomplete: {core_dir}; missing={missing}")
    text = str(core_dir)
    if text not in sys.path:
        sys.path.insert(0, text)
    cosmo = importlib.import_module("cosmo_genus")
    amplitude = importlib.import_module("genus_amplitude_contract")
    direct = importlib.import_module("run_direct_trial_cic_grid")
    measure = importlib.import_module("factorial_v2.measure")
    spectral = importlib.import_module("spectral_smoothing")
    core_hashes = {name: sha256_file(core_dir / name) for name in required}
    backend_info = {"name": "reference", "algorithm": "frozen_external_genus_v1", "reference_core_hashes": core_hashes, "source_hashes": {}}
    measure_function = measure.measure_genus
    if measurement_backend == "fast":
        from .fast_measure import FAST_MEASUREMENT_ID, build_measure
        local_dir = Path(__file__).resolve().parent
        backend_info = {
            "name": "fast", "algorithm": FAST_MEASUREMENT_ID,
            "reference_core_hashes": core_hashes,
            "source_hashes": {name: sha256_file(local_dir / name) for name in ("fast_counting.py", "fast_statistics.py", "fast_measure.py")},
        }
        measure_function = build_measure(threshold_fn=cosmo.make_thresholds,
                                         gaussian_fn=amplitude.uniform_gaussian_template_amplitude,
                                         hermite_fn=amplitude.uniform_appleby_hermite_amplitude)
    return SimpleNamespace(
        core_dir=core_dir,
        ap_factors=cosmo.ap_factors,
        make_thresholds=cosmo.make_thresholds,
        compute_curve=cosmo.compute_curve,
        gaussian_amplitude=amplitude.uniform_gaussian_template_amplitude,
        hermite_amplitude=amplitude.uniform_appleby_hermite_amplitude,
        gaussian_contract=amplitude.CONTRACT_ID,
        hermite_contract=amplitude.APPLEBY_HERMITE_CONTRACT_ID,
        physical_wedge_slope_m=direct.physical_wedge_slope_m,
        measure_genus=measure_function,
        measurement_backend=measurement_backend,
        measurement_backend_info=backend_info,
        measurement_backend_signature=_canonical_hash(backend_info),
        prepare_density_spectrum=spectral.prepare_density_spectrum,
        smooth_from_prepared_spectrum=spectral.smooth_from_prepared_spectrum,
        observer_fixed_wedge_mask_rfft=spectral.observer_fixed_wedge_mask_rfft,
        spectral_backend=spectral.SPECTRAL_BACKEND,
        hashes=core_hashes,
    )


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _axis(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite 1D array")
    if result.size > 1 and np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def _trial_lengths(core: SimpleNamespace, redshift: float, omega_m: float, w_value: float) -> np.ndarray:
    alpha_parallel, alpha_perp = core.ap_factors(redshift, omega_m, w_value)
    return np.asarray(
        [alpha_perp * BOX_SIZE_MPC_H, alpha_perp * BOX_SIZE_MPC_H, alpha_parallel * BOX_SIZE_MPC_H],
        dtype=np.float64,
    )


def _checkpoint_path(
    checkpoint_dir: Path,
    coordinate: str,
    role: str,
    i: int,
    j: int,
) -> Path:
    return checkpoint_dir / f"{coordinate}__{role}__i{i:03d}_j{j:03d}.npz"


def _load_checkpoint(
    path: Path,
    *,
    config_hash: str,
    coordinate: str,
    omega_m: float,
    w_value: float,
    scales: np.ndarray,
    nu: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        checks = {
            "schema": str(data["checkpoint_schema"].item()) == CHECKPOINT_SCHEMA,
            "config_hash": str(data["config_hash"].item()) == config_hash,
            "coordinate": str(data["coordinate"].item()) == coordinate,
            "omega_m": float(data["omega_m"].item()) == float(omega_m),
            "w": float(data["w"].item()) == float(w_value),
            "foregrounds": np.array_equal(np.asarray(data["foregrounds"]).astype(str), np.asarray(FOREGROUNDS)),
            "scales": np.array_equal(np.asarray(data["scales_mpc_h"], dtype=float), scales),
            "nu": np.array_equal(np.asarray(data["nu_values"], dtype=float), nu),
        }
        shape = (len(FOREGROUNDS), scales.size, nu.size)
        curves = np.asarray(data["curves"], dtype=np.float64)
        gaussian = np.asarray(data["gaussian"], dtype=np.float64)
        hermite = np.asarray(data["hermite"], dtype=np.float64)
        checks["curve_shape"] = curves.shape == shape and np.all(np.isfinite(curves))
        checks["amplitudes"] = (
            gaussian.shape == shape[:-1]
            and hermite.shape == shape[:-1]
            and np.all(np.isfinite(gaussian))
            and np.all(np.isfinite(hermite))
            and np.all(gaussian > 0.0)
            and np.all(hermite > 0.0)
        )
        if not all(checks.values()):
            failed = [key for key, value in checks.items() if not value]
            raise ValueError(f"{path}: checkpoint validation failed: {failed}")
        return {
            "curves": curves,
            "gaussian": gaussian,
            "hermite": hermite,
            "thresholds": np.asarray(data["thresholds"], dtype=np.float64),
            "actual_fraction": np.asarray(data["actual_volume_fraction_above"], dtype=np.float64),
            "lengths": np.asarray(data["trial_lengths_mpc_h"], dtype=np.float64),
        }


def _measure_checkpoint(
    path: Path,
    *,
    core: SimpleNamespace,
    spectrum: np.ndarray,
    grid_shape: tuple[int, int, int],
    coordinate: str,
    config_hash: str,
    omega_m: float,
    w_value: float,
    redshift: float,
    scales: np.ndarray,
    nu: np.ndarray,
    wedge_mask: np.ndarray,
    threads: int,
) -> dict[str, np.ndarray]:
    if path.is_file():
        return _load_checkpoint(
            path,
            config_hash=config_hash,
            coordinate=coordinate,
            omega_m=omega_m,
            w_value=w_value,
            scales=scales,
            nu=nu,
        )
    lengths = _trial_lengths(core, redshift, omega_m, w_value)
    curves = np.empty((len(FOREGROUNDS), scales.size, nu.size), dtype=np.float64)
    thresholds = np.empty_like(curves)
    actual_fraction = np.empty_like(curves)
    gaussian = np.empty((len(FOREGROUNDS), scales.size), dtype=np.float64)
    hermite = np.empty_like(gaussian)
    started = time.time()
    for foreground_index, foreground in enumerate(FOREGROUNDS):
        remove_mask = None if foreground == "nofg" else wedge_mask
        for scale_index, scale in enumerate(scales):
            smooth = core.smooth_from_prepared_spectrum(
                spectrum,
                grid_shape,
                lengths,
                float(scale),
                remove_mask=remove_mask,
                workers=threads,
            )
            measurement = core.measure_genus(
                smooth,
                trial_lengths_mpc_h=lengths,
                nu_values=nu,
            )
            threshold = np.asarray(measurement["thresholds"], dtype=np.float64)
            curve = np.asarray(measurement["curve"], dtype=np.float64)
            gaussian_value = float(measurement["amplitude_gaussian_signed"])
            hermite_value = float(measurement["amplitude_hermite_signed"])
            if (
                not np.isfinite(gaussian_value)
                or not np.isfinite(hermite_value)
                or gaussian_value <= 0.0
                or hermite_value <= 0.0
            ):
                raise RuntimeError(
                    f"non-positive genus amplitude for {coordinate}/{foreground}, "
                    f"Omega_m={omega_m}, w={w_value}, R={scale}"
                )
            curves[foreground_index, scale_index] = curve
            thresholds[foreground_index, scale_index] = threshold
            actual_fraction[foreground_index, scale_index] = measurement[
                "actual_volume_fraction_above"
            ]
            gaussian[foreground_index, scale_index] = gaussian_value
            hermite[foreground_index, scale_index] = hermite_value
            del smooth, threshold, curve, measurement
            gc.collect()
    _atomic_npz(
        path,
        checkpoint_schema=np.asarray(CHECKPOINT_SCHEMA),
        config_hash=np.asarray(config_hash),
        coordinate=np.asarray(coordinate),
        omega_m=np.asarray(omega_m),
        w=np.asarray(w_value),
        foregrounds=np.asarray(FOREGROUNDS),
        scales_mpc_h=scales,
        nu_values=nu,
        trial_lengths_mpc_h=lengths,
        curves=curves,
        thresholds=thresholds,
        actual_volume_fraction_above=actual_fraction,
        gaussian=gaussian,
        hermite=hermite,
        elapsed_sec=np.asarray(time.time() - started),
        completed=np.asarray(True),
    )
    return _load_checkpoint(
        path,
        config_hash=config_hash,
        coordinate=coordinate,
        omega_m=omega_m,
        w_value=w_value,
        scales=scales,
        nu=nu,
    )


def validate_measurement_config(output_dir: Path, expected_backend: str) -> dict[str, Any]:
    """Read-only resume guard, including legacy reference configurations."""
    if expected_backend not in ("reference", "fast"):
        raise ValueError("expected backend must be reference or fast")
    config_path = Path(output_dir) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in config.items() if key not in ("config_hash", "runtime")}
    if _canonical_hash(payload) != config.get("config_hash"):
        raise ValueError(f"{output_dir}: configuration hash mismatch")
    stored_backend = config.get("measurement_backend", {}).get("name", "reference")
    if stored_backend != expected_backend:
        raise ValueError(f"{output_dir}: measurement backend mismatch; use a new output directory")
    core = _load_core(Path(config["validated_core_dir"]), measurement_backend=expected_backend)
    if config["validated_core_hashes"] != core.hashes:
        raise ValueError(f"{output_dir}: reference core source changed")
    if expected_backend == "fast" and (
        config.get("measurement_backend") != core.measurement_backend_info
        or config.get("measurement_backend_signature") != core.measurement_backend_signature
    ):
        raise ValueError(f"{output_dir}: fast measurement source changed; use a new output directory")
    return config


def validate_genus_file(path: Path, *, expected_snapshot: int | None = None,
                        expected_measurement_backend: str | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if expected_measurement_backend is not None:
            config = validate_measurement_config(path.parent, expected_measurement_backend)
            if str(data["config_hash"].item()) != config["config_hash"]:
                raise ValueError(f"{path}: NPZ/config hash mismatch")
            stored_backend = str(data["measurement_backend"].item()) if "measurement_backend" in data else "reference"
            if stored_backend != expected_measurement_backend:
                raise ValueError(f"{path}: NPZ measurement backend mismatch")
            if stored_backend == "fast" and (
                "measurement_backend_signature" not in data
                or str(data["measurement_backend_signature"].item()) != config["measurement_backend_signature"]
            ):
                raise ValueError(f"{path}: NPZ measurement signature mismatch")
        if str(data["schema_version"].item()) != GENUS_SCHEMA:
            raise ValueError(f"{path}: wrong genus schema")
        snapshot = int(data["snapshot"].item())
        if expected_snapshot is not None and snapshot != expected_snapshot:
            raise ValueError(f"{path}: snapshot mismatch")
        curves = np.asarray(data["genus_curves"], dtype=np.float64)
        gaussian = np.asarray(data["amplitude_gaussian_signed"], dtype=np.float64)
        hermite = np.asarray(data["amplitude_hermite_signed"], dtype=np.float64)
        if (np.any(~np.isfinite(curves)) or np.any(~np.isfinite(gaussian))
            or np.any(~np.isfinite(hermite)) or np.any(gaussian <= 0.0) or np.any(hermite <= 0.0)):
            raise ValueError(f"{path}: non-finite curve or non-positive amplitude")
        return {
            "status": "PASS",
            "path": str(path.resolve()),
            "snapshot": snapshot,
            "redshift": float(data["redshift"].item()),
            "curve_shape": list(curves.shape),
            "gaussian_range": [float(np.min(gaussian)), float(np.max(gaussian))],
            "hermite_range": [float(np.min(hermite)), float(np.max(hermite))],
        }


def run_full_hi_genus_grid(
    density_path: Path,
    output_dir: Path,
    *,
    omega_m_values: np.ndarray,
    w_values: np.ndarray,
    scales_mpc_h: np.ndarray = np.asarray((15.0, 20.0)),
    nu_values: np.ndarray = np.linspace(-2.5, 2.5, 41),
    wedge_theta_deg: float = 30.0,
    wedge_k_intercept_h_mpc: float = 0.0,
    threads: int = 8,
    core_dir: Path = DEFAULT_CORE_DIR,
    measurement_backend: str = "reference",
) -> dict[str, Any]:
    """Run a checkpointed idealized full-HI genus grid for one snapshot."""
    density_info = validate_density_file(density_path, require_complete_catalog=True)
    snapshot = int(density_info["snapshot"])
    redshift = float(density_info["redshift"])
    omega_m = _axis(omega_m_values, "omega_m_values")
    w_axis = _axis(w_values, "w_values")
    scales = _axis(scales_mpc_h, "scales_mpc_h")
    nu = _axis(nu_values, "nu_values")
    # This is a sky angle in sin(theta), so the 90-degree horizon is finite.
    if not 0.0 <= wedge_theta_deg <= 90.0 or wedge_k_intercept_h_mpc < 0.0:
        raise ValueError("invalid wedge parameters")
    if threads < 1:
        raise ValueError("threads must be positive")
    set_num_threads(min(threads, int(numba.config.NUMBA_DEFAULT_NUM_THREADS)))
    threads = min(threads, int(numba.get_num_threads()))
    core = _load_core(core_dir, measurement_backend=measurement_backend)
    density_sha256 = sha256_file(density_path)
    grid_size = int(density_info["grid_size"])
    grid_shape = (grid_size, grid_size, grid_size)
    wedge_slope = float(core.physical_wedge_slope_m(redshift, wedge_theta_deg))
    wedge_mask = core.observer_fixed_wedge_mask_rfft(
        grid_shape,
        2,
        wedge_slope,
        wedge_k_intercept_h_mpc,
        BOX_SIZE_MPC_H,
    )
    wedge_hash = hashlib.sha256(
        np.packbits(wedge_mask, bitorder="little").tobytes()
    ).hexdigest()
    config = {
        "schema_version": GENUS_SCHEMA,
        "density_path": str(Path(density_path).resolve()),
        "density_sha256": density_sha256,
        "snapshot": snapshot,
        "redshift": redshift,
        "grid_shape": list(grid_shape),
        "coordinates": list(COORDINATES),
        "foregrounds": list(FOREGROUNDS),
        "branches": list(BRANCHES),
        "omega_m_values": omega_m.tolist(),
        "w_values": w_axis.tolist(),
        "scales_mpc_h": scales.tolist(),
        "nu_values": nu.tolist(),
        "wedge_policy": "per_snapshot_redshift_fixed_observer_indices",
        "wedge_theta_deg": float(wedge_theta_deg),
        "wedge_slope_m": wedge_slope,
        "wedge_k_intercept_h_mpc": float(wedge_k_intercept_h_mpc),
        "wedge_mask_sha256_packbits_little": wedge_hash,
        "cic_deconvolution": True,
        "spectral_backend": core.spectral_backend,
        "gaussian_amplitude_contract": core.gaussian_contract,
        "hermite_amplitude_contract": core.hermite_contract,
        "validated_core_dir": str(core.core_dir),
        "validated_core_hashes": core.hashes,
        "physical_scope": "dimensionless noiseless full-HI field; no beam/noise/absolute Tb",
        "operation_order": [
            "streamed_full_HI_observer_mesh",
            "real_or_fiducial_RSD",
            "rFFT_and_CIC_deconvolution",
            "per_redshift_observer_wedge_or_identity",
            "trial_physical_Gaussian_smoothing",
            "volume_fraction_thresholds",
            "periodic_cubical_complex_genus_per_trial_volume",
        ],
    }
    # Preserve legacy reference config hashes; only the opt-in fast operator
    # adds implementation identity, preventing either backend from reusing the other's files.
    if measurement_backend != "reference":
        config["measurement_backend"] = core.measurement_backend_info
        config["measurement_backend_signature"] = core.measurement_backend_signature
    config_hash = _canonical_hash(config)
    config["config_hash"] = config_hash
    config["runtime"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "threads": threads,
    }
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    if config_path.is_file():
        existing = validate_measurement_config(output_dir, measurement_backend)
        if existing.get("config_hash") != config_hash:
            raise ValueError(f"{output_dir}: existing output belongs to another configuration")
    else:
        if ((output_dir / f"full_hi_idealized_genus_snap{snapshot}.npz").exists()
            or any((output_dir / "checkpoints").glob("*.npz"))):
            raise ValueError(f"{output_dir}: existing result lacks its configuration; use a new output directory")
        _atomic_json(config_path, config)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    exact_omega_m = 0.3111
    exact_w = -1.0
    point_paths: dict[tuple[str, int, int], Path] = {}
    exact_paths: dict[str, Path] = {}
    started = time.time()
    with h5py.File(density_path, "r") as density_file:
        for coordinate in COORDINATES:
            density = np.asarray(density_file[f"density_contrast_{coordinate}"], dtype=np.float32)
            spectrum = core.prepare_density_spectrum(
                density,
                deconvolve_cic=True,
                los_axis=2,
                workers=threads,
            )
            spectrum[0, 0, 0] = 0.0
            del density
            gc.collect()
            exact_path = _checkpoint_path(checkpoint_dir, coordinate, "exact", -1, -1)
            _measure_checkpoint(
                exact_path,
                core=core,
                spectrum=spectrum,
                grid_shape=grid_shape,
                coordinate=coordinate,
                config_hash=config_hash,
                omega_m=exact_omega_m,
                w_value=exact_w,
                redshift=redshift,
                scales=scales,
                nu=nu,
                wedge_mask=wedge_mask,
                threads=threads,
            )
            exact_paths[coordinate] = exact_path
            for i, trial_omega_m in enumerate(omega_m):
                for j, trial_w in enumerate(w_axis):
                    if trial_omega_m == exact_omega_m and trial_w == exact_w:
                        point_paths[(coordinate, i, j)] = exact_path
                    else:
                        path = _checkpoint_path(checkpoint_dir, coordinate, "trial", i, j)
                        _measure_checkpoint(
                            path,
                            core=core,
                            spectrum=spectrum,
                            grid_shape=grid_shape,
                            coordinate=coordinate,
                            config_hash=config_hash,
                            omega_m=float(trial_omega_m),
                            w_value=float(trial_w),
                            redshift=redshift,
                            scales=scales,
                            nu=nu,
                            wedge_mask=wedge_mask,
                            threads=threads,
                        )
                        point_paths[(coordinate, i, j)] = path
                    print(
                        f"[full-hi-genus] snap{snapshot} {coordinate} "
                        f"Om={trial_omega_m:.5f} w={trial_w:.5f} "
                        f"elapsed={time.time()-started:.1f}s",
                        flush=True,
                    )
            del spectrum
            gc.collect()

    shape = (len(COORDINATES), len(FOREGROUNDS), scales.size, omega_m.size, w_axis.size, nu.size)
    curves = np.empty(shape, dtype=np.float64)
    thresholds = np.empty_like(curves)
    actual_fraction = np.empty_like(curves)
    gaussian = np.empty(shape[:-1], dtype=np.float64)
    hermite = np.empty_like(gaussian)
    exact_curves = np.empty((len(COORDINATES), len(FOREGROUNDS), scales.size, nu.size))
    exact_gaussian = np.empty(exact_curves.shape[:-1])
    exact_hermite = np.empty_like(exact_gaussian)
    for coordinate_index, coordinate in enumerate(COORDINATES):
        exact = _load_checkpoint(
            exact_paths[coordinate],
            config_hash=config_hash,
            coordinate=coordinate,
            omega_m=exact_omega_m,
            w_value=exact_w,
            scales=scales,
            nu=nu,
        )
        exact_curves[coordinate_index] = exact["curves"]
        exact_gaussian[coordinate_index] = exact["gaussian"]
        exact_hermite[coordinate_index] = exact["hermite"]
        for i, trial_omega_m in enumerate(omega_m):
            for j, trial_w in enumerate(w_axis):
                item = _load_checkpoint(
                    point_paths[(coordinate, i, j)],
                    config_hash=config_hash,
                    coordinate=coordinate,
                    omega_m=float(trial_omega_m),
                    w_value=float(trial_w),
                    scales=scales,
                    nu=nu,
                )
                curves[coordinate_index, :, :, i, j] = item["curves"]
                thresholds[coordinate_index, :, :, i, j] = item["thresholds"]
                actual_fraction[coordinate_index, :, :, i, j] = item["actual_fraction"]
                gaussian[coordinate_index, :, :, i, j] = item["gaussian"]
                hermite[coordinate_index, :, :, i, j] = item["hermite"]

    final_path = output_dir / f"full_hi_idealized_genus_snap{snapshot}.npz"
    implementation_metadata = {} if measurement_backend == "reference" else {
        "measurement_backend": np.asarray(measurement_backend),
        "measurement_backend_signature": np.asarray(core.measurement_backend_signature),
    }
    _atomic_npz(
        final_path,
        **implementation_metadata,
        schema_version=np.asarray(GENUS_SCHEMA),
        generated_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
        config_hash=np.asarray(config_hash),
        density_path=np.asarray(str(Path(density_path).resolve())),
        density_sha256=np.asarray(density_sha256),
        snapshot=np.asarray(snapshot),
        redshift=np.asarray(redshift),
        coordinate_labels=np.asarray(COORDINATES),
        foreground_labels=np.asarray(FOREGROUNDS),
        flat_branch_labels=np.asarray(BRANCHES),
        omega_m_values=omega_m,
        w_values=w_axis,
        scales_mpc_h=scales,
        nu_values=nu,
        genus_curves=curves,
        thresholds=thresholds,
        actual_volume_fraction_above=actual_fraction,
        amplitude_gaussian_signed=gaussian,
        amplitude_hermite_signed=hermite,
        exact_fiducial_genus_curves=exact_curves,
        exact_fiducial_amplitude_gaussian_signed=exact_gaussian,
        exact_fiducial_amplitude_hermite_signed=exact_hermite,
        wedge_slope_m=np.asarray(wedge_slope),
        wedge_mask_sha256_packbits_little=np.asarray(wedge_hash),
        physical_scope=np.asarray(config["physical_scope"]),
        elapsed_sec=np.asarray(time.time() - started),
    )
    result = validate_genus_file(final_path, expected_snapshot=snapshot,
                                 expected_measurement_backend=measurement_backend)
    _atomic_json(output_dir / "summary.json", result)
    return result


__all__ = [
    "BRANCHES",
    "CHECKPOINT_SCHEMA",
    "DEFAULT_CORE_DIR",
    "GENUS_SCHEMA",
    "run_full_hi_genus_grid",
    "validate_genus_file",
]
