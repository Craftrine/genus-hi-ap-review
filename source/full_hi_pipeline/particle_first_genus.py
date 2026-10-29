"""Trial CIC field -> Gaussian smoothing -> optional wedge -> thresholds/genus.

This module receives an already AP-transformed, newly deposited density file.
It never changes that file's physical lengths and never loads a fiducial mesh.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.fft import fftfreq, rfftfreq, rfftn, irfftn

from .genus_grid import DEFAULT_CORE_DIR, _load_core, _canonical_hash, _atomic_json, _atomic_npz, sha256_file
from .particle_first_coordinates import distance_integral, expansion_e
from .particle_first_density import validate_density, output_lock
from .stream_density import OMEGA_M, W_VALUE

GENUS_SCHEMA = 'jiutian_particle_first_smooth_then_wedge_genus_v2'


def _lengths(lengths):
    lengths = np.asarray(lengths, dtype=np.float64)
    if lengths.shape != (3,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Three positive finite box lengths are required')
    return lengths


def physical_wave_numbers(shape, lengths):
    lengths = _lengths(lengths)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError('Invalid field shape')
    return (2 * np.pi * fftfreq(shape[0], d=lengths[0] / shape[0]),
            2 * np.pi * fftfreq(shape[1], d=lengths[1] / shape[1]),
            2 * np.pi * rfftfreq(shape[2], d=lengths[2] / shape[2]))


def prepare_trial_spectrum(delta, *, deconvolve_cic=True, threads=2):
    """Step 6a: rFFT of the NEW trial field; optional CIC-window division."""
    delta = np.asarray(delta, dtype=np.float32)
    if delta.ndim != 3 or not np.isfinite(delta).all():
        raise ValueError('Expected a finite 3D trial density field')
    spectrum = rfftn(delta, workers=threads)
    if deconvolve_cic:
        # k_i*Delta_i/(2*pi) = integer Fourier index / N_i on THIS trial grid.
        freq = (fftfreq(delta.shape[0]), fftfreq(delta.shape[1]), rfftfreq(delta.shape[2]))
        for axis in range(3):
            view = [1, 1, 1]
            view[axis] = len(freq[axis])
            spectrum /= (np.sinc(freq[axis])**2).astype(np.float32).reshape(view)
    spectrum[0, 0, 0] = 0
    return np.asarray(spectrum, dtype=np.complex64)


def observer_wedge_mask(shape, observer_lengths, redshift, theta_deg, *,
                         los_axis=2, k_intercept=0.):
    """Step 7: define the same observer-coordinate wedge for every AP trial.

    None means noFG; 0 degrees explicitly removes k_parallel=0. The grid
    discretization can differ in fixed-cell mode, but the observer-space
    geometric boundary is unchanged. theta is a sky angle, not a k-plane angle.
    """
    if los_axis not in (0, 1, 2) or not np.isfinite(k_intercept) or k_intercept < 0:
        raise ValueError('Invalid wedge intercept or LOS axis')
    if theta_deg is None:
        return None, 0.
    if not np.isfinite(theta_deg) or not 0 <= theta_deg <= 90:
        raise ValueError('Wedge sky angle must be in [0,90]')
    slope = float(np.sin(np.deg2rad(theta_deg)) * expansion_e(redshift, OMEGA_M, W_VALUE)
                  * distance_integral(redshift, OMEGA_M, W_VALUE) / (1 + redshift))
    axes = physical_wave_numbers(shape, observer_lengths)
    k = np.meshgrid(*axes, indexing='ij', sparse=True)
    transverse2 = sum(k[axis]**2 for axis in range(3) if axis != los_axis)
    mask = np.broadcast_to(abs(k[los_axis]) <= slope * np.sqrt(transverse2) + k_intercept,
                           (shape[0], shape[1], shape[2] // 2 + 1)).copy()
    return mask, slope


def filter_spectrum(spectrum, remove_mask):
    filtered = np.array(spectrum, dtype=np.complex64, copy=True)
    if remove_mask is not None:
        if remove_mask.shape != filtered.shape or remove_mask.dtype != bool:
            raise ValueError('Wedge mask shape/dtype mismatch')
        filtered[remove_mask] = 0
    filtered[0, 0, 0] = 0
    return filtered


def smooth_trial_spectrum(spectrum, shape, trial_lengths, radius_mpc_h):
    """Step 6: apply the fixed comoving Gaussian kernel BEFORE choosing wedge."""
    if not np.isfinite(radius_mpc_h) or radius_mpc_h <= 0:
        raise ValueError('Smoothing radius must be positive and finite')
    work = np.array(spectrum, dtype=np.complex64, copy=True)
    if work.shape != (shape[0], shape[1], shape[2] // 2 + 1):
        raise ValueError('Spectrum shape mismatch')
    axes = physical_wave_numbers(shape, trial_lengths)
    for axis in range(3):
        view = [1, 1, 1]
        view[axis] = len(axes[axis])
        kernel = np.exp(-.5 * radius_mpc_h**2 * axes[axis]**2)
        work *= kernel.astype(np.float32).reshape(view)
    return work


def inverse_trial_field(spectrum, shape, *, threads=2):
    return np.asarray(irfftn(spectrum, s=shape, workers=threads), dtype=np.float32)


def gaussian_smooth(spectrum, shape, trial_lengths, radius_mpc_h, *, threads=2):
    """Convenience helper for analytic tests; production branches after smoothing."""
    work = smooth_trial_spectrum(spectrum, shape, trial_lengths, radius_mpc_h)
    return inverse_trial_field(work, shape, threads=threads)


def measure_thresholds_and_genus(smoothed, trial_lengths, nu, core):
    """Step 8: new volume-fraction thresholds, periodic Euler genus, signed A."""
    return core.measure_genus(smoothed, trial_lengths_mpc_h=np.asarray(trial_lengths),
                              nu_values=np.asarray(nu, dtype=np.float64))


def _payload_hash(payload):
    digest = hashlib.sha256()
    for key in sorted(payload):
        a = np.asarray(payload[key])
        digest.update(key.encode())
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _load_result(path, config):
    with np.load(path, allow_pickle=False) as f:
        payload = {key: f[key].copy() for key in f.files if key != 'payload_sha256'}
        if str(f['config_sha256']) != _canonical_hash(config):
            raise ValueError('Genus cache request mismatch; use a new output directory')
        if str(f['payload_sha256']) != _payload_hash(payload):
            raise ValueError('Genus content checksum mismatch')
    return payload


def load_measurement_core(core_dir=DEFAULT_CORE_DIR):
    core = _load_core(core_dir, measurement_backend='fast')
    # Reject a different already-imported core rather than misattribute its hashes.
    import sys
    for name in ('cosmo_genus', 'genus_amplitude_contract', 'genus_counting_numba', 'volume_fraction_numba'):
        if Path(sys.modules[name].__file__).resolve().parent != Path(core_dir).resolve():
            raise RuntimeError('A different numerical core is already loaded: ' + name)
    return core


def measure_trial_density(density_path: Path, output_path: Path, *, scales=(15., 20.),
                          angles=(None, 10., 30., 90.), nu_points=41, threads=2,
                          deconvolve_cic=True, wedge_k_intercept=0.,
                          core_dir: Path = DEFAULT_CORE_DIR) -> dict:
    scales = np.asarray(scales, dtype=float)
    if scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Invalid smoothing scales')
    if len(set(scales.tolist())) != len(scales):
        raise ValueError('Duplicate smoothing scales')
    if isinstance(nu_points, bool) or int(nu_points) != nu_points or nu_points < 5:
        raise ValueError('nu_points must be an integer >=5')
    if isinstance(threads, bool) or int(threads) != threads or threads < 1:
        raise ValueError('threads must be a positive integer')
    if not np.isfinite(wedge_k_intercept) or wedge_k_intercept < 0:
        raise ValueError('Invalid observer wedge intercept')
    angles = [None if a is None else float(a) for a in angles]
    if not angles or len(set(angles)) != len(angles) or any(a is not None and (not np.isfinite(a) or not 0 <= a <= 90) for a in angles):
        raise ValueError('Angles must be unique, nonempty, and in [0,90], or None for noFG')
    info = validate_density(density_path)
    density_config = info['config']
    core = load_measurement_core(core_dir)
    from numba import set_num_threads
    set_num_threads(threads)
    nu = np.linspace(-2.5, 2.5, int(nu_points))
    labels = ['nofg' if a is None else f'theta{a:g}' for a in angles]
    config = {'schema': GENUS_SCHEMA, 'density_config_sha256': info['config_sha256'],
              'density_delta_sha256': info['delta_sha256'], 'source_complete': density_config['source']['complete_catalog'],
              'source_kind': density_config['source_kind'],
              'scales_mpc_h': scales.tolist(), 'angles_deg': angles, 'nu': nu.tolist(),
              'deconvolve_cic': bool(deconvolve_cic), 'wedge_k_intercept_observer_h_mpc': float(wedge_k_intercept),
              'wedge_policy': 'same_geometric_boundary_in_fiducial_observer_coordinates',
              'trial_lengths_mpc_h': density_config['grid']['trial_lengths_mpc_h'],
              'threads': int(threads), 'measurement_backend': core.measurement_backend_info,
              'measurement_source_sha256': sha256_file(Path(__file__)),
              'step_order': density_config['operation_order'] + ['trial_rFFT_and_CIC_window_correction',
                 'fixed_comoving_Gaussian_smoothing_in_Fourier', 'optional_observer_wedge_filter', 'inverse_rFFT',
                 'new_volume_fraction_thresholds', 'periodic_Euler_genus_per_trial_volume', 'signed_amplitudes']}
    output_path = Path(output_path).resolve()
    if output_path == Path(density_path).resolve():
        raise ValueError('Genus output may not overwrite the density input')
    with output_lock(output_path):
        if output_path.exists():
            payload = _load_result(output_path, config)
            reused = True
        else:
            with h5py.File(density_path, 'r') as h:
                delta = np.asarray(h['density_contrast_trial'])
            shape = delta.shape
            lengths = np.array(density_config['grid']['trial_lengths_mpc_h'])
            spectrum = prepare_trial_spectrum(delta, deconvolve_cic=deconvolve_cic, threads=threads)
            del delta
            observer_lengths = lengths / np.array(density_config['ap_factors_xyz'])
            curve_shape = (len(angles), len(scales), len(nu))
            curves, thresholds, fractions = (np.empty(curve_shape) for _ in range(3))
            gaussian, hermite = (np.empty(curve_shape[:2]) for _ in range(2))
            statistics = [[None for _ in scales] for _ in angles]
            masks = [None for _ in angles]
            for ri, radius in enumerate(scales):
                # First smooth the newly deposited trial field in Fourier space.
                smoothed_spectrum = smooth_trial_spectrum(spectrum, shape, lengths, radius)
                for ci, angle in enumerate(angles):
                    # Then choose noFG or wedge; both start from that same smoothed spectrum.
                    mask, slope = observer_wedge_mask(shape, observer_lengths, density_config['redshift'], angle,
                                                      k_intercept=wedge_k_intercept)
                    filtered = filter_spectrum(smoothed_spectrum, mask)
                    field = inverse_trial_field(filtered, shape, threads=threads)
                    row = measure_thresholds_and_genus(field, lengths, nu, core)
                    curves[ci, ri] = row['curve']
                    thresholds[ci, ri] = row['thresholds']
                    fractions[ci, ri] = row['actual_volume_fraction_above']
                    gaussian[ci, ri] = row['amplitude_gaussian_signed']
                    hermite[ci, ri] = row['amplitude_hermite_signed']
                    statistics[ci][ri] = {key: row[key] for key in ['field_mean', 'field_std', 'field_min', 'field_max']}
                    if ri == 0:
                        masks[ci] = {'theta_deg': angle, 'observer_slope': slope,
                                     'mask_sha256_packbits_little': None if mask is None else hashlib.sha256(np.packbits(mask, bitorder='little').tobytes()).hexdigest()}
                    del field, filtered, mask, row
                del smoothed_spectrum
            payload = {'config_json': np.array(json.dumps(config, sort_keys=True)), 'config_sha256': np.array(_canonical_hash(config)),
                       'labels': np.array(labels), 'scales_mpc_h': scales, 'nu': nu,
                       'trial_lengths_mpc_h': lengths, 'grid_shape': np.array(shape),
                       'omega_m': np.array(density_config['trial_cosmology']['omega_m']),
                       'w': np.array(density_config['trial_cosmology']['w']), 'redshift': np.array(density_config['redshift']),
                       'snapshot': np.array(density_config['snapshot']), 'complete_catalog': np.array(config['source_complete']),
                       'curves': np.array(curves), 'thresholds': np.array(thresholds),
                       'actual_volume_fraction_above': np.array(fractions),
                       'amplitude_gaussian': np.array(gaussian), 'amplitude_hermite': np.array(hermite),
                       'field_statistics_json': np.array(json.dumps(statistics)), 'mask_metadata_json': np.array(json.dumps(masks))}
            _atomic_npz(output_path, **payload, payload_sha256=np.array(_payload_hash(payload)))
            reused = False
        summary = {'status': 'MEASURED', 'path': str(output_path), 'cache_reused': reused,
                   'complete_catalog': config['source_complete'], 'source_kind': config['source_kind'], 'config': config,
                   'curve_shape': list(payload['curves'].shape), 'labels': labels,
                   'amplitude_gaussian': payload['amplitude_gaussian'].tolist(),
                   'amplitude_hermite': payload['amplitude_hermite'].tolist(),
                   'scope': 'Literal particle-first AP; fixed simulation velocities/growth. No beam, thermal noise or cosmological coverage calibration.'}
        _atomic_json(output_path.with_suffix('.summary.json'), summary)
        return summary
