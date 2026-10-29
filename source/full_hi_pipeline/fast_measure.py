"""Opt-in accelerated implementation of the unchanged periodic genus statistic."""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

from .fast_statistics import field_statistics

FAST_MEASUREMENT_ID='cpu_exact_rank_histogram_shifted_statistics_v1'


def build_measure(*,threshold_fn,gaussian_fn,hermite_fn):
    """Bind frozen quantile/compression routines without importing global core state."""
    from .fast_counting import compute_curve_and_fractions_fast

    def measure_genus(smoothed_field,*,trial_lengths_mpc_h,nu_values):
        field=np.ascontiguousarray(smoothed_field,dtype=np.float32)
        nu=np.asarray(nu_values,dtype=np.float64)
        lengths=np.asarray(trial_lengths_mpc_h,dtype=np.float64)
        if lengths.shape!=(3,) or not np.isfinite(lengths).all() or np.any(lengths<=0):
            raise ValueError('positive finite 3-vector physical lengths required')
        statistics=field_statistics(field)
        thresholds=threshold_fn(field,nu,convention='volume-fraction')
        volume=float(np.prod(lengths))
        curve,actual_fraction=compute_curve_and_fractions_fast(field,thresholds,volume,periodic=True)
        gaussian=float(gaussian_fn(curve,nu));hermite=float(hermite_fn(curve,nu))
        if not np.isfinite(curve).all() or not np.isfinite([gaussian,hermite]).all():
            raise RuntimeError('accelerated measurement returned nonfinite genus')
        return {'thresholds':np.asarray(thresholds,dtype=np.float64),'curve':np.asarray(curve,dtype=np.float64),
                'amplitude_gaussian_signed':gaussian,'amplitude_hermite_signed':hermite,
                'target_volume_fraction_above':.5*erfc(nu/np.sqrt(2)),
                'actual_volume_fraction_above':np.asarray(actual_fraction,dtype=np.float64),
                'field_mean':statistics['field_mean'],'field_std':statistics['field_std'],
                'field_min':statistics['field_min'],'field_max':statistics['field_max'],
                'physical_volume_mpc_h3':volume}
    return measure_genus
