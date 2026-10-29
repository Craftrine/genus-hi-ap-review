"""Memory-bounded field statistics; no full float64 cube is constructed.

Genus uses the original exact thresholds and is independent of these diagnostic
moments. Shifted float64 moments reduce cancellation for large offsets; a
centered second pass is used whenever the first-pass variance is ill-conditioned.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _shifted_scan(field, shift):
    nx,ny,nz=field.shape
    partial=np.zeros((nx,4),dtype=np.float64)
    valid=np.ones(nx,dtype=np.uint8)
    for i in prange(nx):
        total=0.;square=0.;minimum=np.inf;maximum=-np.inf
        for j in range(ny):
            for k in range(nz):
                value=float(field[i,j,k])
                if not np.isfinite(value):
                    valid[i]=0
                    continue
                delta=value-shift
                total+=delta
                square+=delta*delta
                minimum=min(minimum,value)
                maximum=max(maximum,value)
        partial[i,0]=total;partial[i,1]=square
        partial[i,2]=minimum;partial[i,3]=maximum
    return partial,valid


@njit(parallel=True, cache=True)
def _centered_square_sum(field,mean):
    nx,ny,nz=field.shape
    partial=np.zeros(nx,dtype=np.float64)
    for i in prange(nx):
        total=0.
        for j in range(ny):
            for k in range(nz):
                delta=float(field[i,j,k])-mean
                total+=delta*delta
        partial[i]=total
    return partial


def field_statistics(field):
    """Return finite/min/max/mean/std for the float32 science field.

Mean/std agree to floating-point reduction tolerance, not necessarily bitwise
with NumPy. The input is never mutated. NaN and infinity raise ValueError.
"""
    array=np.ascontiguousarray(field,dtype=np.float32)
    if array.ndim!=3 or array.size==0:
        raise ValueError('nonempty 3D field required')
    shift=float(array.flat[0])
    if not np.isfinite(shift):raise ValueError('field contains nonfinite values')
    partial,valid=_shifted_scan(array,shift)
    if not np.all(valid):raise ValueError('field contains nonfinite values')
    minimum=float(np.min(partial[:,2]));maximum=float(np.max(partial[:,3]))
    mean_delta=float(np.sum(partial[:,0],dtype=np.float64)/array.size)
    moment=float(np.sum(partial[:,1],dtype=np.float64)/array.size)
    mean=shift+mean_delta
    variance=moment-mean_delta*mean_delta
    fallback=minimum!=maximum and (variance<0 or variance<=1e-8*max(moment,np.finfo(float).tiny))
    if minimum==maximum:
        mean=minimum;variance=0.
    elif fallback:
        # NumPy's float64 mean is a reduction, not a float64 copy of the cube.
        mean=float(np.mean(array,dtype=np.float64))
        variance=float(np.sum(_centered_square_sum(array,mean),dtype=np.float64)/array.size)
    return {'field_mean':mean,'field_std':float(np.sqrt(max(variance,0.))),
            'field_min':minimum,'field_max':maximum,'stable_second_pass_used':bool(fallback)}
