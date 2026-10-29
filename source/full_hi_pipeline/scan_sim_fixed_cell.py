"""Fixed-cell cropped scan geometry, conservative edge CIC and open -Euler.

No periodic topology and no AP transport of the foreground mask. Fourier
filtering uses explicit zero extension of density contrast. Distances: Mpc/h.
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange
from scipy.integrate import quad
from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq, next_fast_len
from scipy.special import ndtr

OMEGA_M_SIM = .3111
W_SIM = -1.
H_SIM = .6766
M_SCAN = np.linspace(.2938445235861259, 1.6921831690611855, 5)
LABELS = ['nofg', 'horizontal_only'] + [f'horizontal_m{i+1}' for i in range(5)]


def expansion(z, omega_m, w):
    return np.sqrt(omega_m*(1+z)**3+(1-omega_m)*(1+z)**(3*(1+w)))


def ap_scan(z, omega_m_scan, w_scan):
    distance_sim = quad(lambda t: 1/expansion(t, OMEGA_M_SIM, W_SIM), 0, z)[0]
    distance_scan = quad(lambda t: 1/expansion(t, omega_m_scan, w_scan), 0, z)[0]
    transverse = distance_scan/distance_sim if z else 1.
    return np.array([transverse, transverse,
                     expansion(z, OMEGA_M_SIM, W_SIM)/expansion(z, omega_m_scan, w_scan)])


def add_rsd_sim(positions_sim, velocities_sim, z):
    positions_rsd_sim = np.array(positions_sim, dtype=np.float64, copy=True)
    positions_rsd_sim[:, 2] += velocities_sim[:, 2]*(1+z)/(100*expansion(z, OMEGA_M_SIM, W_SIM))
    return positions_rsd_sim


def crop_geometry(positions_rsd_sim, weights, factors, cell_size_target, box_length_sim=1000.):
    if cell_size_target <= 0 or np.any(np.asarray(factors) <= 0):
        raise ValueError('Positive spacing and AP factors required')
    box_lengths_scan = box_length_sim*np.asarray(factors)
    shape = np.floor(box_lengths_scan/cell_size_target).astype(int)
    if np.any(shape < 2):
        raise ValueError('Fewer than two complete cells')
    retained_lengths_scan = shape*cell_size_target
    # Wrap ONLY in full transformed simulation box, before cropping.
    positions_rsd_scan = np.mod(positions_rsd_sim*factors, box_lengths_scan)
    keep = np.all(positions_rsd_scan < retained_lengths_scan, axis=1)
    metadata = dict(box_lengths_scan=box_lengths_scan.tolist(),
        retained_lengths_scan=retained_lengths_scan.tolist(), grid_shape_scan=shape.tolist(),
        discarded_widths_scan=(box_lengths_scan-retained_lengths_scan).tolist(),
        cell_size_target=float(cell_size_target), object_count_before=len(weights),
        object_count_after=int(keep.sum()), weight_before=float(weights.sum()),
        weight_after=float(weights[keep].sum()),
        retained_volume_scan=float(np.prod(retained_lengths_scan)))
    return np.ascontiguousarray(positions_rsd_scan[keep]), np.ascontiguousarray(weights[keep]), metadata


@njit(cache=True)
def _cic(positions, weights, spacing, mass):
    for p in range(len(weights)):
        base = np.floor(positions[p]/spacing-.5).astype(np.int64)
        frac = positions[p]/spacing-.5-base
        norm = 0.
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    i,j,k=base[0]+a,base[1]+b,base[2]+c
                    if 0<=i<mass.shape[0] and 0<=j<mass.shape[1] and 0<=k<mass.shape[2]:
                        norm += (frac[0] if a else 1-frac[0])*(frac[1] if b else 1-frac[1])*(frac[2] if c else 1-frac[2])
        if norm <= 0:
            raise ValueError('Object outside CIC domain')
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    i,j,k=base[0]+a,base[1]+b,base[2]+c
                    if 0<=i<mass.shape[0] and 0<=j<mass.shape[1] and 0<=k<mass.shape[2]:
                        mass[i,j,k] += weights[p]/norm*(frac[0] if a else 1-frac[0])*(frac[1] if b else 1-frac[1])*(frac[2] if c else 1-frac[2])


def deposit(positions, weights, shape, spacing):
    if not np.isfinite(positions).all() or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError('Invalid objects')
    if np.any(positions<0) or np.any(positions>=np.asarray(shape)*spacing):
        raise ValueError('Crop objects before deposition')
    mass=np.zeros(tuple(shape),np.float64)
    _cic(positions, weights, spacing, mass)
    error=abs(mass.sum()-weights.sum())/weights.sum()
    if error>1e-12:
        raise ArithmeticError('CIC mass conservation failure')
    delta=(mass/(weights.sum()/mass.size)-1).astype(np.float32)
    return delta, dict(relative_mass_error=float(error), delta_mean=float(delta.mean(dtype=np.float64)))


def prepare_spectrum(delta, spacing, padding_cells):
    shape_fft=tuple(next_fast_len(n+2*padding_cells, real=True) for n in delta.shape)
    slices=tuple(slice((m-n)//2,(m-n)//2+n) for m,n in zip(shape_fft,delta.shape))
    padded=np.zeros(shape_fft,np.float32); padded[slices]=delta
    spectrum=rfftn(padded,workers=1); del padded
    axes=(2*np.pi*fftfreq(shape_fft[0],spacing),2*np.pi*fftfreq(shape_fft[1],spacing),
          2*np.pi*rfftfreq(shape_fft[2],spacing))
    for axis,k in enumerate(axes):
        view=[1,1,1];view[axis]=len(k)
        spectrum /= (np.sinc(k*spacing/(2*np.pi))**2).astype(np.float32).reshape(view)
    spectrum[0,0,0]=0
    return spectrum,axes,shape_fft,slices


def removal_mask(axes, branch, k_parallel_cut=.1):
    if branch==0:
        return None
    kx,ky,kz=axes
    low=np.abs(kz)[None,None,:]<k_parallel_cut
    if branch==1:
        return low
    kp=np.sqrt(kx[:,None]**2+ky[None,:]**2)
    return low | (np.abs(kz)[None,None,:] <= M_SCAN[branch-2]*kp[:,:,None])


def filtered_field(spectrum, axes, shape_fft, slices, radius, branch):
    work=spectrum.copy()
    for axis,k in enumerate(axes):
        view=[1,1,1];view[axis]=len(k)
        work *= np.exp(-.5*(radius*k)**2).astype(np.float32).reshape(view)
    mask=removal_mask(axes,branch)
    if mask is not None:
        work *= ~mask
    work[0,0,0]=0
    padded=irfftn(work,s=shape_fft,workers=1)
    return np.ascontiguousarray(padded[slices])


@njit(cache=True,parallel=True)
def _raw_histogram(ranks, nt):
    nx,ny,nz=ranks.shape
    hist=np.zeros((nx,nt+1),np.int64)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                a=ranks[i,j,k];hist[i,a]-=1
                if i+1<nx: hist[i,min(a,ranks[i+1,j,k])]+=1
                if j+1<ny: hist[i,min(a,ranks[i,j+1,k])]+=1
                if k+1<nz: hist[i,min(a,ranks[i,j,k+1])]+=1
                if i+1<nx and j+1<ny:
                    hist[i,min(a,ranks[i+1,j,k],ranks[i,j+1,k],ranks[i+1,j+1,k])]-=1
                if i+1<nx and k+1<nz:
                    hist[i,min(a,ranks[i+1,j,k],ranks[i,j,k+1],ranks[i+1,j,k+1])]-=1
                if j+1<ny and k+1<nz:
                    hist[i,min(a,ranks[i,j+1,k],ranks[i,j,k+1],ranks[i,j+1,k+1])]-=1
                if i+1<nx and j+1<ny and k+1<nz:
                    hist[i,min(a,ranks[i+1,j,k],ranks[i,j+1,k],ranks[i,j,k+1],
                        ranks[i+1,j+1,k],ranks[i+1,j,k+1],ranks[i,j+1,k+1],ranks[i+1,j+1,k+1])]+=1
    return hist.sum(axis=0)


def negative_euler(field, thresholds):
    from .fast_counting import _encode_ranks
    ranks=np.empty(field.shape,np.uint8)
    if len(thresholds)>254 or _encode_ranks(field,thresholds,ranks):
        raise ValueError('Invalid field or too many thresholds')
    hist=_raw_histogram(ranks,len(thresholds))
    return np.cumsum(hist[::-1],dtype=np.int64)[::-1][1:]


def measure(field, spacing, nu):
    if not np.isfinite(field).all() or np.ptp(field)==0:
        raise ValueError('Invalid or degenerate field')
    thresholds=np.quantile(field,ndtr(nu),method='linear')
    counts=negative_euler(field,thresholds)
    curve=counts/(field.size*spacing**3)
    use=np.abs(nu)<=2.5
    x=nu[use]; env=np.exp(-x*x/2); template=(1-x*x)*env
    amplitude=float(curve[use]@template/(template@template))
    design=np.array([template,env,x*env,(x**3-3*x)*env,(x**4-6*x*x+3)*env]).T
    hermite=float(np.linalg.lstsq(design,curve[use],rcond=None)[0][0])
    return dict(thresholds=thresholds,negative_euler_counts=counts,curve=curve,
        amplitude_gaussian=amplitude,amplitude_hermite=hermite,
        field_mean=float(field.mean(dtype=np.float64)),field_std=float(field.std(dtype=np.float64)))
