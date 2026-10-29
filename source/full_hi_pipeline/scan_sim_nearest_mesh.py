"""Full AP box, per-axis integer mesh nearest target spacing, no cropping.

Preserves the original open-Euler estimator and scan-fixed foreground masks.
Only the mesh geometry, anisotropic cell spacing and volume handling differ.
"""
import numpy as np
from scipy.fft import rfftn,fftfreq,rfftfreq,next_fast_len
from . import scan_sim_fixed_cell as old

def geometry(positions_rsd,weights,factors,target=4.,box=1000.):
    factors=np.asarray(factors,float)
    if target<=0 or np.any(factors<=0):raise ValueError('Positive target and AP factors required')
    lengths=box*factors;counts=[]
    for L in lengths:
        n=max(2,int(np.floor(L/target)));candidates=np.array([n,n+1]);counts.append(int(candidates[np.argmin(abs(L/candidates-target))]))
    shape=np.array(counts);spacing=lengths/shape;p=np.ascontiguousarray(np.mod(positions_rsd*factors,lengths))
    meta=dict(box_lengths_scan=lengths.tolist(),retained_lengths_scan=lengths.tolist(),grid_shape_scan=shape.tolist(),spacing_scan=spacing.tolist(),retained_volume_scan=float(np.prod(lengths)),discarded_widths_scan=[0.,0.,0.],object_count_before=len(weights),object_count_after=len(weights),weight_before=float(weights.sum()),weight_after=float(weights.sum()))
    return p,np.ascontiguousarray(weights),meta

def deposit(positions,weights,shape,spacing):
    # Original CIC already accepts vector spacing through NumPy elementwise operations.
    return old.deposit(positions,weights,shape,np.asarray(spacing,float))

def prepare_spectrum(delta,spacing,padding_cells):
    spacing=np.asarray(spacing,float)
    shape=tuple(next_fast_len(n+2*padding_cells,real=True) for n in delta.shape)
    slices=tuple(slice((m-n)//2,(m-n)//2+n) for m,n in zip(shape,delta.shape))
    padded=np.zeros(shape,np.float32);padded[slices]=delta
    spec=rfftn(padded,workers=1);del padded
    axes=(2*np.pi*fftfreq(shape[0],spacing[0]),2*np.pi*fftfreq(shape[1],spacing[1]),2*np.pi*rfftfreq(shape[2],spacing[2]))
    for axis,k in enumerate(axes):
        view=[1,1,1];view[axis]=len(k)
        spec/=(np.sinc(k*spacing[axis]/(2*np.pi))**2).astype(np.float32).reshape(view)
    spec[0,0,0]=0
    return spec,axes,shape,slices

def measure(field,spacing,nu):
    m=old.measure(field,1.,nu);cell_volume=float(np.prod(spacing))
    m['curve']=m['negative_euler_counts']/(field.size*cell_volume)
    x=np.asarray(nu);use=abs(x)<=2.5;x=x[use];env=np.exp(-x*x/2);template=(1-x*x)*env
    m['amplitude_gaussian']=float(m['curve'][use]@template/(template@template))
    design=np.array([template,env,x*env,(x**3-3*x)*env,(x**4-6*x*x+3)*env]).T
    m['amplitude_hermite']=float(np.linalg.lstsq(design,m['curve'][use],rcond=None)[0][0])
    return m
