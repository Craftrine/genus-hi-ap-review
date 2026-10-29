"""Evaluate the saved FFT lattice geometry and analytic filter, never the field."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.fft import next_fast_len

O=Path(__file__).resolve().parent;ROOT=O.parents[1]
M=.2938445235861259;R=20.;cut=.1
records=json.loads((ROOT/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/geometry.json').read_text())
for file in sorted((ROOT.parent/'genus_workbuddy/dense_band_20260917/points_A').glob('*.npz')):
    with np.load(file) as z:
        om,w=float(z['omega_m']),float(z['w'])
        for rec in json.loads(str(z['meta'])):
            records.append(dict(omega_m=om,w=w,snapshot=rec['snap'],shape=rec['shape'],spacing=rec['spacing']))
assert len(records)==1285*4
results=[]
for r in records:
    nfft=np.array([next_fast_len(n+320,real=True) for n in r['shape']])
    spacing=np.array(r['spacing'])
    dk=2*np.pi/(nfft*spacing)
    # rfftn stores nonnegative kz; symmetry supplies equivalent negative modes.
    iz=int(np.ceil(cut/dk[2]));kz=iz*dk[2]
    kx=np.arange(nfft[0]//2+1)*dk[0];ky=np.arange(nfft[1]//2+1)*dk[1]
    kp2=kx[:,None]**2+ky[None,:]**2
    removed=kz<=M*np.sqrt(kp2)
    candidate=np.where(removed,kp2,np.inf)
    ix,iy=np.unravel_index(candidate.argmin(),candidate.shape)
    if not np.isfinite(candidate[ix,iy]):raise AssertionError('No added wedge modes at lowest retained kz')
    k=np.array([kx[ix],ky[iy],kz]);wg=float(np.exp(-.5*R**2*(k@k)))
    wg32=float(np.prod(np.exp(-.5*(R*k)**2).astype(np.float32),dtype=np.float32))
    cic=float(np.prod(np.sinc(k*spacing/(2*np.pi))**2))
    results.append(dict(omega_m=r['omega_m'],w=r['w'],snapshot=r['snapshot'],
        fft_nx=nfft[0],fft_ny=nfft[1],fft_nz=nfft[2],spacing_x=spacing[0],spacing_y=spacing[1],spacing_z=spacing[2],
        kx=k[0],ky=k[1],k_parallel=kz,k_perp=np.sqrt(candidate[ix,iy]),k_norm=np.linalg.norm(k),
        max_R20_weight=wg,R20_weight_float32=wg32,R20_over_CIC_at_max_R20=wg/cic,
        weight_over_float32_eps=wg/np.finfo(np.float32).eps))
df=pd.DataFrame(results);df.to_csv(O/'tables/weak_wedge_filter_weights.csv',index=False,float_format='%.17g')
# For kz>=cut and kz<=m*kperp, k^2>=cut^2*(1+m^-2).
# This continuum bound is rigorous for all discrete modes. The lattice minimum
# is at the lowest allowed kz since both the transverse bound and kz increase.
kmin=cut*np.sqrt(1+M**-2);bound=float(np.exp(-.5*R**2*kmin**2))
f=np.finfo(np.float32)
report=dict(m=M,R=R,k_parallel_cut=cut,geometries=len(df),continuum_k_min=float(kmin),
    continuum_R20_weight_upper_bound=bound,discrete_max_R20=float(df.max_R20_weight.max()),
    discrete_min_of_max_R20=float(df.max_R20_weight.min()),
    float32_eps=float(f.eps),float32_smallest_normal=float(f.tiny),
    float32_smallest_subnormal=float(np.nextafter(np.float32(0),np.float32(1))),
    all_weights_nonzero_in_float32=bool((df.R20_weight_float32>0).all()),
    continuum_bound_after_CIC_correction=bound*(np.pi/2)**6,
    conclusion='R20 strongly suppresses added modes; weights remain representable in float32. Comparing a multiplier with eps at 1 is not an underflow test and does not prove identical inverse-FFT fields or genus. Saved spectra and per-vertex threshold margins are not available in this analysis, so an exact rounding mechanism is not established.',
    no_fft_or_genus_run=True)
assert df.max_R20_weight.max()<=bound*(1+1e-12)
(O/'qa/weak_wedge_filter_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
