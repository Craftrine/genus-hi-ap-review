"""Recompute diagnostic tables from saved measurements; never rerun measurements."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from sklearn.covariance import oas
O=Path(__file__).resolve().parent;ROOT=O.parents[1]
SN=[118,112,105,94];BI=[0,1,2,5];BR=['nofg','horizontal_only','horizontal_m1','horizontal_m4']
MODES=['original','A','AB']
plan=json.loads((ROOT/'outputs/local_refinement_20260920/plan_boundary.json').read_text())
ref=np.load(ROOT/'outputs/wedge_all_m_20260920/reference_covariances.npz')
C=ref['covariances'][BI];I=ref['intrinsic'][:,BI]; G=ref['curves'][:,BI]
def read(om,w):
 files=[O/'measurements'/f's{sn}_om{om:.12f}_w{w:.12f}.npz' for sn in SN]
 if not all(f.exists() for f in files):return None
 return [dict(np.load(f)) for f in files]
fid=read(.3111,-1.)
if fid is None:raise RuntimeError('Fiducial measurements not yet complete')
for key in ['original','A']:
 err=np.max(abs(np.array([f[key] for f in fid])-G))
 print('Fiducial replay',key,err);assert err<1e-18
BG=np.array([f['AB'] for f in fid]); IB=BG[1:]-BG[0]
loo=np.array([(f['AB'][:,None,:]-f['contributions'])/(1-f['fractions'][None,:,None]) for f in fid]);CB=[]
for j in range(4):
 x=(loo[1:,j]-loo[0,j]).transpose(1,0,2).reshape(64,123);x-=x.mean(0);raw=63/64*x.T@x;sd=np.sqrt(np.diag(raw));corr,_=oas(x/np.sqrt(np.mean(x*x,axis=0)),assume_centered=True);CB.append(corr*sd[:,None]*sd[None,:])
CB=np.array(CB);np.testing.assert_allclose(CB[0],C[0],rtol=1e-10,atol=1e-30)
ch=[cho_factor(c) for c in C];chb=[cho_factor(c) for c in CB]
def qs(data,mode,matched=False):
 g=np.array([f[mode] for f in data]);r=(g[1:]-g[0]-(IB if mode=='AB' else I)).transpose(1,0,2).reshape(4,123)
 return np.array([v@cho_solve(c,v) for v,c in zip(r,chb if matched and mode=='AB' else ch)]),r
rows=[]
for probe in plan['probes']:
 ds=[read(*p) for p in probe['points']]
 if any(d is None for d in ds):continue
 for mode in MODES:
  for matched in ([False,True] if mode=='AB' else [False]):
   qq=[qs(d,mode,matched) for d in ds]
   for j,br in enumerate(BR):
    dr=qq[1][1][j]-qq[0][1][j];factor=(chb if matched and mode=='AB' else ch)[j]
    rows.append(dict(probe=probe['edge_id'],scheme=mode,covariance='matched' if matched else 'common',branch=br,Q_left=qq[0][0][j],Q_right=qq[1][0][j],delta_Q=qq[1][0][j]-qq[0][0][j],jump_norm2=dr@cho_solve(factor,dr)))
if rows:pd.DataFrame(rows).to_csv(O/'boundary_results.csv',index=False)
# Tight bracket of the additional fixed-N hard-mask crossing found on the 1D cut.
root=.27316318004647655;ww=-.9245077896843572
ds=[read(root-5e-8,ww),read(root+5e-8,ww)];maskrows=[]
if all(d is not None for d in ds):
 for mode in MODES:
  qq=[qs(d,mode) for d in ds]
  for j,b in enumerate(BR):maskrows.append(dict(branch=b,scheme=mode,omega_m_left=root-5e-8,omega_m_right=root+5e-8,w=ww,Q_left=qq[0][0][j],Q_right=qq[1][0][j],delta_Q=qq[1][0][j]-qq[0][0][j]))
 pd.DataFrame(maskrows).to_csv(O/'hard_mask_pair.csv',index=False)
center=np.mean(plan['probes'][0]['points'],axis=0);rows=[]
for path in sorted((O/'measurements').glob(f's118_*_w{center[1]:.12f}.npz')):
 with np.load(path) as f:om=float(f['omega_m']);w=float(f['w'])
 ds=read(om,w)
 if ds is None:continue
 for mode in MODES:
  q,_=qs(ds,mode)
  for j,b in enumerate(BR):rows.append(dict(omega_m=om,w=w,scheme=mode,branch=b,chi2_int=q[j]))
if rows:pd.DataFrame(rows).to_csv(O/'line.csv',index=False)
