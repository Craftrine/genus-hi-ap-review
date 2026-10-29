import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import importlib.util,json,time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
P=Path('/Users/LOCAL_USER/Documents/ChatGPT/genus');O=Path(__file__).parent;D=P.parent/'genus_workbuddy/dense_band_20260917'
spec=importlib.util.spec_from_file_location('dense_original',D/'run_dense_fast.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);r.FFTW=1
OUT=O/'old_points_tail';OUT.mkdir(exist_ok=True)
def init():r.load_cats()
def worker(point):
 om,w=point;fp=OUT/f'om{om:.7f}_w{w:.7f}.npz'
 if fp.exists():return fp.name
 curves=np.zeros((4,4,41));metas=[]
 for si,snap in enumerate(r.SNAPS):
  p,wt=r.CATS[snap];alpha=r.old.ap_scan(r.ZS[snap],om,w);pp,w2,g=r.n.geometry(p.copy(),wt,alpha);delta,cons=r.n.deposit(pp,w2,g['grid_shape_scan'],g['spacing_scan']);del pp
  spec,axes,shape,sl=r.n.prepare_spectrum(delta,g['spacing_scan'],160);del delta;smooth=r.smooth_spectrum(spec,axes);del spec
  for j,br in enumerate([0,3,4,5]):
   field=r.branch_field(smooth,axes,shape,sl,br,'A');m=r.n.measure(field,g['spacing_scan'],r.NU);curves[j,si]=m['curve'];del field
  metas.append(g);del smooth
 np.savez_compressed(fp,omega_m=om,w=w,curves=curves,branches=[0,3,4,5]);return fp.name
if __name__=='__main__':
 old=pd.read_csv(P/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/before_after_chisq.csv');pts=old[['omega_m','w']].drop_duplicates().to_numpy()[-90:][::-1];start=time.time()
 with ProcessPoolExecutor(max_workers=4,initializer=init) as pool:
  for i,f in enumerate(as_completed([pool.submit(worker,tuple(pt)) for pt in pts]),1):print(i,len(pts),f.result(),round(time.time()-start,1),flush=True)
