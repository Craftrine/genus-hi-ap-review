import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
plan=json.loads((OUT/sys.argv[1]).read_text());points=plan['points'];BR=['nofg','horizontal_only','horizontal_m1','horizontal_m5'];SN=[118,112,105,94];a=np.load(ROOT/'paper_figures/fig__intrinsic_subtracted_contours/analysis_arrays.npz');res=[]
for om,w in points:
 g=[]
 for snap in SN:
  d=ROOT/f'outputs/nearest_mesh_run_20260910/points/snap{snap}_om{om:.7f}_w{w:.7f}';gg=[]
  for b in BR:
   with np.load(d/f'R20_{b}.npz') as f:gg.append(f['curve'])
  g.append(gg)
 g=np.array(g);p=g[1:]-g[:1]-a['intrinsic'];qs={}
 for bi,b in enumerate(BR):
  v=p[:,bi].ravel();qs[b]=float(v@np.linalg.solve(a[f'C_v0_b{bi}'],v))
 res.append(dict(omega_m=om,w=w,u=round(om-.3111,7),v=round(w+1+2.25*(om-.3111),7),joint_Q=qs))
(OUT/sys.argv[2]).write_text(json.dumps(res,indent=2))
for r in res:print(r)
