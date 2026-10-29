from pathlib import Path
import time,subprocess,os,signal,shutil
import numpy as np
import pandas as pd
O=Path(__file__).parent;P=O.parents[1]
d=pd.read_csv(P/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/before_after_chisq.csv');want={f'om{x:.7f}_w{y:.7f}.npz' for x,y in d[['omega_m','w']].drop_duplicates().to_numpy()}
while True:
 good={}
 for folder in ['old_points','old_points_tail']:
  for f in (O/folder).glob('*.npz'):
   try:
    with np.load(f) as a:
     assert a['curves'].shape==(4,4,41)
     assert np.isfinite(a['curves']).all()
    good[f.name]=f
   except Exception:pass
 if want<=good.keys():break
 time.sleep(10)
# Stop only the redundant head runner and its own spawned workers, after all requested files exist and load.
lines=subprocess.check_output(['ps','-axo','pid,ppid,command'],text=True).splitlines();entries=[]
for line in lines[1:]:
 p=line.strip().split(None,2)
 if len(p)==3:entries.append((int(p[0]),int(p[1]),p[2]))
heads={pid for pid,ppid,cmd in entries if cmd.endswith('outputs/wedge_all_m_20260920/fill_old_points.py')}
for pid,ppid,cmd in entries:
 if ppid in heads and 'spawn_main' in cmd:
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
for pid in heads:
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
for name,f in good.items():
 if f.parent.name=='old_points_tail':shutil.copy2(f,O/'old_points'/name)
print('All 233 points validated and assembled.',flush=True)
for script in ['analyze.py','replot.py']:
 subprocess.run(['/Users/LOCAL_USER/anaconda3/bin/python3','-u',str(O/script)],check=True)
print('COMPLETE',flush=True)
