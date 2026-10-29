from pathlib import Path
import json
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent
BR=['nofg','horizontal_only','horizontal_m1','horizontal_m4'];rows=[]
for f in (O/'measurements').glob('*.npz'):
 with np.load(f) as d:
  if 'timing' not in d:continue
  if abs(float(d['omega_m'])-.27316318004647655)<1e-6:continue
  tt=json.loads(str(d['timing']));overhead=max(0.,float(d['wall_seconds'])-sum(tt.values()));sn=int(d['snapshot']);om=float(d['omega_m']);w=float(d['w'])
 for mode in ['original','A','AB']:
  shared=tt[('original' if mode=='original' else 'fixed')+'_shared']
  for br in BR:
   key=f'{"A" if mode=="AB" and br=="nofg" else mode}_{br}'
   rows.append(dict(omega_m=om,w=w,snapshot=sn,scheme=mode,branch=br,shared_seconds=shared,branch_seconds=tt[key],unallocated_wall_seconds=overhead))
d=pd.DataFrame(rows);d.to_csv(O/'timing_measurements.csv',index=False);out=[]
for (mode,br),g in d.groupby(['scheme','branch'],sort=False):
 node=g.groupby(['omega_m','w']).filter(lambda x:len(x)==4).groupby(['omega_m','w'])[['shared_seconds','branch_seconds']].sum()
 if not len(node):continue
 total=node.sum(axis=1);out.append(dict(scheme=mode,branch=br,complete_parameter_nodes=len(node),shared_seconds_per_node=node.shared_seconds.median(),branch_seconds_per_node=node.branch_seconds.median(),standalone_seconds_per_node=total.median(),hours_1285_serial=1285*total.median()/3600,hours_1285_two_workers=1285*total.median()/7200))
pd.DataFrame(out).to_csv(O/'timing_summary.csv',index=False)
print(pd.DataFrame(out).to_string(index=False))
# Actual four-snapshot completion span, plus shared-cost estimates for all four branches.
wall=[]
for (om,w),g in d.groupby(['omega_m','w']):
 files=[O/'measurements'/f's{sn}_om{om:.12f}_w{w:.12f}.npz' for sn in [118,112,105,94]]
 if not all(f.exists() for f in files):continue
 starts=[];ends=[];secs=[]
 for f in files:
  with np.load(f) as z:
   if 'wall_seconds' not in z:break
   elapsed=float(z['wall_seconds']);end=f.stat().st_mtime
  starts.append(end-elapsed);ends.append(end);secs.append(elapsed)
 if len(secs)==4:wall.append(dict(omega_m=om,w=w,all_three_schemes_four_branches_worker_seconds=sum(secs),node_elapsed_seconds_approx=max(ends)-min(starts)))
pd.DataFrame(wall).to_csv(O/'node_wall_times.csv',index=False)
agg=[]
for mode,g in d.groupby('scheme',sort=False):
 costs=[]
 for (om,w),node in g.groupby(['omega_m','w']):
  if len(node)!=16:continue
  shared=node.groupby('snapshot').shared_seconds.first().sum();branch=node.branch_seconds.sum();costs.append(shared+branch)
 if costs:agg.append(dict(scheme=mode,seconds_per_node_all_four_foregrounds=np.median(costs),hours_1285_two_workers=np.median(costs)*1285/7200))
pd.DataFrame(agg).to_csv(O/'timing_all_foregrounds.csv',index=False)
