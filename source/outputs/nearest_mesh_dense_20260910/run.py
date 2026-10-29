import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import importlib.util,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OM=[round(.3111+i*.001,7) for i in range(-4,5)];W=[round(-1+i*.003,7) for i in range(-6,7)];SN=[118,112,105,94]
def worker(job):
 spec=importlib.util.spec_from_file_location('nearest_runner',ROOT/'outputs/nearest_mesh_run_20260910/run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.worker(job)
if __name__=='__main__':
 grid=[(o,w) for o in OM for w in W];old=[tuple(p) for p in json.loads((ROOT/'outputs/nearest_mesh_run_20260910/plan.json').read_text())['points']];points=sorted(set(grid+old));missing=[]
 for o,w in grid:
  for snap in SN:
   if not(ROOT/f'outputs/nearest_mesh_run_20260910/points/snap{snap}_om{o:.7f}_w{w:.7f}/summary.json').exists():missing.append((snap,o,w))
 (OUT/'plan.json').write_text(json.dumps(dict(grid=grid,points=points,omega_m=OM,w=W,existing=old,new_field_jobs=len(missing),workers=8,snapshots=SN,domain=[min(OM),max(OM),min(W),max(W)],fixed_intrinsic_and_covariance=True),indent=2))
 start=time.time();print('START new field jobs',len(missing),'total merged points',len(points),flush=True)
 with ProcessPoolExecutor(max_workers=8) as pool:
  for i,f in enumerate(as_completed([pool.submit(worker,j) for j in missing]),1):
   r=f.result();(OUT/'status.json').write_text(json.dumps(dict(completed=i,total=len(missing),seconds=time.time()-start,status='COMPLETE' if i==len(missing) else 'RUNNING')))
   if i%8==0 or i==len(missing):print(i,'/',len(missing),round(time.time()-start,1),'seconds',flush=True)
