import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import importlib.util,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
U=[-.048,-.024,-.012,.012,.024,.048];V=[-.03,0.,.03];SN=[118,112,105,94]
def worker(job):
 spec=importlib.util.spec_from_file_location('nearest_runner',ROOT/'outputs/nearest_mesh_run_20260910/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.worker(job)
if __name__=='__main__':
 points=[(round(.3111+u,7),round(-1-2.25*u+v,7)) for u in U for v in V];jobs=[(s,o,w) for o,w in points for s in SN if not(ROOT/f'outputs/nearest_mesh_run_20260910/points/snap{s}_om{o:.7f}_w{w:.7f}/summary.json').exists()];(OUT/'pilot_plan.json').write_text(json.dumps(dict(points=points,u=U,v=V,slope=-2.25,snapshots=SN,jobs=len(jobs),workers=8),indent=2));start=time.time()
 with ProcessPoolExecutor(max_workers=8) as pool:
  for i,f in enumerate(as_completed([pool.submit(worker,j) for j in jobs]),1):
   f.result();(OUT/'pilot_status.json').write_text(json.dumps(dict(done=i,total=len(jobs),seconds=time.time()-start,status='COMPLETE' if i==len(jobs) else 'RUNNING')))
   if i%8==0 or i==len(jobs):print(i,'/',len(jobs),round(time.time()-start,1),'s',flush=True)
