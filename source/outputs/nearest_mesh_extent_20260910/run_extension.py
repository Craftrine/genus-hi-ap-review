import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import importlib.util,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;SN=[118,112,105,94]
def worker(job):
 spec=importlib.util.spec_from_file_location('nearest_runner',ROOT/'outputs/nearest_mesh_run_20260910/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.worker(job)
if __name__=='__main__':
 uv=[(u,v) for u in [-.048,-.036,-.024,-.018,-.012,-.006,.006,.012,.018,.024,.036,.048] for v in [-.03,-.015,0,.015,.03]]
 uv += [(u,v) for u in [-.072,.072,-.096,.096] for v in [-.06,0,.06]]
 uv += [(u,v) for u in [-.1111,.1089] for v in [-.06,0,.06]]
 uv += [(u,v) for u in [-.048,0,.048] for v in [-.06,.06]]
 pts=sorted(set((round(.3111+u,7),round(-1-2.25*u+v,7)) for u,v in uv))
 central=[tuple(v) for v in json.loads((ROOT/'outputs/nearest_mesh_dense_20260910/plan.json').read_text())['points']];pilot=[tuple(v) for v in json.loads((OUT/'pilot_plan.json').read_text())['points']];allpts=sorted(set(pts+central+pilot));jobs=[]
 for o,w in pts:
  for s in SN:
   if not(ROOT/f'outputs/nearest_mesh_run_20260910/points/snap{s}_om{o:.7f}_w{w:.7f}/summary.json').exists():jobs.append((s,o,w))
 (OUT/'extension_plan.json').write_text(json.dumps(dict(points=pts,all_points=allpts,new_jobs=len(jobs),workers=8,snapshots=SN,domain='convex hull of all new-contract measured points; no interpolation outside'),indent=2));print('NEW',len(jobs),'jobs; merged',len(allpts),'cosmologies',flush=True);start=time.time()
 with ProcessPoolExecutor(max_workers=8) as pool:
  for i,f in enumerate(as_completed([pool.submit(worker,j) for j in jobs]),1):
   f.result();(OUT/'extension_status.json').write_text(json.dumps(dict(done=i,total=len(jobs),seconds=time.time()-start,status='COMPLETE' if i==len(jobs) else 'RUNNING')))
   if i%8==0 or i==len(jobs):print(i,'/',len(jobs),round(time.time()-start,1),'s',flush=True)
