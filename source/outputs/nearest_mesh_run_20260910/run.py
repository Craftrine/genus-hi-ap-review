"""New no-crop mesh contract; four workers; verify fiducial replay first."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from full_hi_pipeline import scan_sim_fixed_cell as old,scan_sim_nearest_mesh as n
from full_hi_pipeline.particle_first_density import valid_objects,output_lock
from full_hi_pipeline.genus_grid import sha256_file,_atomic_json,_atomic_npz
SN=[118,112,105,94];BR=[0,1,2,6];nu=np.linspace(-2.5,2.5,41)
def worker(job):
 snap,om,w=job;d=OUT/'points'/f'snap{snap}_om{om:.7f}_w{w:.7f}';d.mkdir(parents=True,exist_ok=True)
 src=ROOT/f'outputs/top_hi_1m_weighted_rsd_all13_20260908_v1/snap{snap}_selection.npz';sm=json.loads((src.parent/f'snap{snap}.json').read_text());z=sm['redshift']
 cfg=dict(snapshot=snap,z=z,omega_m=om,w=w,target_spacing=4,R=20,padding=160,source_sha256=sha256_file(src),core_sha256=sha256_file(Path(n.__file__)),old_core_sha256=sha256_file(Path(old.__file__)),branches=BR,contract='full_AP_box_nearest_per_axis_spacing_no_crop')
 with output_lock(d/'point'):
  if (d/'summary.json').exists():
   m=json.loads((d/'summary.json').read_text());assert m['config']==cfg
   for r in m['measurements']:assert sha256_file(d/r['file'])==r['sha256']
   return m
  start=time.time()
  with np.load(src) as f:rec=f['records']
  p,v,wt,_,_=valid_objects(rec);del rec;p=old.add_rsd_sim(p,v,z);alpha=old.ap_scan(z,om,w);p,wt,g=n.geometry(p,wt,alpha)
  delta,conservation=n.deposit(p,wt,g['grid_shape_scan'],g['spacing_scan']);del p,v,wt
  spec,axes,shape,slices=n.prepare_spectrum(delta,g['spacing_scan'],160);del delta;rows=[]
  for b in BR:
   field=old.filtered_field(spec,axes,shape,slices,20,b);m=n.measure(field,g['spacing_scan'],nu);del field
   if om==.3111 and w==-1:
    path=ROOT/f'outputs/scan_sim_fixed_cell_20260909_v1/four_branch_campaign/points/snap{snap}_om0.3111000_w-1.0000000_d4_p160/R20_{old.LABELS[b]}.npz'
    with np.load(path) as f:
     np.testing.assert_array_equal(m['curve'],f['curve']);np.testing.assert_array_equal(m['thresholds'],f['thresholds'])
   filename=f'R20_{old.LABELS[b]}.npz';_atomic_npz(d/filename,**m);rows.append(dict(branch=old.LABELS[b],file=filename,sha256=sha256_file(d/filename)))
  result=dict(config=cfg,geometry=g,alpha=alpha.tolist(),fft_shape=list(shape),conservation=conservation,measurements=rows,seconds=time.time()-start,fiducial_exact_replay=(om==.3111 and w==-1));_atomic_json(d/'summary.json',result)
 return result

def main():
 p1=json.loads((ROOT/'outputs/local_refinement_step1_20260910/plan.json').read_text());grid=[(o,w) for o in p1['omega_m'] for w in p1['w']];axial=[tuple(v) for v in json.loads((ROOT/'outputs/local_refinement_step2a_20260910/plan.json').read_text())['points']];points=grid+axial;assert len(set(points))==37
 _atomic_json(OUT/'plan.json',dict(points=points,grid_points=grid,axial_points=axial,snapshots=SN,workers=4,total_jobs=148,total_curves=592,scope='new full-box nearest mesh, no old cropped scan curves merged',fixed_covariance='old fiducial covariance reused only after exact fiducial replay'))
 start=time.time();done=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  # Do not dispatch non-fiducial points until the complete four-epoch replay passes.
  for f in as_completed([pool.submit(worker,(s,.3111,-1.)) for s in SN]):done.append(f.result())
  print('FIDUCIAL REPLAY PASSED: all16 curves identical',flush=True)
  jobs=[(s,o,w) for o,w in points if not(o==.3111 and w==-1) for s in SN]
  for f in as_completed([pool.submit(worker,j) for j in jobs]):
   r=f.result();done.append(r);_atomic_json(OUT/'status.json',dict(completed=len(done),total=148,elapsed_seconds=time.time()-start,latest=r['config'],status='COMPLETE' if len(done)==148 else 'RUNNING'));print(len(done),'/148',round(time.time()-start,1),'s',r['config']['snapshot'],r['config']['omega_m'],r['config']['w'],flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
