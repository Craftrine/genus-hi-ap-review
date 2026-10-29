import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'paper_figures/fig__cosmo_redshift_evolution'))
from full_hi_pipeline import scan_sim_fixed_cell as old,scan_sim_nearest_mesh as n
from full_hi_pipeline.particle_first_density import valid_objects
from full_hi_pipeline.genus_grid import sha256_file
from full_hi_pipeline.fast_counting import _encode_ranks
from matched_stencil_source import matched_hist
P=[(.3111,-1.),(.2,-1.4),(.42,-.6)];SN=[118,112,105,94];BR=[0,3,4,5];nu=np.linspace(-2.5,2.5,41)
def worker(job):
 ci,snap=job;om,w=P[ci];cache=OUT/f'c{ci}_snap{snap}.npz'
 if cache.exists():return str(cache)
 src=ROOT/f'outputs/top_hi_1m_weighted_rsd_all13_20260908_v1/snap{snap}_selection.npz';sm=json.loads((src.parent/f'snap{snap}.json').read_text());z=sm['redshift'];p0=ROOT/f'outputs/scan_sim_fixed_cell_20260909_v1/four_branch_campaign/points/snap{snap}_om0.3111000_w-1.0000000_d4_p160';cfg=json.loads((p0/'config.json').read_text());assert sha256_file(src)==cfg['source_sha256']
 with np.load(src) as f:rec=f['records']
 p,v,wt,_,_=valid_objects(rec);del rec;p=old.add_rsd_sim(p,v,z);alpha=old.ap_scan(z,om,w);p,wt,g=n.geometry(p,wt,alpha);delta,cons=n.deposit(p,wt,g['grid_shape_scan'],g['spacing_scan']);del p,v,wt
 edges=np.array([np.clip(np.ceil(np.arange(5)*nn/4-.5).astype(int),0,nn) for nn in g['grid_shape_scan']]);edges[:,0]=0;edges[:,-1]=g['grid_shape_scan'];spec,axes,shape,sl=n.prepare_spectrum(delta,g['spacing_scan'],160);del delta
 curves=[];ths=[];tot=[];sig=[];extras={}
 for bi,b in enumerate(BR):
  field=old.filtered_field(spec,axes,shape,sl,20,b);m=n.measure(field,g['spacing_scan'],nu)
  if ci==0:
   with np.load((p0 if b==0 else Path(str(p0).replace('four_branch_campaign','remaining_three_campaign')))/f'R20_{old.LABELS[b]}.npz') as f:np.testing.assert_array_equal(m['curve'],f['curve']);np.testing.assert_array_equal(m['thresholds'],f['thresholds'])
  ranks=np.empty(field.shape,np.uint8);assert not _encode_ranks(field,m['thresholds'],ranks);hist,sizes=matched_hist(ranks,edges,41);counts=np.cumsum(hist[:,::-1],axis=1)[:,::-1][:,1:];np.testing.assert_array_equal(counts.sum(0),m['negative_euler_counts']);tot.append(counts/g['retained_volume_scan']);curves.append(m['curve']);ths.append(m['thresholds']);sig.append(m['field_std'])
  if ci==0 and snap in [118,94]:extras[f'xz_{bi}']=field[:,124,:];extras[f'xy_{bi}']=field[:,:,124]
  del field,ranks
 np.savez_compressed(cache,curves=curves,thresholds=ths,contributions=tot,fractions=sizes/sizes.sum(),sigma=sig,z=z,**extras)
 (OUT/f'c{ci}_snap{snap}.json').write_text(json.dumps(dict(omega_m=om,w=w,z=z,geometry=g,conservation=cons,source_sha256=cfg['source_sha256'],fiducial_replay=ci==0,all_block_sums_match=True,core_sha256=sha256_file(Path(n.__file__))),indent=2))
 return str(cache)
if __name__=='__main__':
 start=time.time()
 with ProcessPoolExecutor(max_workers=2) as pool:
  for i,f in enumerate(as_completed([pool.submit(worker,(ci,snap)) for ci in [0] for snap in SN]),1):print(i,'/4',f.result(),round(time.time()-start,1),flush=True)
 (OUT/'status.json').write_text(json.dumps(dict(status='COMPLETE',jobs=4,curves=16,seconds=time.time()-start,workers=2)))
