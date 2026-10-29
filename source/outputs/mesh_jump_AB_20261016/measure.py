"""Standalone diagnostic; does not write into paper or production directories."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'): os.environ[key]='1'
os.environ['NUMBA_CACHE_DIR']='/tmp/genus_mesh_AB_numba'
from pathlib import Path
import sys,json,time,gc
import numpy as np
from scipy.fft import irfftn
from concurrent.futures import ProcessPoolExecutor,as_completed
O=Path(__file__).resolve().parent; ROOT=O.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'paper_figures/fig__cosmo_redshift_evolution'))
from full_hi_pipeline import scan_sim_fixed_cell as core,scan_sim_nearest_mesh as mesh
from full_hi_pipeline.particle_first_density import valid_objects
from full_hi_pipeline.fast_counting import _encode_ranks
from matched_stencil_source import matched_hist
SN=[118,112,105,94]; BI=[0,1,2,5]; NU=np.linspace(-2.5,2.5,41)
BR=[core.LABELS[b] for b in BI]
def filename(sn,om,w):return O/'measurements'/f's{sn}_om{om:.12f}_w{w:.12f}.npz'
def worker(job):
 sn,om,w,original=job; target=filename(sn,om,w)
 if target.exists():return (sn,om,w,'cached')
 t=time.time(); src=ROOT/f'outputs/top_hi_1m_weighted_rsd_all13_20260908_v1/snap{sn}_selection.npz'
 z=json.loads(src.with_name(f'snap{sn}.json').read_text())['redshift'];alpha=core.ap_scan(z,om,w)
 with np.load(src) as d:rec=d['records']
 p,v,wt,_,_=valid_objects(rec);del rec;p=core.add_rsd_sim(p,v,z);del v
 curves={}; extras={}; timing={}; isfid=om==.3111 and w==-1.
 for mode in (['original','fixed'] if original else ['fixed']):
  shared_start=time.time()
  if mode=='original':pp,ww,g=mesh.geometry(p,wt,alpha);spacing=np.array(g['spacing_scan']);shape=g['grid_shape_scan']
  else:
   shape=[250]*3;spacing=4*alpha
   pp=np.ascontiguousarray(np.mod(p,1000.));ww=wt
  # Fixed-N AP coordinates and spacing cancel exactly in CIC. Deposit once in sim units.
  delta,_=mesh.deposit(pp,ww,shape,4. if mode=='fixed' else spacing);del pp
  spec,axes,ffshape,slices=mesh.prepare_spectrum(delta,spacing,160);del delta
  for ax,k in enumerate(axes):
   view=[1,1,1];view[ax]=len(k);spec*=np.exp(-.5*(20*k)**2).astype(np.float32).reshape(view)
  kp=np.sqrt(axes[0][:,None]**2+axes[1][None,:]**2);kz=abs(axes[2])[None,None,:]
  timing[mode+'_shared']=time.time()-shared_start
  cc=[];bc=[];contrib=[]
  for j,b in enumerate(BI):
   for taper in ([False,True] if mode=='fixed' and b else [False]):
    branch_start=time.time();work=spec.copy()
    if b:
     cut=.1 if b==1 else np.maximum(.1,core.M_SCAN[b-2]*kp[:,:,None])
     if taper:weight=(.5*(1+np.tanh((kz-cut)/.01))).astype(np.float32)
     else:weight=~core.removal_mask(axes,b)
     work*=weight;del weight
    work[0,0,0]=0
    padded=irfftn(work,s=ffshape,workers=1,overwrite_x=True);del work
    field=np.ascontiguousarray(padded[slices]);del padded
    m=mesh.measure(field,spacing,NU)
    timing[(('AB' if taper else 'A') if mode=='fixed' else 'original')+'_'+BR[j]]=time.time()-branch_start
    (bc if taper else cc).append(m['curve'])
    if mode=='fixed' and b==0:bc.append(m['curve'])
    if isfid and mode=='fixed' and (taper or b==0):
     edges=np.array([np.clip(np.ceil(np.arange(5)*250/4-.5).astype(int),0,250)]*3);edges[:,0]=0;edges[:,-1]=250
     ranks=np.empty(field.shape,np.uint8);assert not _encode_ranks(field,m['thresholds'],ranks)
     hist,sizes=matched_hist(ranks,edges,41);counts=np.cumsum(hist[:,::-1],axis=1)[:,::-1][:,1:]
     np.testing.assert_array_equal(counts.sum(0),m['negative_euler_counts']);contrib.append(counts/(field.size*np.prod(spacing)));del ranks
    del field,m
   gc.collect()
  curves['original' if mode=='original' else 'A']=np.array(cc)
  if mode=='fixed':
   curves['AB']=np.array(bc)
   if isfid:extras.update(contributions=np.array(contrib),fractions=sizes/sizes.sum())
  del spec,kp,kz;gc.collect()
 np.savez_compressed(target,omega_m=om,w=w,snapshot=sn,timing=json.dumps(timing),wall_seconds=time.time()-t,**curves,**extras)
 return (sn,om,w,round(time.time()-t,1))
def run(stage):
 (O/'measurements').mkdir(exist_ok=True)
 plan=json.loads((ROOT/'outputs/local_refinement_20260920/plan_boundary.json').read_text())
 if stage=='fid':points=[(.3111,-1.)];orig=True
 elif stage=='pairs':points=[(p['omega_m'],p['w']) for p in plan['points']];orig=True
 elif stage=='line':
  # Omega_m cut through the z=0.19 switching boundary.
  center=np.mean(plan['probes'][0]['points'],axis=0)
  xx=np.unique(np.r_[np.linspace(center[0]-.003,center[0]+.003,41),center[0]+np.array([-1e-7,0,1e-7])])
  points=[(x,center[1]) for x in xx];orig=True
 elif stage=='maskpair':
  center=.27316318004647655;ww=-.9245077896843572
  points=[(center-5e-8,ww),(center+5e-8,ww)];orig=True
 elif stage=='surface':
  import pandas as pd
  df=pd.read_csv(ROOT/'outputs/wedge_all_m_20260920/joint_chisq.csv')
  points=list(df[df.branch=='nofg'][['omega_m','w']].itertuples(index=False,name=None));orig=False
 else:raise ValueError(stage)
 jobs=[(sn,float(om),float(w),orig) for om,w in points for sn in SN]
 with ProcessPoolExecutor(max_workers=1 if stage=='maskpair' else 2) as pool:
  for i,f in enumerate(as_completed([pool.submit(worker,j) for j in jobs]),1):print(stage,i,len(jobs),f.result(),flush=True)
if __name__=='__main__':run(sys.argv[1])
