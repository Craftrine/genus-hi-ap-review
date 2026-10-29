"""Two-foreground subset; cache invariant fixed-N raw FFT, preserving arithmetic."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
os.environ['NUMBA_CACHE_DIR']='/tmp/genus_mesh_AB_numba'
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
from scipy.fft import irfftn,rfftn,fftfreq,rfftfreq,next_fast_len
from concurrent.futures import ProcessPoolExecutor,as_completed
O=Path(__file__).resolve().parent;ROOT=O.parents[1];sys.path.insert(0,str(ROOT))
from full_hi_pipeline import scan_sim_fixed_cell as core,scan_sim_nearest_mesh as mesh
from full_hi_pipeline.particle_first_density import valid_objects
SN=[118,112,105,94];BI=[1,5];NU=np.linspace(-2.5,2.5,41);CACHE=None
def context(sn):
 global CACHE
 if CACHE is not None and CACHE[0]==sn:return CACHE[1:]
 CACHE=None
 src=ROOT/f'outputs/top_hi_1m_weighted_rsd_all13_20260908_v1/snap{sn}_selection.npz';z=json.loads(src.with_name(f'snap{sn}.json').read_text())['redshift']
 with np.load(src) as d:rec=d['records']
 p,v,wt,_,_=valid_objects(rec);del rec;p=np.ascontiguousarray(np.mod(core.add_rsd_sim(p,v,z),1000.));del v
 delta,_=mesh.deposit(p,wt,[250]*3,4.);del p,wt
 shape=tuple(next_fast_len(n+320,real=True) for n in delta.shape)
 slices=tuple(slice((m-n)//2,(m-n)//2+n) for m,n in zip(shape,delta.shape))
 padded=np.zeros(shape,np.float32);padded[slices]=delta;del delta
 raw=rfftn(padded,workers=1);del padded
 CACHE=(sn,z,raw,shape,slices);return CACHE[1:]
def calculate(sn,om,w,scheme):
 start=time.time();z,raw,shape,slices=context(sn);alpha=core.ap_scan(z,om,w);spacing=4*alpha
 axes=(2*np.pi*fftfreq(shape[0],spacing[0]),2*np.pi*fftfreq(shape[1],spacing[1]),2*np.pi*rfftfreq(shape[2],spacing[2]));spec=raw.copy()
 # Preserve the production deconvolution and smoothing order exactly.
 for ax,k in enumerate(axes):
  view=[1,1,1];view[ax]=len(k);spec/=(np.sinc(k*spacing[ax]/(2*np.pi))**2).astype(np.float32).reshape(view)
 spec[0,0,0]=0
 for ax,k in enumerate(axes):
  view=[1,1,1];view[ax]=len(k);spec*=np.exp(-.5*(20*k)**2).astype(np.float32).reshape(view)
 kp=np.sqrt(axes[0][:,None]**2+axes[1][None,:]**2);kz=abs(axes[2])[None,None,:];shared=time.time()-start;curves=[];times=[]
 for b in BI:
  t=time.time();work=spec.copy()
  if scheme=='A':work*=~core.removal_mask(axes,b)
  else:
   cut=.1 if b==1 else np.maximum(.1,core.M_SCAN[b-2]*kp[:,:,None]);weight=(.5*(1+np.tanh((kz-cut)/.01))).astype(np.float32);work*=weight;del weight
  work[0,0,0]=0;padded=irfftn(work,s=shape,workers=1,overwrite_x=True);del work
  field=np.ascontiguousarray(padded[slices]);del padded
  curves.append(mesh.measure(field,spacing,NU)['curve']);del field;times.append(time.time()-t)
 return np.array(curves),shared,times,time.time()-start
def worker(job):
 sn,om,w,scheme=job;f=O/'subset_measurements'/f'{scheme}_s{sn}_om{om:.12f}_w{w:.12f}.npz'
 if f.exists():return 'cached'
 existing=O/'measurements'/f's{sn}_om{om:.12f}_w{w:.12f}.npz'
 if existing.exists():
  with np.load(existing) as d:cc=d[scheme][[1,3]]
  np.savez_compressed(f,curves=cc,omega_m=om,w=w,snapshot=sn);return 'reused'
 curves,shared,times,wall=calculate(sn,om,w,scheme)
 np.savez_compressed(f,curves=curves,omega_m=om,w=w,snapshot=sn,shared_seconds=shared,branch_seconds=times,wall_seconds=wall)
 return round(wall,2)
def main(scheme):
 started=time.time()
 assert scheme in ['A','AB'];(O/'subset_measurements').mkdir(exist_ok=True)
 points=pd.read_csv(O/'subset_nodes.csv')[['omega_m','w']].to_numpy();jobs=[]
 for start in range(0,len(points),16):
  for sn in SN:
   jobs.extend((sn,om,w,scheme) for om,w in points[start:start+16])
 with ProcessPoolExecutor(max_workers=2) as pool:
  for i,f in enumerate(as_completed([pool.submit(worker,j) for j in jobs]),1):
   result=f.result()
   if i%16==0 or i==len(jobs):print('subset',scheme,i,len(jobs),result,flush=True)
 elapsed=time.time()-started
 pd.DataFrame([dict(scheme=scheme,foregrounds=2,parameter_nodes=len(points),workers=2,elapsed_seconds=elapsed,hours_1285_same_two_foregrounds=elapsed/len(points)*1285/3600)]).to_csv(O/'subset_runtime.csv',index=False)
 print('subset complete seconds',elapsed,flush=True)
def verify():
 for sn,om,w in [(118,.3111,-1.),(118,.2727603578368017,-.9245077896843572)]:
  got,_,_,wall=calculate(sn,om,w,'AB')
  with np.load(O/'measurements'/f's{sn}_om{om:.12f}_w{w:.12f}.npz') as d:want=d['AB'][[1,3]]
  np.testing.assert_array_equal(got,want);print('Cached raw-FFT exact replay passed',sn,om,w,wall,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='verify':verify()
 else:main(sys.argv[1])
