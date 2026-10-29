"""Bounded matched top-HI selection and weighting/RSD comparison; no producer changes."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,time,gc
import numpy as np
from numba import set_num_threads
from full_hi_pipeline import particle_first_density as pd, particle_first_genus as pg
from full_hi_pipeline import particle_first_coordinates as pc
from full_hi_pipeline.stream_density import JIUTIAN_DTYPE,REDSHIFT_BY_SNAPSHOT
from full_hi_pipeline.genus_grid import _atomic_json,sha256_file
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/top_hi_1m_weighted_rsd_all13_20260908_v1';OUT.mkdir(exist_ok=True)
SNAPS=[118,112,105,100,99,94,88,83,76,71,66,60,55]
set_num_threads(1)
core=pg.load_measurement_core(ROOT/'exports/jiutian_particle_first_R20_expert_review_20260906_v3/external_core')
rows=[]
for snap in SNAPS:
 start=time.time(); dest=OUT/f'snap{snap}.json'
 if dest.exists():rows.append(json.loads(dest.read_text()));continue
 pp=ROOT/f'outputs/literal_unitcount_R20_w5_512/snap{snap}/selection_provenance.json'
 prov=json.loads(pp.read_text()) if pp.exists() else {}
 threshold=prov.get('selection_threshold_raw',0.);src=Path('/Users/LOCAL_USER/Downloads/halo')/f'snap{snap}'
 before=src.stat();assert before.st_size%40==0
 blocks=[];ids=[];offset=0;dig=hashlib.sha256();totalmass=0.;totalmass2=0.;nvalid=0
 _atomic_json(OUT/'status.json',{'stage':'reading_and_selecting','snapshot':snap,'completed':len(rows),'target':13})
 with src.open('rb') as h:
  while True:
   raw=h.read(40*1000000)
   if not raw:break
   dig.update(raw);r=np.frombuffer(raw,dtype=JIUTIAN_DTYPE)
   good=np.isfinite(r['HImass'])&(r['HImass']>0)&np.isfinite(r['Pos']).all(1)&np.isfinite(r['Vel']).all(1)
   mass=r['HImass'][good].astype(float);totalmass+=mass.sum();totalmass2+=np.dot(mass,mass);nvalid+=len(mass)
   take=good&(r['HImass']>=threshold)
   blocks.append(r[take].copy());ids.append(np.flatnonzero(take)+offset);offset+=len(r)
   if not prov and sum(len(b) for b in blocks)>1000000:
    candidates=np.concatenate(blocks);ci=np.concatenate(ids)
    cutoff=np.partition(candidates['HImass'],len(candidates)-1000000)[len(candidates)-1000000]
    greater=np.flatnonzero(candidates['HImass']>cutoff);ties=np.flatnonzero(candidates['HImass']==cutoff)
    ties=ties[np.argsort(ci[ties])][:1000000-len(greater)]
    chosen=np.concatenate((greater,ties));blocks=[candidates[chosen].copy()];ids=[ci[chosen].copy()]
    threshold=float(cutoff)
    del candidates,ci

 after=src.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 r=np.concatenate(blocks);ids=np.concatenate(ids);del blocks
 assert np.count_nonzero(r['HImass']>threshold)<=1000000 and len(r)>=1000000
 order=np.lexsort((ids,-r['HImass'].astype(float)))[:1000000]
 order=order[np.argsort(ids[order])];r=r[order];ids=ids[order]
 assert len(r)==1000000
 np.savez_compressed(OUT/f'snap{snap}_selection.npz',records=r,record_indices=ids)
 pos,vel,weights,_,_=pd.valid_objects(r)
 assert len(weights)==1000000
 z=REDSHIFT_BY_SNAPSHOT[snap];rsd=pc.add_fiducial_rsd(pos,vel,z);lengths=np.full(3,1000.)
 result={'snapshot':snap,'redshift':z,'source_sha256':dig.hexdigest(),'source_records':offset,'source_path':str(src),'selection':'top1000000 finite positive HI; ties by original record index','selection_threshold':threshold,'selected_indices_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'old_indices_sha256':prov.get('record_index_sha256'),'n_valid_all':nvalid,'HI_fraction':float(weights.sum()/totalmass),'n_eff_top':float(weights.sum()**2/(1e9*np.dot(weights,weights))),'n_eff_all':float(totalmass**2/(1e9*totalmass2)),'config':{'grid':512,'R':20,'nu_points':41,'nu_range':[-2.5,2.5],'CIC_deconvolution':True,'omega_m':.3111,'w':-1.,'h':.6766,'foreground':'none','los_axis':2},'measurements':{}}
 for label,positions,wts in [('HI_rsd',rsd,weights),('count_rsd',rsd,np.ones(len(weights))),('count_real',pos,np.ones(len(weights)))]:
  _atomic_json(OUT/'status.json',{'stage':label,'snapshot':snap,'completed':len(rows),'target':13})
  mesh=np.zeros((1,512,512,512),dtype=np.float64)
  pd.cic_deposit(positions,wts,lengths,mesh)
  delta,check=pd.to_density_contrast(mesh[0],float(wts.sum()));del mesh;gc.collect()
  spec=pg.prepare_trial_spectrum(delta,deconvolve_cic=True,threads=1);del delta
  smooth=pg.smooth_trial_spectrum(spec,(512,)*3,lengths,20.);del spec
  field=pg.inverse_trial_field(smooth,(512,)*3,threads=1);del smooth
  m=pg.measure_thresholds_and_genus(field,lengths,np.linspace(-2.5,2.5,41),core);del field;gc.collect()
  np.savez_compressed(OUT/f'snap{snap}_{label}.npz',**m)
  result['measurements'][label]={'A':float(m['amplitude_gaussian_signed']),'AR3':float(m['amplitude_gaussian_signed']*8000),'hermite_AR3':float(m['amplitude_hermite_signed']*8000),'conservation':check}
 result['elapsed_seconds']=time.time()-start
 _atomic_json(dest,result);rows.append(result);print(snap,result['measurements'],flush=True)
_atomic_json(OUT/'summary.json',{'rows':rows,'producer_sha256':sha256_file(Path(__file__))})
_atomic_json(OUT/'status.json',{'stage':'COMPLETE','completed':13,'target':13})
