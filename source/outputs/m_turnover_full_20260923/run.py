"""Complete the six missing m branches on the identical 1285 cosmologies.
Reuse validated pilot spectra-derived curves; remeasure horizontal control everywhere else.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,fcntl,traceback,subprocess,importlib.util
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
O=Path(__file__).resolve().parent;P=O.parents[1]
PILOT=P/'outputs/m_turnover_pilot_20260923';S=P/'outputs/wedge_all_m_20260920'
spec=importlib.util.spec_from_file_location('pilot_measurement',PILOT/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
ALL_M=np.array(json.loads((PILOT/'plan.json').read_text())['m_values'])
NEW_I=[1,2,3,5,6,7];NEW_M=ALL_M[NEW_I];BMAP=[0]+[i+1 for i in NEW_I]
REF=None;FAC=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def writejson(path,obj):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def prepare():
    base=pd.read_csv(S/'joint_chisq.csv');coords=base[base.branch=='nofg'][['omega_m','w']].copy()
    assert len(coords)==1285 and not coords.duplicated().any()
    pts=[dict(id=f'p{i:04d}',omega_m=float(x),w=float(y)) for i,(x,y) in enumerate(coords.to_numpy())]
    coordinate_sources={}
    for p in pts:
        p['display_omega_m']=p['omega_m'];p['display_w']=p['w']
        for folder in [P.parent/'genus_workbuddy/dense_band_20260917/points_A',S/'old_points']:
            fp=folder/f"om{p['omega_m']:.7f}_w{p['w']:.7f}.npz"
            if fp.exists():
                with np.load(fp) as f:p['omega_m']=float(f['omega_m']);p['w']=float(f['w'])
                p['coordinate_source']=str(fp);coordinate_sources[str(fp)]=sha(fp);break
        else:raise RuntimeError(f'No original coordinate source for {p}')
    (O/'source_Q.csv').write_bytes((S/'joint_chisq.csv').read_bytes())
    (O/'references.npz').write_bytes((PILOT/'references.npz').read_bytes())
    pcfg=json.loads((PILOT/'plan.json').read_text());reused=0
    for pp in pcfg['points']:
        match=[p for p in pts if abs(p['omega_m']-pp['omega_m'])<1e-11 and abs(p['w']-pp['w'])<1e-11];assert len(match)==1
        p=match[0]
        for snap in r.SN:
            dest=O/'measurements'/f'{p["id"]}_snap{snap}.npz'
            if dest.exists():continue
            with np.load(PILOT/'measurements'/f'{pp["id"]}_snap{snap}.npz') as f:
                meta=json.loads(str(f['metadata']));meta['removed_rfft_bins']=[meta['removed_rfft_bins'][i] for i in BMAP]
                data={k:f[k][BMAP] for k in ['curves','thresholds','counts','sigma']}
                data.update(omega_m=p['omega_m'],w=p['w'],z=float(f['z']),m_values=NEW_M,metadata=json.dumps(meta),origin='validated nine-m pilot')
                np.savez_compressed(dest,**data);reused+=1
    for path,digest in pcfg['sources'].items():assert sha(path)==digest
    cfg=dict(points=pts,m_values=ALL_M.tolist(),new_m_indices=NEW_I,new_m_values=NEW_M.tolist(),snapshots=r.SN,
        workers=8,target_spacing=4,padding_cells=160,R=20,k_parallel_cut=.1,n_cosmologies=1285,
        version=2,coordinate_policy='Measure at exact source NPZ coordinates; retain common archived display coordinates for triangulation',coordinate_source_hashes=coordinate_sources,
        reused_pilot_cosmologies=13,initial_new_cosmologies=1272,initial_new_genus_curves=30528,
        initial_horizontal_replay_curves=5088,scope='Nine m values at all 1285 coordinates; no k-cut variation or spatial refinement',
        source_Q_sha256=sha(S/'joint_chisq.csv'),reference_sha256=sha(O/'references.npz'),
        input_hashes=pcfg['sources'],code_hashes={str(f):sha(f) for f in [Path(__file__),PILOT/'run.py',Path(r.old.__file__),Path(r.n.__file__),O/'finish.py']})
    writejson(O/'plan.json',cfg);return cfg

def init():
    global REF,FAC
    r.O=O;r.MS=NEW_M;r.init()
    with np.load(O/'references.npz') as f:REF=f['intrinsic'][:,BMAP];FAC=[cho_factor(c) for c in f['covariances'][BMAP]]

def worker(p):
    start=time.time();new=0
    for snap in r.SN:
        result=r.worker((p,snap));new+=not result['cached']
    curves=[];metadata=[]
    for snap in r.SN:
        with np.load(O/'measurements'/f'{p["id"]}_snap{snap}.npz') as f:
            assert f['curves'].shape==(7,41);curves.append(f['curves']);metadata.append(json.loads(str(f['metadata'])))
    g=np.array(curves);q=[]
    for b in range(7):
        res=(g[1:,b]-g[0,b]-REF[:,b]).ravel();q.append(float(res@cho_solve(FAC[b],res)))
    return dict(point_id=p['id'],omega_m=p['omega_m'],w=p['w'],Q=q,new_snapshot_jobs=new,seconds=time.time()-start)

def main():
    (O/'measurements').mkdir(parents=True,exist_ok=True);lock=open(O/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    writejson(O/'process.json',dict(pid=os.getpid(),started_unix=time.time()))
    start=time.time();cfg=prepare();base=pd.read_csv(O/'source_Q.csv');hq=base[base.branch=='horizontal_only']
    lookup={(round(a,10),round(b,10)):q for a,b,q in hq[['omega_m','w','delta_chi_squared']].to_numpy()}
    results={};maxerr=0.;gate_done=False;fresh_start=None;fresh_count=0
    def accept(result):
        nonlocal maxerr,fresh_count
        err=abs(result['Q'][0]-lookup[(round(result['omega_m'],10),round(result['w'],10))]);assert err<1e-7,(result['point_id'],err)
        maxerr=max(maxerr,err);results[result['point_id']]=result
        if result['new_snapshot_jobs']:fresh_count+=1
        with open(O/'progress.jsonl','a') as f:f.write(json.dumps(result)+'\n')
        elapsed=time.time()-start
        eta=(time.time()-fresh_start)/fresh_count*(1285-len(results)) if fresh_start and fresh_count else None
        writejson(O/'status.json',dict(status='RUNNING',completed_cosmologies=len(results),total_cosmologies=1285,
            fresh_cosmologies=fresh_count,elapsed_seconds=elapsed,eta_seconds=eta,max_horizontal_Q_replay_error=maxerr,gate_passed=gate_done,pid=os.getpid()))
        print(f'{len(results)}/1285; new={fresh_count}; elapsed={elapsed:.1f}s; ETA={eta}; replay={maxerr:.3g}',flush=True)
    cached=[p for p in cfg['points'] if all((O/'measurements'/f'{p["id"]}_snap{s}.npz').exists() for s in r.SN)]
    todo=[p for p in cfg['points'] if p not in cached]
    print(f'REUSED/RESUMED {len(cached)} cosmologies; FRESH {len(todo)}; workers=8',flush=True)
    pool=ProcessPoolExecutor(max_workers=8,initializer=init)
    try:
        for f in as_completed([pool.submit(worker,p) for p in cached]):accept(f.result())
        fresh_start=time.time()
        # Complete an eight-point benchmark/replay gate before dispatching the full scan.
        gate=todo[:8]
        for f in as_completed([pool.submit(worker,p) for p in gate]):accept(f.result())
        gate_done=True;print('EIGHT-POINT FULL-GRID GATE PASSED',flush=True)
        for f in as_completed([pool.submit(worker,p) for p in todo[8:]]):accept(f.result())
    except BaseException:
        pool.shutdown(wait=True,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    assert len(results)==1285
    writejson(O/'point_results.json',results)
    for path,digest in cfg['code_hashes'].items():assert sha(path)==digest
    for path,digest in cfg['input_hashes'].items():assert sha(path)==digest
    for path,digest in cfg['coordinate_source_hashes'].items():assert sha(path)==digest
    writejson(O/'status.json',dict(status='ANALYZING',completed_cosmologies=1285,total_cosmologies=1285,elapsed_seconds=time.time()-start))
    subprocess.run([sys.executable,str(O/'finish.py')],check=True)
    writejson(O/'status.json',dict(status='COMPLETE',completed_cosmologies=1285,total_cosmologies=1285,elapsed_seconds=time.time()-start,max_horizontal_Q_replay_error=maxerr))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        writejson(O/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time_unix=time.time()))
        writejson(O/'status.json',dict(status='FAILED',error=repr(e),resume_command=f'{sys.executable} {__file__}'))
        raise
