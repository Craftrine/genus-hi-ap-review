"""Actual nine-m pilot; isolated outputs, exact production geometry and estimator."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,fcntl
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.fft import irfftn
from scipy.linalg import cho_factor,cho_solve
from sklearn.covariance import oas
O=Path(__file__).resolve().parent;P=O.parents[1]
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'paper_figures/fig__cosmo_redshift_evolution'))
from full_hi_pipeline import scan_sim_fixed_cell as old,scan_sim_nearest_mesh as n
from full_hi_pipeline.particle_first_density import valid_objects
from full_hi_pipeline.fast_counting import _encode_ranks
from matched_stencil_source import matched_hist
SN=[118,112,105,94];NU=np.linspace(-2.5,2.5,41)
MS=np.linspace(old.M_SCAN[2],old.M_SCAN[4],9);MS[4]=old.M_SCAN[3]
S=P/'outputs/wedge_all_m_20260920';CATS={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plan():
    d=pd.read_csv(S/'joint_chisq.csv');points=[dict(id='sim',omega_m=.3111,w=-1.,selected_for='reference')];chosen=set()
    areas=pd.read_csv(P/'outputs/m_area_ratios_20260923/areas_and_ratios.csv')
    for br in ['horizontal_m3','horizontal_m4','horizontal_m5']:
        s=d[d.branch==br].copy()
        level=areas[(areas.branch==br)&(areas.grid==800)&(areas.variant=='no_gaussian')&(areas.target==.683)].Q_level.iloc[0]
        ang=np.arctan2((s.w.to_numpy()+1)/.025,(s.omega_m.to_numpy()-.3111)/.01)
        for target in [0.,np.pi/2,np.pi,-np.pi/2]:
            score=abs(np.log((s.delta_chi_squared.to_numpy()+1e-12)/level))+2*abs(np.angle(np.exp(1j*(ang-target))))
            for k in np.argsort(score):
                row=s.iloc[k];key=(float(row.omega_m),float(row.w))
                if key in chosen or row.delta_chi_squared<.05:continue
                chosen.add(key);points.append(dict(id=f'p{len(points):02d}',omega_m=key[0],w=key[1],selected_for=br,target_angle=float(target),baseline_Q=float(row.delta_chi_squared)));break
    assert len(points)==13
    sources=[P/f'outputs/top_hi_1m_weighted_rsd_all13_20260908_v1/snap{s}_selection.npz' for s in SN]
    for snap,src in zip(SN,sources):
        cfg=json.loads((P/f'outputs/scan_sim_fixed_cell_20260909_v1/four_branch_campaign/points/snap{snap}_om0.3111000_w-1.0000000_d4_p160/config.json').read_text())
        assert sha(src)==cfg['source_sha256']
    result=dict(points=points,m_values=MS.tolist(),snapshots=SN,branches=['horizontal_only']+[f'm{i}' for i in range(9)],
        target_spacing=4,padding_cells=160,R=20,k_parallel_cut=.1,n_curves=520,workers=4,
        scope='13 cosmologies x 4 snapshots x (9 m values + horizontal control); no area inference',
        sources={str(f):sha(f) for f in sources},core_hashes={str(f):sha(f) for f in [Path(old.__file__),Path(n.__file__),Path(__file__)]})
    (O/'plan.json').write_text(json.dumps(result,indent=2));return result

def init():
    for snap in SN:
        src=P/f'outputs/top_hi_1m_weighted_rsd_all13_20260908_v1/snap{snap}_selection.npz'
        z=json.loads(src.with_name(f'snap{snap}.json').read_text())['redshift']
        with np.load(src) as f:rec=f['records']
        p,v,wt,_,_=valid_objects(rec);CATS[snap]=(old.add_rsd_sim(p,v,z),wt,z)

def worker(job):
    point,snap=job;start=time.time();fp=O/'measurements'/f'{point["id"]}_snap{snap}.npz'
    if fp.exists():return dict(file=fp.name,cached=True,seconds=0.)
    p0,wt,z=CATS[snap];om,w=point['omega_m'],point['w']
    pp,w2,g=n.geometry(p0,wt,old.ap_scan(z,om,w));delta,cons=n.deposit(pp,w2,g['grid_shape_scan'],g['spacing_scan']);del pp
    spectrum,axes,shape,sl=n.prepare_spectrum(delta,g['spacing_scan'],160);del delta
    for axis,k in enumerate(axes):
        view=[1,1,1];view[axis]=len(k);spectrum*=np.exp(-.5*(20*k)**2).astype(np.float32).reshape(view)
    kp=np.sqrt(axes[0][:,None]**2+axes[1][None,:]**2);kz=abs(axes[2])[None,None,:];low=kz<.1
    curves=[];ths=[];counts=[];contributions=[];std=[];mask_counts=[]
    if point['id']=='sim':
        edges=np.array([np.clip(np.ceil(np.arange(5)*nn/4-.5).astype(int),0,nn) for nn in g['grid_shape_scan']]);edges[:,0]=0;edges[:,-1]=g['grid_shape_scan']
    for i,mvalue in enumerate([None]+list(MS)):
        mask=low if mvalue is None else low|(kz<=mvalue*kp[:,:,None])
        work=spectrum.copy();work*=~mask;work[0,0,0]=0
        padded=irfftn(work,s=shape,workers=1);field=np.ascontiguousarray(padded[sl]);del work,padded
        result=n.measure(field,g['spacing_scan'],NU)
        curves.append(result['curve']);ths.append(result['thresholds']);counts.append(result['negative_euler_counts']);std.append(result['field_std'])
        if point['id']=='sim':
            ranks=np.empty(field.shape,np.uint8);assert not _encode_ranks(field,result['thresholds'],ranks)
            hist,sizes=matched_hist(ranks,edges,41);cc=np.cumsum(hist[:,::-1],axis=1)[:,::-1][:,1:]
            np.testing.assert_array_equal(cc.sum(0),result['negative_euler_counts']);contributions.append(cc/g['retained_volume_scan']);del ranks
        mask_counts.append(int(np.count_nonzero(mask))*(shape[0]*shape[1] if mvalue is None else 1));del field
    data=dict(curves=np.array(curves),thresholds=np.array(ths),counts=np.array(counts),sigma=np.array(std),omega_m=om,w=w,z=z,
        metadata=json.dumps(dict(geometry=g,conservation=cons,fft_shape=list(map(int,shape)),removed_rfft_bins=mask_counts)),m_values=MS)
    if point['id']=='sim':data.update(contributions=np.array(contributions),fractions=sizes/sizes.sum())
    tmp=fp.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);tmp.replace(fp)
    return dict(file=fp.name,cached=False,seconds=time.time()-start)

def reference():
    data=[np.load(O/'measurements'/f'sim_snap{s}.npz') for s in SN]
    g=np.array([f['curves'] for f in data]);intr=g[1:]-g[0]
    loo=np.array([(f['curves'][:,None,:]-f['contributions'])/(1-f['fractions'][None,:,None]) for f in data])
    cov=[];diagnostic=[]
    for b in range(10):
        x=(loo[1:,b]-loo[0,b]).transpose(1,0,2).reshape(64,123);x-=x.mean(0)
        raw=63/64*x.T@x;sd=np.sqrt(np.diag(raw));corr,rho=oas(x/np.sqrt(np.mean(x*x,axis=0)),assume_centered=True);C=corr*sd[:,None]*sd[None,:]
        cho_factor(C);cov.append(C);diagnostic.append(dict(branch=b,shrinkage=float(rho),raw_rank=int(np.linalg.matrix_rank(raw))))
    previous=np.load(S/'reference_covariances.npz');errors=[]
    for b,oldb in [(0,1),(1,4),(5,5),(9,6)]:
        np.testing.assert_array_equal(g[:,b],previous['curves'][:,oldb]);np.testing.assert_allclose(cov[b],previous['covariances'][oldb],rtol=1e-10,atol=1e-30)
        errors.append(float(np.max(abs(cov[b]-previous['covariances'][oldb]))))
    np.savez_compressed(O/'references.npz',curves=g,intrinsic=intr,covariances=cov)
    (O/'reference_validation.json').write_text(json.dumps(dict(old_curves_exact=True,max_cov_abs_error=max(errors),covariance_diagnostics=diagnostic),indent=2))
    for f in data:f.close()
    return intr,np.array(cov)

def analyze(cfg,intr,cov,gate_only=False):
    oldq=pd.read_csv(S/'joint_chisq.csv');rows=[];res=[];errs=[];curverows=[]
    factors=[cho_factor(c) for c in cov];common=factors[5]
    for point in cfg['points']:
        paths=[O/'measurements'/f'{point["id"]}_snap{s}.npz' for s in SN]
        if not all(p.exists() for p in paths):continue
        cc=np.array([np.load(p)['curves'] for p in paths])
        for b,mvalue in enumerate([np.nan]+list(MS)):
            r=(cc[1:,b]-cc[0,b]-intr[:,b]).ravel();q=float(r@cho_solve(factors[b],r));qc=float(r@cho_solve(common,r))
            rows.append(dict(point_id=point['id'],omega_m=point['omega_m'],w=point['w'],branch=b,m=mvalue,Q=q,Q_common_C_m4=qc,residual_norm=float(np.linalg.norm(r))))
            res.append(r)
            if b in [0,1,5,9]:
                br={0:'horizontal_only',1:'horizontal_m3',5:'horizontal_m4',9:'horizontal_m5'}[b]
                s=oldq[(oldq.branch==br)&(abs(oldq.omega_m-point['omega_m'])<1e-11)&(abs(oldq.w-point['w'])<1e-11)]
                assert len(s)==1;error=abs(q-s.delta_chi_squared.iloc[0]);errs.append(float(error));assert error<1e-7,(point,b,error)
            for si,snap in enumerate(SN):
                for k,nu in enumerate(NU):curverows.append(dict(point_id=point['id'],branch=b,m=mvalue,snapshot=snap,nu=nu,genus=cc[si,b,k]))
    pd.DataFrame(rows).to_csv(O/'point_Q.csv',index=False);pd.DataFrame(curverows).to_csv(O/'curves.csv',index=False)
    np.savez_compressed(O/'residuals.npz',residuals=res)
    (O/('gate_validation.json' if gate_only else 'validation.json')).write_text(json.dumps(dict(old_Q_max_abs_error=max(errs),replayed_Q_count=len(errs),n_curves=len(curverows)//41,n_parameter_points=len(rows)//10,gate_only=gate_only),indent=2))

if __name__=='__main__':
    O.mkdir(exist_ok=True);(O/'measurements').mkdir(exist_ok=True)
    lock=open(O/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();cfg=plan();print('PLAN',json.dumps(cfg['points']),flush=True)
    timings=[]
    def run_batch(pool,points):
        jobs=[(p,s) for p in points for s in SN]
        for f in as_completed([pool.submit(worker,j) for j in jobs]):
            result=f.result();timings.append(result);elapsed=time.time()-start
            status=dict(status='RUNNING',completed_jobs=len(timings),total_jobs=52,elapsed_seconds=elapsed,last=result)
            (O/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    with ProcessPoolExecutor(max_workers=4,initializer=init) as pool:
        run_batch(pool,cfg['points'][:1]);intr,cov=reference();print('SIM REFERENCE GATE PASSED',flush=True)
        run_batch(pool,cfg['points'][1:2]);analyze(cfg,intr,cov,True);print('NONFIDUCIAL Q GATE PASSED',flush=True)
        run_batch(pool,cfg['points'][2:])
    analyze(cfg,intr,cov)
    for path,digest in cfg['core_hashes'].items():assert sha(path)==digest
    (O/'timings.json').write_text(json.dumps(timings,indent=2))
    (O/'status.json').write_text(json.dumps(dict(status='COMPLETE',completed_jobs=52,total_jobs=52,n_curves=520,elapsed_seconds=time.time()-start),indent=2))
    print('COMPLETE',time.time()-start,flush=True)
