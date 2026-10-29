"""Bounded full-catalog comparison, preserving open-edge CIC and Euler.

At fixed integer shape the normalized CIC coordinates p'/d' = p*N/L
are AP-invariant. Each shape is streamed once; physical Fourier spacings
and filters are recalculated at every cosmology. Replay gates check this
optimization against existing direct-AP top-1M measurements.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'paper_figures/fig__cosmo_redshift_evolution'))
from full_hi_pipeline import scan_sim_fixed_cell as old,scan_sim_nearest_mesh as mesh
from full_hi_pipeline.particle_first_density import valid_objects
from full_hi_pipeline.stream_density import JIUTIAN_DTYPE
from full_hi_pipeline.fast_counting import _encode_ranks
from full_hi_pipeline.genus_grid import _atomic_json,_atomic_npz,sha256_file
from matched_stencil_source import matched_hist
BASE=ROOT/'outputs/nearest_mesh_run_20260910';SELECT=ROOT/'outputs/top_hi_1m_weighted_rsd_all13_20260908_v1'
PLAN=json.loads((BASE/'plan.json').read_text());SN=PLAN['snapshots'];BR=[0,1,2,6];NU=np.linspace(-2.5,2.5,41)
POINTS=[(.3111,-1.)]+[tuple(p) for p in PLAN['points'] if tuple(p)!=(.3111,-1.)]

@njit(cache=True)
def deposit_scalar(p,w,m,spacing):
    nx,ny,nz=m.shape
    for v in range(len(w)):
        x=p[v,0]/spacing[0]-.5;y=p[v,1]/spacing[1]-.5;z=p[v,2]/spacing[2]-.5
        ix=int(np.floor(x));iy=int(np.floor(y));iz=int(np.floor(z))
        fx=x-ix;fy=y-iy;fz=z-iz;norm=0.
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    if 0<=ix+a<nx and 0<=iy+b<ny and 0<=iz+c<nz:
                        norm+=(fx if a else 1-fx)*(fy if b else 1-fy)*(fz if c else 1-fz)
        if norm<=0:raise ValueError('Object outside full box')
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    if 0<=ix+a<nx and 0<=iy+b<ny and 0<=iz+c<nz:
                        m[ix+a,iy+b,iz+c]+=w[v]/norm*(fx if a else 1-fx)*(fy if b else 1-fy)*(fz if c else 1-fz)

def configs(snap):
    result=[]
    for om,w in POINTS:
        p=BASE/'points'/f'snap{snap}_om{om:.7f}_w{w:.7f}'
        s=json.loads((p/'summary.json').read_text())
        assert s['config']['core_sha256']==sha256_file(Path(mesh.__file__))
        assert s['config']['old_core_sha256']==sha256_file(Path(old.__file__))
        result.append((om,w,p,s))
    return result

def replay_gate():
    rng=np.random.default_rng(603);p=rng.uniform(0,40,(10000,3));w=rng.uniform(.01,10,10000)
    p[:4]=[[0,0,0],[39.999,39.999,39.999],[2,2,2],[38,38,38]]
    for sh in [(10,10,10),(9,10,11)]:
        sp=40/np.array(sh);a=np.zeros(sh);b=np.zeros(sh)
        deposit_scalar(p,w,a,sp);old._cic(p,w,sp,b)
        np.testing.assert_array_equal(a,b)
    # One direct-AP representative per shape per snapshot, all four branches.
    records=[]
    for snap in SN:
        cfg=configs(snap);meta=json.loads((SELECT/f'snap{snap}.json').read_text())
        with np.load(SELECT/f'snap{snap}_selection.npz') as f:r=f['records']
        p,v,w,_,_=valid_objects(r);p=np.mod(old.add_rsd_sim(p,v,meta['redshift']),1000.)
        seen=set()
        for om,ww,path,s in cfg:
            sh=tuple(s['geometry']['grid_shape_scan'])
            if sh in seen:continue
            seen.add(sh);mass=np.zeros(sh);t=time.time();deposit_scalar(p,w,mass,1000/np.array(sh))
            delta=(mass/(w.sum()/mass.size)-1).astype(np.float32)
            spec,axes,shape,sl=mesh.prepare_spectrum(delta,s['geometry']['spacing_scan'],160)
            for b in BR:
                field=old.filtered_field(spec,axes,shape,sl,20,b);m=mesh.measure(field,s['geometry']['spacing_scan'],NU)
                with np.load(path/f'R20_{old.LABELS[b]}.npz') as f:
                    np.testing.assert_array_equal(m['negative_euler_counts'],f['negative_euler_counts'])
                    np.testing.assert_allclose(m['thresholds'],f['thresholds'],rtol=0,atol=2e-7)
                del field
            del mass,delta,spec
            records.append(dict(snapshot=snap,shape=sh,omega_m=om,w=ww,all_four_counts_exact=True,seconds=time.time()-t))
            print('REPLAY',snap,sh,'passed',flush=True)
    _atomic_json(OUT/'replay_gate.json',dict(scalar_CIC_bitwise_identical=True,replays=records,producer_sha256=sha256_file(Path(__file__))))

def worker(snap):
    meta=json.loads((SELECT/f'snap{snap}.json').read_text());src=Path(meta['source_path']);stat=src.stat()
    cfg=configs(snap);shapes=sorted(set(tuple(s['geometry']['grid_shape_scan']) for _,_,_,s in cfg))
    d=OUT/f'snap{snap}';d.mkdir(exist_ok=True)
    contract=dict(snapshot=snap,source_sha256=meta['source_sha256'],producer_sha256=sha256_file(Path(__file__)),core_sha256=sha256_file(Path(old.__file__)),mesh_sha256=sha256_file(Path(mesh.__file__)),shapes=shapes,R=20,padding=160,selection='all finite position/velocity and positive finite HI; HI-mass weighted',points=POINTS)
    contract=json.loads(json.dumps(contract));start=time.time()
    if (d/'density_summary.json').exists():
        summary=json.loads((d/'density_summary.json').read_text());assert summary['config']==contract
        for r in summary['densities']:assert sha256_file(d/r['file'])==r['sha256']
    else:
        masses={sh:np.zeros(sh,np.float64) for sh in shapes};total=0;valid=0;weight=0.;weight2=0.;bad={'nonfinite':0,'nonpositive':0};dig=hashlib.sha256()
        with src.open('rb') as h:
            while True:
                raw=h.read(40*1000000)
                if not raw:break
                dig.update(raw);r=np.frombuffer(raw,dtype=JIUTIAN_DTYPE)
                p,v,w,_,exclude=valid_objects(r,backend='numba');p=np.mod(old.add_rsd_sim(p,v,meta['redshift']),1000.)
                for sh,m in masses.items():deposit_scalar(p,w,m,1000/np.array(sh))
                total+=len(r);valid+=len(w);weight+=w.sum();weight2+=w@w
                for key in bad:bad[key]+=exclude[key]
                if total%10000000==0:
                    _atomic_json(d/'progress.json',dict(stage='streaming_catalog',records=total,total_records=stat.st_size//40,seconds=time.time()-start,shapes=len(shapes)))
                    print('STREAM',snap,total,'/',stat.st_size//40,round(time.time()-start,1),flush=True)
        assert src.stat().st_mtime_ns==stat.st_mtime_ns and src.stat().st_size==stat.st_size
        assert dig.hexdigest()==meta['source_sha256'];assert total==meta['source_records'];assert valid==meta['n_valid_all']
        density_rows=[]
        for sh,m in masses.items():
            error=abs(m.sum()-weight)/weight;assert error<1e-11
            delta=(m/(weight/m.size)-1).astype(np.float32);filename='delta_'+'x'.join(map(str,sh))+'.npy';np.save(d/filename,delta)
            density_rows.append(dict(shape=sh,file=filename,sha256=sha256_file(d/filename),mass_relative_error=error))
        summary=dict(config=contract,source_records=total,valid_objects=valid,excluded=bad,HI_weight=weight,HI_weight_squared=weight2,n_eff=weight**2/(1e9*weight2),densities=density_rows,stream_seconds=time.time()-start)
        _atomic_json(d/'density_summary.json',summary);del masses
    cache={}
    for om,w,oldpath,s in cfg:
        name=f'om{om:.7f}_w{w:.7f}';dest=d/name;dest.mkdir(exist_ok=True)
        if (dest/'summary.json').exists():
            done=json.loads((dest/'summary.json').read_text())
            for f in done['files']:assert sha256_file(dest/f['file'])==f['sha256']
            continue
        geo=s['geometry'];sh=tuple(geo['grid_shape_scan']);sp=np.array(geo['spacing_scan']);vol=geo['retained_volume_scan']
        filename='delta_'+'x'.join(map(str,sh))+'.npy'
        delta=np.load(d/filename);spec,axes,shape,sl=mesh.prepare_spectrum(delta,sp,160);del delta
        curves=[];thresholds=[];contributions=[];sig=[];amplitudes=[];records=[]
        fid=(om==.3111 and w==-1)
        for b in BR:
            field=old.filtered_field(spec,axes,shape,sl,20,b);m=mesh.measure(field,sp,NU)
            f=f'R20_{old.LABELS[b]}.npz';_atomic_npz(dest/f,**m);records.append(dict(file=f,sha256=sha256_file(dest/f)))
            if fid:
                edges=np.array([np.clip(np.ceil(np.arange(5)*nn/4-.5).astype(int),0,nn) for nn in sh]);edges[:,0]=0;edges[:,-1]=sh
                ranks=np.empty(sh,np.uint8);assert not _encode_ranks(field,m['thresholds'],ranks)
                hist,sizes=matched_hist(ranks,edges,41);ct=np.cumsum(hist[:,::-1],axis=1)[:,::-1][:,1:]
                np.testing.assert_array_equal(ct.sum(0),m['negative_euler_counts']);contributions.append(ct/vol);del ranks
                curves.append(m['curve']);thresholds.append(m['thresholds']);sig.append(m['field_std']);amplitudes.append(m['amplitude_gaussian'])
            del field
        del spec
        if fid:_atomic_npz(d/'fiducial_blocks.npz',curves=curves,thresholds=thresholds,contributions=contributions,fractions=sizes/sizes.sum(),sigma=sig,amplitudes=amplitudes)
        _atomic_json(dest/'summary.json',dict(snapshot=snap,omega_m=om,w=w,geometry=geo,selection='all positive-HI valid catalog records',geometry_metadata_origin='matched top1m geometry only; top1m object/weight fields are not all-catalog counts',files=records))
        _atomic_json(d/'progress.json',dict(stage='measuring_AP_points',last_point=[om,w],seconds=time.time()-start))
        print('POINT',snap,om,w,round(time.time()-start,1),flush=True)
    _atomic_json(d/'progress.json',dict(stage='COMPLETE',points=len(POINTS),curves=len(POINTS)*4,seconds=time.time()-start))
    return snap

if __name__=='__main__':
    if '--gate' in sys.argv:replay_gate()
    else:
        gate=json.loads((OUT/'replay_gate.json').read_text());assert gate['producer_sha256']==sha256_file(Path(__file__))
        _atomic_json(OUT/'plan.json',dict(snapshots=SN,points=POINTS,branches=[old.LABELS[b] for b in BR],R=20,target_spacing=4,padding=160,workers=4,all_curves=592,top_curves_reused=592,all_covariance='recompute matched64-block OAS at own fiducial',scope='matched local comparison; no reuse of top1m intrinsic/covariance for all-sample contours'))
        start=time.time();done=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for fut in as_completed([pool.submit(worker,s) for s in SN]):
                done.append(fut.result());_atomic_json(OUT/'status.json',dict(completed_snapshots=done,total_snapshots=4,seconds=time.time()-start,status='COMPLETE' if len(done)==4 else 'RUNNING'))
