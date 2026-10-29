"""Matched full-HI/top1M curves, intrinsic baselines and sample-specific JK/OAS.

Postprocessing only. The reference covariance is estimated independently for
each sample with exactly the existing matched64-block prescription.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,csv,json,shutil
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from sklearn.covariance import oas
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from full_hi_pipeline.genus_grid import sha256_file
BASE=ROOT/'outputs/nearest_mesh_run_20260910';FIG=ROOT/'paper_figures'
plan=json.loads((OUT/'plan.json').read_text());SN=plan['snapshots'];BR=plan['branches'];PTS=np.array(plan['points']);NU=np.linspace(-2.5,2.5,41)
SAMPLES=['top1m','all_halo'];g=np.zeros((2,4,len(PTS),4,41));ths=np.zeros_like(g);G=np.zeros_like(g,dtype=np.int64)
loo=np.zeros((2,4,4,64,41));amp=np.zeros((2,4,len(PTS),4));audit={};sample_rows=[]
for si,sample in enumerate(SAMPLES):
    for zi,snap in enumerate(SN):
        sm=json.loads((ROOT/f'outputs/top_hi_1m_weighted_rsd_all13_20260908_v1/snap{snap}.json').read_text())
        summary=json.loads((OUT/f'snap{snap}/density_summary.json').read_text())
        sample_rows.append(dict(sample=sample,snapshot=snap,z=sm['redshift'],objects=1000000 if si==0 else summary['valid_objects'],HI_fraction=sm['HI_fraction'] if si==0 else 1.,n_eff=sm['n_eff_top'] if si==0 else summary['n_eff'],HI_weight=summary['HI_weight']*(sm['HI_fraction'] if si==0 else 1.)))
        for pi,(om,w) in enumerate(PTS):
            d=(BASE/'points'/f'snap{snap}_om{om:.7f}_w{w:.7f}') if si==0 else OUT/f'snap{snap}'/f'om{om:.7f}_w{w:.7f}'
            s=json.loads((d/'summary.json').read_text())
            for bi,b in enumerate(BR):
                f=d/f'R20_{b}.npz';h=sha256_file(f)
                rec=next(r for r in (s['measurements'] if si==0 else s['files']) if r['file']==f.name)
                assert h==rec['sha256'];audit[str(f.relative_to(ROOT))]=h
                with np.load(f) as a:
                    g[si,zi,pi,bi]=a['curve'];ths[si,zi,pi,bi]=a['thresholds'];G[si,zi,pi,bi]=a['negative_euler_counts'];amp[si,zi,pi,bi]=a['amplitude_gaussian']
                    np.testing.assert_allclose(a['curve'],a['negative_euler_counts']/s['geometry']['retained_volume_scan'],rtol=1e-14)
        blocks=(ROOT/f'outputs/nearest_mesh_figure_refresh_20260910/c0_snap{snap}.npz') if si==0 else OUT/f'snap{snap}/fiducial_blocks.npz'
        audit[str(blocks.relative_to(ROOT))]=sha256_file(blocks)
        with np.load(blocks) as a:
            np.testing.assert_array_equal(a['curves'],g[si,zi,0]);np.testing.assert_allclose(a['contributions'].sum(1),a['curves'],atol=1e-20,rtol=1e-13)
            loo[si,zi]=(a['curves'][:,None]-a['contributions'])/(1-a['fractions'][None,:,None])
intr=g[:,:,0][:,1:]-g[:,:,0][:,:1]
res=g[:,1:]-g[:,:1]-intr[:,:,None]
np.testing.assert_array_equal(res[:,:,0],0)
cov=np.zeros((2,4,123,123));rawcov=np.zeros_like(cov);covmeta=[];covvariants={}
ref=np.load(FIG/'fig__intrinsic_subtracted_contours/analysis_arrays.npz')
for si,sample in enumerate(SAMPLES):
    for bi,b in enumerate(BR):
        x=(loo[si,1:,bi]-loo[si,:1,bi]).transpose(1,0,2).reshape(64,123);x-=x.mean(0)
        raw=63/64*(x.T@x);std=np.sqrt(np.diag(raw));assert np.all(std>0)
        Z=x/np.sqrt(np.mean(x*x,axis=0));corr,rho=oas(Z,assume_centered=True)
        cov[si,bi]=corr*std[:,None]*std[None,:];rawcov[si,bi]=raw
        if si==0:np.testing.assert_allclose(cov[si,bi],ref[f'C_v0_b{bi}'],rtol=1e-10,atol=1e-27)
        S=Z.T@Z/64;mu=np.trace(S)/123
        for label,mult in [('half_shrinkage',.5),('double_shrinkage',2.)]:
            lam=min(1.,rho*mult);covvariants[(si,bi,label)]=((1-lam)*S+lam*mu*np.eye(123))*std[:,None]*std[None,:]
        covmeta.append(dict(sample=sample,branch=b,rho=float(rho),dimension=123,raw_rank=int(np.linalg.matrix_rank(raw)),condition=float(np.linalg.cond(cov[si,bi]))))
def write(name,rows):
    with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('sample_selection.csv',sample_rows);write('covariance_summary.csv',covmeta)
curve_rows=[];comparison=[];amprows=[]
template=(1-NU**2)*np.exp(-NU**2/2)
for si,sample in enumerate(SAMPLES):
    for zi,snap in enumerate(SN):
        z=next(r['z'] for r in sample_rows if r['snapshot']==snap)
        for pi,(om,w) in enumerate(PTS):
            for bi,b in enumerate(BR):
                for ni,nu in enumerate(NU):curve_rows.append(dict(sample=sample,snapshot=snap,z=z,omega_m=om,w=w,branch=b,nu_f=nu,g=g[si,zi,pi,bi,ni],threshold=ths[si,zi,pi,bi,ni],negative_euler=int(G[si,zi,pi,bi,ni])))
        for bi,b in enumerate(BR):
            lv=loo[si,zi,bi];centered=lv-lv.mean(0);err=np.sqrt(63/64*np.sum(centered**2,axis=0))
            aa=lv@template/(template@template)*8000;asigma=np.sqrt(63/64*np.sum((aa-aa.mean())**2))
            amprows.append(dict(sample=sample,snapshot=snap,z=z,branch=b,AR3=amp[si,zi,0,bi]*8000,AR3_sigma=asigma))
            diffloo=loo[1,zi,bi]-loo[0,zi,bi];xx=diffloo-diffloo.mean(0);differr=np.sqrt(63/64*np.sum(xx**2,axis=0))
            for ni,nu in enumerate(NU):comparison.append(dict(sample=sample,snapshot=snap,z=z,branch=b,nu_f=nu,g=g[si,zi,0,bi,ni],sigma=err[ni],all_minus_top=g[1,zi,0,bi,ni]-g[0,zi,0,bi,ni],all_minus_top_paired_sigma=differr[ni]))
write('curves.csv',curve_rows);write('fiducial_comparison.csv',comparison);write('amplitudes.csv',amprows)
qrows=[];sens=[];response=[]
for si,sample in enumerate(SAMPLES):
    for bi,b in enumerate(BR):
        residual=res[si,:,:,bi,:].transpose(1,0,2).reshape(len(PTS),123)
        for mode,ix in [('joint',np.arange(123)),('z2-z1',np.arange(41)),('z3-z1',np.arange(41,82)),('z4-z1',np.arange(82,123))]:
            for variant,v in [('intrinsic_subtracted',residual),('intrinsic_not_subtracted',residual+intr[si,:,bi,:].ravel())]:
                vv=v[:,ix];C=cov[si,bi][np.ix_(ix,ix)];q=np.sum(vv*cho_solve(cho_factor(C,lower=True),vv.T).T,axis=1)
                white=np.linalg.solve(np.linalg.cholesky(C),vv.T);np.testing.assert_allclose(q,np.sum(white**2,axis=0),rtol=1e-10,atol=1e-10)
                for (om,w),qq in zip(PTS,q):qrows.append(dict(sample=sample,variant=variant,branch=b,mode=mode,omega_m=om,w=w,chi_squared=qq,delta_chi_squared=qq-q.min()))
        for case in ['own_covariance','top1m_covariance','half_shrinkage','double_shrinkage']:
            C=cov[si,bi] if case=='own_covariance' else (cov[0,bi] if case=='top1m_covariance' else covvariants[(si,bi,case)])
            q=np.sum(residual*cho_solve(cho_factor(C,lower=True),residual.T).T,axis=1)
            for pi,((om,w),qq) in enumerate(zip(PTS,q)):
                sens.append(dict(sample=sample,branch=b,covariance_case=case,omega_m=om,w=w,chi_squared=qq))
                if case=='own_covariance':response.append(dict(sample=sample,branch=b,omega_m=om,w=w,residual_rms=float(np.sqrt(np.mean(residual[pi]**2))),joint_Q=qq))
write('chisq.csv',qrows);write('covariance_sensitivity.csv',sens);write('ap_response.csv',response)
cr=[]
for si,sample in enumerate(SAMPLES):
    for bi,b in enumerate(BR):
        for i in range(123):
            for j in range(123):cr.append(dict(sample=sample,branch=b,i=i,j=j,covariance=cov[si,bi,i,j]))
write('covariance.csv',cr)
np.savez_compressed(OUT/'analysis_arrays.npz',g=g,thresholds=ths,counts=G,intrinsic=intr,residuals=res,loo=loo,covariance=cov,raw_covariance=rawcov,points=PTS,nu=NU)
for sample in SAMPLES:
    d=OUT/f'hpd_{sample}';d.mkdir(exist_ok=True)
    rr=[{k:v for k,v in r.items() if k!='sample'} for r in qrows if r['sample']==sample]
    with (d/'before_after_chisq.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    shutil.copy2(FIG/'fig__intrinsic_subtracted_contours/extent_mesh/exact_hpd.py',d/'exact_hpd.py')
(OUT/'input_manifest.json').write_text(json.dumps(audit,indent=2))
(OUT/'analysis_verification.json').write_text(json.dumps(dict(status='PASS',all_curves=592,top1m_curves=592,all_curve_hashes_verified=True,negative_euler_volume_identity=True,own_intrinsic_per_sample=True,own_covariance_per_sample=True,top1m_covariance_reproduced=True,block_sums_match_full_counts=True,whitened_Q_verified=True,paired_sample_difference_errors=True,scope='conditional comparison on common local37-point domain; independent coverage not calibrated'),indent=2))
print('PASS: independent sample baselines and covariances, matched top1m covariance replay, all curve hashes and white Q checked.')
