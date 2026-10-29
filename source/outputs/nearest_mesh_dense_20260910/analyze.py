"""Analyze only new full-box measurements; fixed verified fiducial baseline."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
from scipy.linalg import cho_factor,cho_solve
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;OLD=ROOT/'paper_figures/fig__intrinsic_subtracted_contours';DEST=OLD/'dense_mesh';DEST.mkdir(exist_ok=True)
plan=json.loads((OUT/'plan.json').read_text());pts=[(.3111,-1.)]+[tuple(v) for v in plan['points'] if tuple(v)!=(.3111,-1.)];SN=plan['snapshots'];BR=['nofg','horizontal_only','horizontal_m1','horizontal_m5'];MOD=['z2-z1','z3-z1','z4-z1','joint'];nu=np.linspace(-2.5,2.5,41);g=np.zeros((4,len(pts),4,41));th=np.zeros_like(g);audit=[];curve_rows=[]
for pi,(om,w) in enumerate(pts):
 for zi,snap in enumerate(SN):
  d=ROOT/'outputs/nearest_mesh_run_20260910/points'/f'snap{snap}_om{om:.7f}_w{w:.7f}';s=json.loads((d/'summary.json').read_text());geo=s['geometry'];assert geo['object_count_before']==geo['object_count_after']==1000000;assert geo['weight_before']==geo['weight_after'];assert s['conservation']['relative_mass_error']<1e-12
  np.testing.assert_allclose(np.array(geo['grid_shape_scan'])*geo['spacing_scan'],geo['box_lengths_scan'],rtol=1e-14)
  for bi,b in enumerate(BR):
   p=d/f'R20_{b}.npz';rec=next(v for v in s['measurements'] if v['branch']==b);assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256']
   with np.load(p) as f:g[zi,pi,bi]=f['curve'];th[zi,pi,bi]=f['thresholds'];np.testing.assert_allclose(f['curve'],f['negative_euler_counts']/geo['retained_volume_scan'],rtol=1e-14,atol=0)
   for ni,x in enumerate(nu):curve_rows.append([om,w,snap,s['config']['z'],b,x,g[zi,pi,bi,ni],th[zi,pi,bi,ni]])
  audit.append(dict(omega_m=om,w=w,snapshot=snap,shape=geo['grid_shape_scan'],spacing=geo['spacing_scan'],volume=geo['retained_volume_scan'],objects=geo['object_count_after'],mass_error=s['conservation']['relative_mass_error']))
a=np.load(OLD/'analysis_arrays.npz');intr=g[1:,0]-g[:1,0];np.testing.assert_array_equal(intr,a['intrinsic']);p=g[1:]-g[:1]-intr[:,None];np.testing.assert_array_equal(p[:,0],0)
rows=[];results=[];diagnostic=[]
for bi,b in enumerate(BR):
 C=a[f'C_v0_b{bi}'];flat=p[:,:,bi,:].transpose(1,0,2).reshape(len(pts),123)
 for mi,(mode,idx) in enumerate(zip(MOD,[np.arange(41),np.arange(41,82),np.arange(82,123),np.arange(123)])):
  for var,v in [('intrinsic_subtracted',flat),('intrinsic_not_subtracted',flat+intr[:,bi,:].ravel())]:
   v=v[:,idx];cc=C[np.ix_(idx,idx)];q=np.sum(v*cho_solve(cho_factor(cc,lower=True),v.T).T,axis=1);white=np.linalg.solve(np.linalg.cholesky(cc),v.T);np.testing.assert_allclose(q,np.sum(white**2,axis=0),rtol=1e-10,atol=1e-10);best=int(q.argmin());dq=q-q[best]
   for (om,w),qq,dd in zip(pts,q,dq):rows.append([var,b,mode,om,w,qq,dd])
   results.append(dict(variant=var,branch=b,mode=mode,Q_min=float(q[best]),Q_sim=float(q[0]),deltaQ_sim=float(dq[0]),best_point=pts[best]))
   if var=='intrinsic_subtracted' and mode=='joint':
    for pi,(om,w) in enumerate(pts):
     if pi and (om==.3111 or w==-1):
      axis='omega_m' if w==-1 else 'w';offset=om-.3111 if axis=='omega_m' else w+1
      diagnostic.append(dict(omega_m=om,w=w,axis=axis,offset=offset,absolute_step=abs(offset),branch=b,joint_Q=float(q[pi]),max_abs_p=float(max(abs(flat[pi])))))
for file,header,r in [('before_after_chisq.csv',['variant','branch','mode','omega_m','w','chi_squared','delta_chi_squared'],rows),('curves.csv',['omega_m','w','snapshot','redshift','branch','nu_f','g','threshold'],curve_rows)]:
 with (DEST/file).open('w') as f:wr=csv.writer(f);wr.writerow(header);wr.writerows(r)
(DEST/'before_after_summary.json').write_text(json.dumps(dict(results=results),indent=2));(DEST/'geometry.json').write_text(json.dumps(audit,indent=2));(DEST/'continuity.json').write_text(json.dumps(diagnostic,indent=2));np.savez_compressed(DEST/'curves.npz',curves=g,points=pts,intrinsic=intr,residuals=p)
with (DEST/'continuity.csv').open('w') as f:wr=csv.DictWriter(f,fieldnames=list(diagnostic[0]));wr.writeheader();wr.writerows(diagnostic)
for name in ['exact_hpd.py','replot_integrated.py']:shutil.copy2(OLD/name,DEST/name)
f=DEST/'exact_hpd.py';s=f.read_text().replace(".22*.8",".008*.036").replace("[0.20,0.42]x[-1.40,-0.60]","[0.3071,0.3151]x[-1.018,-0.982]");f.write_text(s)
f=DEST/'replot_integrated.py';s=f.read_text().replace('xlim=(.2,.42),ylim=(-1.4,-.6)','xlim=(.3071,.3151),ylim=(-1.018,-.982)').replace('[.2,.25,.3,.35,.4]','[.308,.3111,.314]').replace('[-1.4,-1.2,-1,-.8,-.6]','[-1.018,-1,-.982]').replace('Integrated-probability HPD regions;','Full-box nearest mesh: LOCAL HPD regions;').replace('integrated over the displayed scan domain with a uniform prior.','normalized ONLY within the displayed local domain; uniform prior.').replace('Linear interpolation of Q; finite-domain normalization and covariance regularization affect these regions.','No old cropped-grid points included. Local credible regions are not global cosmological constraints.');f.write_text(s)
(DEST/'verification.json').write_text(json.dumps(dict(jobs=len(pts)*4,curves=len(pts)*16,all_objects_retained=True,mass_conservation=True,all_file_hashes_verified=True,geometry_tiling=True,fiducial_exact_replay=True,intrinsic_identical=True,covariance_frozen=True),indent=2))
print('Dense mesh points',len(pts),'all objects retained; covariance fixed')
