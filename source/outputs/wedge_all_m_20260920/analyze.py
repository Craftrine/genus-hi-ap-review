import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import ast,json,hashlib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from sklearn.covariance import oas
from scipy.spatial import Delaunay,ConvexHull,cKDTree
from scipy.optimize import brentq
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import path as mpath
P=Path('/Users/LOCAL_USER/Documents/ChatGPT/genus');O=Path(__file__).parent;D=P.parent/'genus_workbuddy/dense_band_20260917'
labels=['nofg','horizontal_only']+[f'horizontal_m{i}' for i in range(1,6)]
ms=[np.nan,0,*np.linspace(.2938445235861259,1.6921831690611855,5)]
curves=[];loo=[];validation=[];manifest=[]
for sn in [118,112,105,94]:
 oldfile=P/f'outputs/nearest_mesh_figure_refresh_20260910/c0_snap{sn}.npz';newfile=O/f'c0_snap{sn}.npz'
 a=np.load(oldfile);b=np.load(newfile)
 np.testing.assert_array_equal(a['curves'][0],b['curves'][0]);np.testing.assert_array_equal(a['contributions'][0],b['contributions'][0]);np.testing.assert_array_equal(a['fractions'],b['fractions'])
 cc=np.array([a['curves'][0],a['curves'][1],a['curves'][2],b['curves'][1],b['curves'][2],b['curves'][3],a['curves'][3]])
 tt=np.array([a['contributions'][0],a['contributions'][1],a['contributions'][2],b['contributions'][1],b['contributions'][2],b['contributions'][3],a['contributions'][3]])
 curves.append(cc);loo.append((cc[:,None,:]-tt)/(1-a['fractions'][None,:,None]))
 for f in [oldfile,newfile]:manifest.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
g=np.array(curves);l=np.array(loo);intr=g[1:]-g[0];covs=[];ch=[];base=np.load(P/'paper_figures/fig__intrinsic_subtracted_contours/analysis_arrays.npz')
for b,br in enumerate(labels):
 x=(l[1:,b]-l[0,b]).transpose(1,0,2).reshape(64,123);x-=x.mean(0);raw=63/64*x.T@x;sd=np.sqrt(np.diag(raw));corr,rho=oas(x/np.sqrt(np.mean(x*x,axis=0)),assume_centered=True);C=corr*sd[:,None]*sd[None,:]
 covs.append(C);ch.append(cho_factor(C,lower=True));val=dict(branch=br,shrinkage=rho,raw_rank=int(np.linalg.matrix_rank(raw)))
 if b in [0,1,2,6]:
  bi=[0,1,2,6].index(b);np.testing.assert_allclose(C,base[f'C_v0_b{bi}'],rtol=1e-10,atol=1e-30)
 validation.append(val)
np.savez_compressed(O/'reference_covariances.npz',covariances=covs,intrinsic=intr,curves=g)
rows=[]
for fp in sorted((D/'points_A').glob('om*.npz')):
 with np.load(fp) as f:
  om,w=float(f['omega_m']),float(f['w']);cc=f['curves']
 for b,br in enumerate(labels):
  rr=(cc[b,1:]-cc[b,0]-intr[:,b]).ravel();Q=float(rr@cho_solve(ch[b],rr));rows.append(dict(branch=br,mode='joint',omega_m=om,w=w,delta_chi_squared=Q))
 manifest.append(dict(path=str(fp),sha256=hashlib.sha256(fp.read_bytes()).hexdigest()))
oldbase=pd.read_csv(P/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/before_after_chisq.csv')
oldbase=oldbase[(oldbase.variant=='intrinsic_subtracted')&(oldbase['mode']=='joint')]
for rr in oldbase.to_dict('records'):rows.append({k:rr[k] for k in ['branch','mode','omega_m','w','delta_chi_squared']})
oldpoints=sorted((O/'old_points').glob('om*.npz'));assert len(oldpoints)==233
oldmap={(round(float(rr['omega_m']),7),round(float(rr['w']),7)):float(rr['delta_chi_squared']) for rr in oldbase[oldbase.branch=='nofg'].to_dict('records')}
old_replay_error=0.
for fp in oldpoints:
 with np.load(fp) as f:om,w=float(f['omega_m']),float(f['w']);cc=f['curves']
 rr=(cc[0,1:]-cc[0,0]-intr[:,0]).ravel();Q=float(rr@cho_solve(ch[0],rr));old_replay_error=max(old_replay_error,abs(Q-oldmap[(round(om,7),round(w,7))]))
 for j,b in enumerate([3,4,5],1):
  rr=(cc[j,1:]-cc[j,0]-intr[:,b]).ravel();Q=float(rr@cho_solve(ch[b],rr));rows.append(dict(branch=labels[b],mode='joint',omega_m=om,w=w,delta_chi_squared=Q))
 manifest.append(dict(path=str(fp),sha256=hashlib.sha256(fp.read_bytes()).hexdigest()))
assert old_replay_error<1e-8,old_replay_error
t=pd.DataFrame(rows)
# Preserve the exact displayed-grid coordinates and ordering for all branches.
reference=pd.read_csv(D/'chisq_dense_A.csv');reference=reference[reference['mode']=='joint'];aligned=[]
for br in labels:
 template=reference[reference.branch=='nofg'][['omega_m','w']].copy(); source=t[t.branch==br].copy()
 template['key']=[(round(x,7),round(y,7)) for x,y in template.to_numpy()];source['key']=[(round(x,7),round(y,7)) for x,y in source[['omega_m','w']].to_numpy()]
 merged=template.merge(source[['key','delta_chi_squared']],on='key',validate='one_to_one');merged['branch']=br;merged['mode']='joint';aligned.append(merged.drop(columns='key'))
t=pd.concat(aligned,ignore_index=True);t.to_csv(O/'joint_chisq.csv',index=False)
oldtable=pd.read_csv(D/'chisq_dense_A.csv');merge=t.merge(oldtable[oldtable['mode']=='joint'],on=['branch','mode','omega_m','w'],suffixes=('_new','_old'));err=float(abs(merge.delta_chi_squared_new-merge.delta_chi_squared_old).max());assert err<1e-8
src=P/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/exact_hpd.py';node=next(n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='meanexp');gn,gw=np.polynomial.legendre.leggauss(16);ns={'np':np,'u':(gn+1)/2,'wt':gw/2};exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns);meanexp=ns['meanexp']
# Reproduce original display interpolation, smoothing and normalization exactly.
XRANGE,YRANGE=(.195,.425),(-1.325,-.67);NG=800;gx=np.linspace(*XRANGE,NG);gy=np.linspace(*YRANGE,NG);GX,GY=np.meshgrid(gx,gy)
src=D/'plot_dense_A.py';node=next(n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='smooth_surface');exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),globals())
results=[];vertices=[];archived_levels=json.loads((D/'hpd_levels_dense_A.json').read_text())
for br,m in zip(labels,ms):
 s=t[t.branch==br];xy=s[['omega_m','w']].to_numpy();q=s.delta_chi_squared.to_numpy();tr=Delaunay(xy).simplices;v=xy[tr];ar=abs(np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]))/2;vals=np.sort(q[tr],axis=1);a,b,c=vals.T;full=ar*meanexp(vals);total=full.sum()
 def integral(level):
  f=np.zeros(len(ar));z=np.zeros(len(ar));yes=c<=level;f[yes]=1;z[yes]=full[yes];lo=(a<level)&(level<b);f[lo]=(level-a[lo])**2/((b[lo]-a[lo])*(c[lo]-a[lo]));z[lo]=ar[lo]*f[lo]*meanexp(np.column_stack([a[lo],np.full(lo.sum(),level),np.full(lo.sum(),level)]));hi=(b<=level)&(level<c)&(a<level);exc=(c[hi]-level)**2/((c[hi]-a[hi])*(c[hi]-b[hi]));f[hi]=1-exc;z[hi]=full[hi]-ar[hi]*exc*meanexp(np.column_stack([np.full(hi.sum(),level),np.full(hi.sum(),level),c[hi]]));return z.sum()/total,(ar*f).sum()
 Q,inside,hp=smooth_surface(xy,q)
 for mass in [.683,.954,.997]:
  level=brentq(lambda v:integral(v)[0]-mass,0,q.max(),xtol=1e-11);assert abs(integral(level)[0]-mass)<1e-9
  display_level=archived_levels[br][str(mass)] if br in archived_levels else level
  fig,ax=plt.subplots();cs=ax.contour(GX,GY,np.ma.masked_invalid(Q),levels=[display_level]);segs=cs.allsegs[0];closed=bool(len(segs) and all(len(s)>2 and np.allclose(s[0],s[-1],atol=1e-12,rtol=0) for s in segs));area=abs(sum(.5*(np.dot(s[:,0],np.roll(s[:,1],-1))-np.dot(s[:,1],np.roll(s[:,0],-1))) for s in segs)) if closed else np.nan
  for k,seg in enumerate(segs):
   for i,(xx,yy) in enumerate(seg):vertices.append(dict(branch=br,mass=mass,segment=k,vertex=i,omega_m=xx,w=yy))
  plt.close(fig);results.append(dict(branch=br,m=m,n_points=len(xy),label_mass=mass,Q_level=level,display_Q_level=display_level,display_closed=closed,display_area=area,raw_hpd_area=integral(level)[1]))
r=pd.DataFrame(results)
for mass in [.683,.954,.997]:
 mask=r.label_mass==mass;ref=r[mask&(r.branch=='horizontal_only')].iloc[0];r.loc[mask,'display_over_radial']=r.loc[mask,'display_area']/ref.display_area;r.loc[mask,'raw_over_radial']=r.loc[mask,'raw_hpd_area']/ref.raw_hpd_area
r.to_csv(O/'areas.csv',index=False);pd.DataFrame(vertices).to_csv(O/'contour_vertices.csv',index=False)
(O/'validation.json').write_text(json.dumps(dict(reference_overlap_exact=True,covariance_overlap_passed=True,joint_Q_replay_max_abs_error=err,old233_nofg_Q_replay_max_abs_error=old_replay_error,covariances=validation),indent=2));(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(r[r.label_mass==.683].to_string(index=False))
