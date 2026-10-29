"""Conditional same-subset area ratios from stored curves and raw linear P."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;ROOT=O.parents[1];scheme=sys.argv[1];matched=len(sys.argv)>2 and sys.argv[2]=='matched';suffix='_matched' if matched else ''
SN=[118,112,105,94];BR=['horizontal_only','horizontal_m4'];indices=[1,5]
ref=np.load(ROOT/'outputs/wedge_all_m_20260920/reference_covariances.npz');I=ref['intrinsic'][:,indices];C=ref['covariances'][indices]
if scheme=='AB':
 fid=[np.load(O/'measurements'/f's{sn}_om0.311100000000_w-1.000000000000.npz') for sn in SN];g=np.array([f['AB'][[1,3]] for f in fid]);I=g[1:]-g[0]
 if matched:
  from sklearn.covariance import oas
  loo=np.array([(f['AB'][[1,3]][:,None,:]-f['contributions'][[1,3]])/(1-f['fractions'][None,:,None]) for f in fid]);C=[]
  for j in range(2):
   x=(loo[1:,j]-loo[0,j]).transpose(1,0,2).reshape(64,123);x-=x.mean(0);sd=np.sqrt(np.diag(63/64*x.T@x));corr,_=oas(x/np.sqrt(np.mean(x*x,axis=0)),assume_centered=True);C.append(corr*sd[:,None]*sd[None,:])
ch=[cho_factor(c) for c in C];nodes=pd.read_csv(O/'subset_nodes.csv');rows=[]
for om,w in nodes.to_numpy():
 files=[O/'subset_measurements'/f'{scheme}_s{sn}_om{om:.12f}_w{w:.12f}.npz' for sn in SN]
 if not all(f.exists() for f in files):continue
 g=np.array([np.load(f)['curves'] for f in files]);r=(g[1:]-g[0]-I).transpose(1,0,2).reshape(2,123)
 for b,v,c in zip(BR,r,ch):rows.append(dict(omega_m=om,w=w,branch=b,scheme=scheme,chi2_int=v@cho_solve(c,v)))
print('Subset nodes',len(rows)//2,'/',len(nodes))
pd.DataFrame(rows).to_csv(O/f'subset_new{suffix}.csv',index=False)
if len(rows)!=2*len(nodes):sys.exit(0)
old=pd.read_csv(O/'subset_original.csv').rename(columns={'delta_chi_squared':'chi2_int'});old['scheme']='original'
d=pd.concat([old,pd.DataFrame(rows)],ignore_index=True);d[['omega_m','w','branch','scheme','chi2_int']].to_csv(O/f'subset_surface{suffix}.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(10,4.7),sharex=True,sharey=True);result=[];vertices=[]
xy=nodes.to_numpy();tri=Delaunay(xy)
for ax,b,title in zip(axes,BR,['Horizontal cut',r'Wedge $m=1.343$']):
 for mode,col in [('original','#555555'),(scheme,'#177ca4' if scheme=='A' else '#c64a2d')]:
  s=d[(d.branch==b)&(d.scheme==mode)]
  values={(round(v.omega_m,12),round(v.w,12)):v.chi2_int for v in s.itertuples()}
  assert len(values)==len(xy)
  q=np.array([values[(round(a,12),round(bb,12))] for a,bb in xy]);pts=xy;p=np.exp(-.5*(q-q.min()))
  for ng in [400,800,1200]:
   x=np.linspace(xy[:,0].min(),xy[:,0].max(),ng);y=np.linspace(xy[:,1].min(),xy[:,1].max(),ng);X,Y=np.meshgrid(x,y)
   PP=griddata(pts,p,(X,Y),method='linear');valid=PP[np.isfinite(PP)];ordered=np.sort(valid)[::-1];cum=ordered.cumsum();levels=[]
   for target in [.683,.954]:
    level=ordered[np.searchsorted(cum,target*cum[-1])];area=np.sum(valid>=level)*(x[1]-x[0])*(y[1]-y[0]);levels.append(level)
    hull=Delaunay(pts).convex_hull;touch=bool(np.any(p[np.unique(hull)]>=level));result.append(dict(branch=b,scheme=mode,target=target,raster=ng,conditional_area=area,touches_subset_boundary=touch))
   if ng==800:
    for target,level,ls in zip([.683,.954],levels,['-','--']):
     cs=ax.contour(X,Y,PP,levels=[level],colors=[col],linestyles=ls,linewidths=1.2)
     for seg,arr in enumerate(cs.allsegs[0]):
      for i,(xx,yy) in enumerate(arr):vertices.append(dict(branch=b,scheme=mode,target=target,segment=seg,index=i,omega_m=xx,w=yy))
    ax.plot([],[],color=col,label=mode)
 ax.set_title(title);ax.plot(.3111,-1,'k*',ms=8);ax.set_xlabel(r'$\Omega_m$');ax.grid(alpha=.15);ax.legend()
axes[0].set_ylabel('$w$');fig.suptitle(f"Same 271 nodes: original vs {scheme}; {'matched' if matched else 'common'} covariance")
fig.text(.5,.01,'Solid: conditional 68.3%; dashed: conditional 95.4%. Unsmoothed linear P.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.035,1,.96])
for ext in ['png','pdf']:fig.savefig(O/f'fig__mesh_AB_subset_regions{suffix}.{ext}',dpi=180)
r=pd.DataFrame(result);r.to_csv(O/f'subset_areas{suffix}.csv',index=False);pd.DataFrame(vertices).to_csv(O/f'subset_contour_vertices{suffix}.csv',index=False)
ratios=[]
for ng in [400,800,1200]:
 for target in [.683,.954]:
  s=r[(r.raster==ng)&(r.target==target)].pivot(index='branch',columns='scheme',values='conditional_area')
  for b in BR:ratios.append(dict(raster=ng,target=target,quantity=b,old_ratio=1.,new_ratio=s.loc[b,scheme]/s.loc[b,'original'],relative_change=s.loc[b,scheme]/s.loc[b,'original']-1))
  oldratio=s.loc[BR[1],'original']/s.loc[BR[0],'original'];newratio=s.loc[BR[1],scheme]/s.loc[BR[0],scheme]
  ratios.append(dict(raster=ng,target=target,quantity='wedge_over_horizontal',old_ratio=oldratio,new_ratio=newratio,relative_change=newratio/oldratio-1))
pd.DataFrame(ratios).to_csv(O/f'subset_area_ratios{suffix}.csv',index=False);print(pd.DataFrame(ratios).query('raster==800').to_string(index=False))
