"""Plot only saved CSV values; no scientific measurements."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay,ConvexHull
O=Path(__file__).resolve().parent
BR=['nofg','horizontal_only','horizontal_m1','horizontal_m4'];TITLE=['noFG','Horizontal cut',r'Wedge $m=0.294$',r'Wedge $m=1.343$'];COLOR=['#555555','#177ca4','#c64a2d']
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
if (O/'line.csv').exists():
 d=pd.read_csv(O/'line.csv');fig,axes=plt.subplots(2,2,figsize=(10,6.5),sharex=True)
 for ax,b,title in zip(axes.flat,BR,TITLE):
  for s,c in zip(['original','A','AB'],COLOR):
   q=d[(d.branch==b)&(d.scheme==s)].sort_values('omega_m');ax.plot(q.omega_m,q.chi2_int,'.-',ms=3,color=c,label={'original':'Original','A':'Fixed N','AB':'Fixed N + taper'}[s])
  ax.set_title(title);ax.set_ylabel(r'$\chi^2_{\rm int}$');ax.grid(alpha=.18)
 for ax in axes[-1]:ax.set_xlabel(r'$\Omega_m$')
 axes[0,0].legend(fontsize=9);fig.suptitle(f"Common covariance; w = {d.w.iloc[0]:.9f}");fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(O/f'fig__mesh_AB_omega_line.{ext}',dpi=180)
 plt.close(fig)
if (O/'boundary_results.csv').exists():
 d=pd.read_csv(O/'boundary_results.csv');d=d[d.covariance=='common'];fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,(probe,g) in zip(axes,d.groupby('probe',sort=False)):
  for j,(s,c) in enumerate(zip(['original','A','AB'],COLOR)):
   vals=[g[(g.scheme==s)&(g.branch==b)].delta_Q.iloc[0] for b in BR]
   ax.bar(np.arange(4)+(j-1)*.24,vals,width=.24,color=c,label={'original':'Original','A':'A','AB':'A+B'}[s])
  ax.axhline(2.3,color='black',ls=':',lw=1,label=r'$2.3$ reference');ax.set_xticks(range(4),['noFG','Horizontal','m=0.294','m=1.343'],rotation=15);ax.set_title({'nofg_3':'Mesh boundary: snapshot 112','horizontal_only_1':'Mesh boundary: snapshot 94'}[probe]);ax.grid(axis='y',alpha=.18)
 axes[0].set_ylabel(r'$\chi^2_{\rm right}-\chi^2_{\rm left}$');axes[0].legend();fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(O/f'fig__mesh_AB_boundary_jumps.{ext}',dpi=180)
 plt.close(fig)
# Subset contours can be redrawn from their saved vertices alone.
for suffix in ['', '_matched']:
 fp=O/f'subset_contour_vertices{suffix}.csv'
 if not fp.exists():continue
 d=pd.read_csv(fp);fig,axes=plt.subplots(1,2,figsize=(10,4.7),sharex=True,sharey=True)
 nodes=pd.read_csv(O/'subset_nodes.csv')[['omega_m','w']].to_numpy();hull=nodes[ConvexHull(nodes).vertices];hull=np.vstack([hull,hull[0]])
 for ax,b,title in zip(axes,['horizontal_only','horizontal_m4'],['Horizontal cut',r'Wedge $m=1.343$']):
  s=d[d.branch==b]
  ax.plot(hull[:,0],hull[:,1],':',color='#aaaaaa',lw=.8);ax.scatter(nodes[:,0],nodes[:,1],s=3,color='#aaaaaa',alpha=.25,zorder=0)
  for mode,g in s.groupby('scheme',sort=False):
   c='#555555' if mode=='original' else ('#177ca4' if mode=='A' else '#c64a2d')
   for (target,segment),v in g.groupby(['target','segment']):
    v=v.sort_values('index');ax.plot(v.omega_m,v.w,color=c,ls='-' if target==.683 else '--',lw=1.2)
   ax.plot([],[],color=c,label={'original':'Original','A':'A','AB':'A+B'}[mode])
  ax.plot(.3111,-1,'k*',ms=8);ax.set_title(title);ax.set_xlabel(r'$\Omega_m$');ax.grid(alpha=.15);ax.legend()
 axes[0].set_ylabel('$w$');fig.suptitle('Same 271 nodes: original vs A+B; '+('matched' if suffix else 'common')+' covariance')
 fig.text(.5,.01,'Solid: conditional 68.3%; dashed: conditional 95.4%. Unsmoothed linear P.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.035,1,.96])
 for ext in ['png','pdf']:fig.savefig(O/f'fig__mesh_AB_subset_regions{suffix}.{ext}',dpi=180)
 plt.close(fig)
