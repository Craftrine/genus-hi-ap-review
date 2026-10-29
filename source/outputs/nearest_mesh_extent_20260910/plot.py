"""Compact publication panels from measured dense-grid Q and exact HPD cuts."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh';rows=list(csv.DictReader((D/'before_after_chisq.csv').open()));hpd=json.loads((D/'integrated_hpd_summary.json').read_text());summary=json.loads((D/'before_after_summary.json').read_text())
BR=['nofg','horizontal_only','horizontal_m1','horizontal_m5'];MOD=['z2-z1','z3-z1','z4-z1','joint'];T=['noFG','Horizontal cut',r'Horizontal + $m_{\rm scan}=0.294$',r'Horizontal + $m_{\rm scan}=1.692$'];R=[r'$z_2-z_1$',r'$z_3-z_1$',r'$z_4-z_1$','Joint'];color='#24668b'
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.linewidth':.7})
for var,label in [('intrinsic_subtracted','Intrinsic evolution subtracted'),('intrinsic_not_subtracted','Intrinsic evolution not subtracted')]:
 fig,axes=plt.subplots(4,4,figsize=(12,10.7),sharex=True,sharey=True)
 for mi,mode in enumerate(MOD):
  for bi,b in enumerate(BR):
   rr=[v for v in rows if v['variant']==var and v['branch']==b and v['mode']==mode];xy=np.array([[float(v['omega_m']),float(v['w'])] for v in rr]);q=np.array([float(v['delta_chi_squared']) for v in rr]);levels=next(v for v in hpd['results'] if v['variant']==var and v['branch']==b and v['mode']==mode)['analytic']['deltaQ_levels'];mesh=tri.Triangulation(xy[:,0],xy[:,1],Delaunay(xy).simplices);ax=axes[mi,bi];ax.set_facecolor('#e9edf0')
   ax.tricontourf(mesh,q,levels=[-1e-9,*levels,max(q.max()+1,levels[-1]+1)],colors=['#8cbad0','#cadfe9','#edf3f7','white'])
   ax.tricontour(mesh,q,levels=levels,colors=color,linewidths=[1.1,.95,.85],linestyles=['-','--',':'])
   hull=ConvexHull(xy);vertices=xy[np.r_[hull.vertices,hull.vertices[0]]];ax.plot(vertices[:,0],vertices[:,1],color='#a8b0b8',lw=.55,zorder=2)
   ax.plot(.3111,-1,marker='*',ms=7.5,color='#e6b23f',mec='#26333c',mew=.55,zorder=6)
   if var=='intrinsic_not_subtracted':
    point=next(v for v in summary['results'] if v['variant']==var and v['branch']==b and v['mode']==mode)['best_point'];ax.plot(*point,'x',color='#3d434b',ms=4,mew=.8)
   ax.set(xlim=(.195,.425),ylim=(-1.325,-.67));ax.set_xticks([.2,.3,.4]);ax.set_yticks([-1.25,-1.,-.75]);ax.tick_params(labelsize=9,length=3,pad=3)
   ax.text(.025,.95,f'({chr(97+mi*4+bi)})',transform=ax.transAxes,va='top',fontsize=9,color='#405565')
   if mi==0:ax.set_title(T[bi],fontsize=11,pad=8)
   if bi==0:ax.set_ylabel(R[mi]+'\n'+r'$w$',fontsize=11)
   if mi==3:ax.set_xlabel(r'$\Omega_m$',fontsize=11)
 fig.subplots_adjust(left=.085,right=.99,top=.88,bottom=.12,wspace=.045,hspace=.045)
 fig.suptitle(label,fontsize=17,y=.990)
 fig.text(.5,.944,r'Full-box nearest-spacing mesh | $R=20\,h^{-1}{\rm Mpc}$ | 233 measured cosmologies; expanded along the degeneracy',ha='center',fontsize=11,color='#526371')
 fig.text(.5,.918,r'$z_1=0.10235,\ z_2=0.19095,\ z_3=0.31970,\ z_4=0.59339$',ha='center',fontsize=10,color='#526371')
 handles=[Line2D([],[],color=color,ls=ls,label=f'{p}% HPD') for ls,p in [('-',68.3),('--',95.4),(':',99.7)]]+[Line2D([],[],marker='*',ls='',color='#e6b23f',mec='#26333c',label='Simulation cosmology')]
 if var=='intrinsic_not_subtracted':handles.append(Line2D([],[],marker='x',ls='',color='#3d434b',label='Best sampled point'))
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.043),ncol=len(handles),frameon=False,fontsize=10)
 fig.text(.5,.019,'Probability density ∝ exp(−Q/2), integrated over the measured convex hull with a uniform prior.\nGray: unmeasured support. Fixed intrinsic template and covariance; no independent coverage calibration.',ha='center',fontsize=9,color='#526371')
 fig.savefig(D/f'overview_{var}.png',dpi=240);fig.savefig(D/f'overview_{var}.pdf');plt.close(fig)
