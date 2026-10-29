from pathlib import Path
import csv,json
import numpy as np
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh';rows=list(csv.DictReader((D/'before_after_chisq.csv').open()));h=json.loads((D/'integrated_hpd_summary.json').read_text());BR=['nofg','horizontal_only','horizontal_m1','horizontal_m5'];titles=['noFG','Horizontal cut','Horizontal + m=0.294','Horizontal + m=1.692'];fig,axes=plt.subplots(1,4,figsize=(15,4.5))
for ax,b,t in zip(axes,BR,titles):
 r=[v for v in rows if v['variant']=='intrinsic_subtracted' and v['branch']==b and v['mode']=='joint'];xy=np.array([[float(v['omega_m']),float(v['w'])] for v in r]);q=np.array([float(v['delta_chi_squared']) for v in r]);levels=next(v['analytic']['deltaQ_levels'] for v in h['results'] if v['variant']=='intrinsic_subtracted' and v['branch']==b and v['mode']=='joint');mesh=tri.Triangulation(*xy.T,Delaunay(xy).simplices);ax.set_facecolor('#e9edf0');ax.tricontourf(mesh,q,levels=[-1e-9,*levels,max(q.max()+1,levels[-1]+1)],colors=['#8cbad0','#cadfe9','#edf3f7','white']);cs=ax.tricontour(mesh,q,levels=levels,colors='#24668b',linestyles=['-','--',':']);ax.plot(.3111,-1,'*',color='#e4ad37',mec='#26333c',ms=9)
 if b=='nofg':limits=[.195,.425,-1.325,-.67]
 else:
  v=np.vstack([s for s in cs.allsegs[-1] if len(s)]);low=v.min(0);high=v.max(0);pad=(high-low)*.15;limits=[low[0]-pad[0],high[0]+pad[0],low[1]-pad[1],high[1]+pad[1]]
 ax.set(xlim=limits[:2],ylim=limits[2:],title=t,xlabel=r'$\Omega_m$',ylabel='$w$');ax.tick_params(labelsize=9);ax.grid(alpha=.12)
fig.suptitle('Joint intrinsic-subtracted regions | panel-specific ranges show both ends',fontsize=13)
fig.legend(handles=[Line2D([],[],color='#24668b',ls=ls,label=f'{p}% HPD') for ls,p in [('-',68.3),('--',95.4),(':',99.7)]],loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
fig.text(.5,.015,'noFG 95.4% and 99.7% still meet sampled support; all three levels close for the filtered cases. Conditional probabilities, not independently calibrated coverage.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.15,1,.9));fig.savefig(D/'joint_zoom.png',dpi=200);fig.savefig(D/'joint_zoom.pdf');plt.close(fig)
