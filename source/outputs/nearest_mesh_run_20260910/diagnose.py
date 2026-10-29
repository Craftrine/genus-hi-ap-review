"""Compare old cropped versus new full-box axial response, saved data only."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/'paper_figures/fig__intrinsic_subtracted_contours/nearest_mesh';new=json.loads((DEST/'continuity.json').read_text());old=list(csv.DictReader((ROOT/'outputs/local_refinement_step2a_20260910/axial_diagnostics.csv').open()))
BR=['nofg','horizontal_only','horizontal_m1','horizontal_m5'];titles=['noFG','Horizontal cut','Horizontal + m=0.294','Horizontal + m=1.692'];fig,axes=plt.subplots(2,4,figsize=(16,7.8),sharey=True);summary=[]
for ri,axis in enumerate(['omega_m','w']):
 for bi,b in enumerate(BR):
  ax=axes[ri,bi]
  for sign,color in [(-1,'#1b789e'),(1,'#d47729')]:
   for source,style,marker in [(old,'--',None),(new,'-','o')]:
    rr=[v for v in source if v['axis']==axis and v['branch']==b and float(v['offset'])*sign>0];rr.sort(key=lambda v:float(v['absolute_step']));x=np.array([float(v['absolute_step']) for v in rr]);q=np.array([float(v['joint_Q']) for v in rr]);ax.plot(x,q,color=color,ls=style,marker=marker,ms=3,lw=1.4)
    if source is new:summary.append(dict(axis=axis,branch=b,sign=sign,smallest_step=float(x[0]),new_Q_smallest=float(q[0])))
  ax.set_xscale('log');ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(0,80);ax.grid(alpha=.15);ax.tick_params(labelsize=8)
  if ri==0:ax.set_title(titles[bi],fontsize=11)
  if bi==0:ax.set_ylabel(r'Joint $Q$ (symmetric-log scale)',fontsize=10)
  ax.set_xlabel(r'$|\Omega_m-0.3111|$ at $w=-1$' if axis=='omega_m' else r'$|w+1|$ at $\Omega_m=0.3111$',fontsize=10)
fig.suptitle('Axial continuity: fixed-spacing cropped mesh vs nearest-spacing full box\nSame intrinsic template and covariance; exact fiducial Q=0',fontsize=13)
fig.legend(handles=[Line2D([],[],color='#1b789e',label='Negative offset'),Line2D([],[],color='#d47729',label='Positive offset'),Line2D([],[],color='.4',ls='--',label='Old: cropped box'),Line2D([],[],color='.4',marker='o',label='New: full box')],loc='lower center',bbox_to_anchor=(.5,.018),ncol=4,frameon=False,fontsize=10)
fig.tight_layout(rect=(0,.075,1,.91),w_pad=.6,h_pad=.8);fig.savefig(DEST/'continuity_old_vs_new.png',dpi=200);fig.savefig(DEST/'continuity_old_vs_new.pdf');plt.close(fig)
for v in summary:
 rr=[r for r in old if r['axis']==v['axis'] and r['branch']==v['branch'] and float(r['offset'])*v['sign']>0];r=min(rr,key=lambda r:float(r['absolute_step']));v['old_Q_smallest']=float(r['joint_Q'])
(DEST/'continuity_comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
