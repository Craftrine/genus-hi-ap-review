"""Redraw the measured pilot using saved CSV; no field measurements."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
t=pd.read_csv(O/'point_Q.csv');cfg=json.loads((O/'plan.json').read_text());points=cfg['points'][1:]
assert len(t)==130
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
f,axs=plt.subplots(3,4,figsize=(14,9.5))
f.subplots_adjust(left=.055,right=.985,top=.87,bottom=.115,wspace=.26,hspace=.45)
summary=[];segments=[]
for ax,p in zip(axs.ravel(),points):
    s=t[(t.point_id==p['id'])&(t.branch>0)].sort_values('m');h=t[(t.point_id==p['id'])&(t.branch==0)].iloc[0]
    ax.plot(s.m,s.Q,'o-',color='#176b9a',ms=4,lw=1.3,label='Own covariance at each m')
    ax.plot(s.m,s.Q_common_C_m4,'s:',color='#806599',ms=3,lw=1,label='Common covariance (m = 1.343)')
    ax.axhline(h.Q,color='#c87b22',ls='--',lw=1,label='Horizontal only (own covariance)')
    ax.set_title(f'{p["id"]}: '+r'$\Omega_m$'+f' = {p["omega_m"]:.4f}, w = {p["w"]:.3f}',fontsize=10)
    ax.set_xticks([.993,1.168,1.343,1.517,1.692]);ax.tick_params(labelsize=8)
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.17)
    ax.set_ylim(bottom=0)
    if ax in axs[-1,:]:ax.set_xlabel('Wedge slope m')
    if ax in axs[:,0]:ax.set_ylabel('Joint Q')
    for col in ['Q','Q_common_C_m4']:
        vals=s[col].to_numpy();delta=np.diff(vals)
        summary.append(dict(point_id=p['id'],variant=col,omega_m=p['omega_m'],w=p['w'],
            Q_min=vals.min(),Q_max=vals.max(),m_at_min=s.m.iloc[np.argmin(vals)],m_at_max=s.m.iloc[np.argmax(vals)],
            has_increase=bool(np.any(delta>1e-8)),has_decrease=bool(np.any(delta<-1e-8)),
            sampled_nonmonotonic=bool(np.any(delta>1e-8) and np.any(delta<-1e-8)),
            note='Signs describe sampled numerical values, not significance tests'))
        for k in range(8):segments.append(dict(point_id=p['id'],variant=col,m_left=s.m.iloc[k],m_right=s.m.iloc[k+1],delta_Q=delta[k]))
f.suptitle('Nine measured wedge slopes at 12 fixed cosmologies',fontsize=16,y=.977)
f.text(.5,.928,'R20; full AP box; target spacing 4; hard foreground cuts. Six new slopes; three old slopes remeasured.',ha='center',fontsize=11)
handles,labels=axs[0,0].get_legend_handles_labels();f.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.036),ncol=3,frameon=False)
f.text(.5,.012,'Lines connect actual measurements; no smoothing. A pointwise Q scan does not determine contour area.',ha='center',fontsize=10)
name='fig__intrinsic_subtracted_contours__m_turnover_pilot'
for ext in ['png','pdf']:f.savefig(O/f'{name}.{ext}',dpi=190)
plt.close(f)
f,axs=plt.subplots(2,2,figsize=(10,7.6))
f.subplots_adjust(left=.085,right=.975,top=.83,bottom=.14,wspace=.23,hspace=.4)
for ax,p,title in zip(axs.ravel(),points[:4],['Higher matter density','Higher w','Lower matter density','Lower w']):
    s=t[(t.point_id==p['id'])&(t.branch>0)].sort_values('m')
    h=t[(t.point_id==p['id'])&(t.branch==0)].iloc[0]
    ax.plot(s.m,s.Q,'o-',color='#176b9a',ms=5,lw=1.6,label='Each m uses its own covariance')
    ax.plot(s.m,s.Q_common_C_m4,'s:',color='#806599',ms=4,lw=1.3,label='All m use the same covariance (diagnostic)')
    ax.set_title(title+'\n'+r'$\Omega_m$'+f' = {p["omega_m"]:.4f}, w = {p["w"]:.3f}',fontsize=11)
    ax.set(xlabel='Wedge slope m',ylabel='Joint Q',ylim=(0,None))
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.17)
    ax.set_xticks([.993,1.168,1.343,1.517,1.692])
f.suptitle('Nine actual m measurements: four directions away from the simulation cosmology',fontsize=13,y=.97)
f.text(.5,.9,'Each panel holds cosmology fixed. Larger Q means this point is more strongly disfavoured.',ha='center',fontsize=10)
handles,labels=axs[0,0].get_legend_handles_labels();f.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.044),ncol=2,frameon=False,fontsize=9)
f.text(.5,.012,'No parameter smoothing. This pilot measures Q responses, not ellipse areas.',ha='center',fontsize=10)
for ext in ['png','pdf']:f.savefig(O/f'fig__intrinsic_subtracted_contours__m_turnover_axes.{ext}',dpi=190)
plt.close(f)
pd.DataFrame(summary).to_csv(O/'point_summary.csv',index=False);pd.DataFrame(segments).to_csv(O/'adjacent_m_changes.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
