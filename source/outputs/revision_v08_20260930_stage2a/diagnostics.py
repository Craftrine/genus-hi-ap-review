"""Simple review plots, drawn only from the exported Stage-2a CSV tables."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
O=Path(__file__).resolve().parent;T=O/'tables';D=O/'diagnostics'
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans'})
labels={'nofg':'noFG','horizontal_only':'Horizontal cut','horizontal_m1':'Horizontal + m=0.294','horizontal_m5':'Horizontal + m=1.692'}
colors={.683:'#176b9a',.954:'#d17920',.997:'#8746aa'}
styles={.683:'-',.954:'--',.997:':'}
def finish(fig,name):
    fig.savefig(D/(name+'.png'),dpi=155);plt.close(fig)

v=pd.read_csv(T/'diagnostic_contour_vertices.csv');a=pd.read_csv(T/'four_cases_regions.csv')
points=pd.read_csv(T/'eleven_wedge_nodal_Q.csv');xy=points[points.branch=='nofg'][['omega_m','w']].to_numpy()
hull=xy[ConvexHull(xy).vertices];hull=np.vstack([hull,hull[0]])
fig,axs=plt.subplots(4,4,figsize=(15,10.5),sharex=True,sharey=True)
for column,br in enumerate(labels):
    for row,mode in enumerate(['z2-z1','z3-z1','z4-z1','joint']):
        ax=axs[row,column];ax.plot(*hull.T,color='.65',lw=.8)
        s=v[(v.branch==br)&(v['mode']==mode)&(v.method=='linear_P')]
        for (target,component),part in s.groupby(['target','component'],sort=False):
            part=part.sort_values('vertex');ax.plot(part.omega_m,part.w,color=colors[target],ls=styles[target],lw=1)
        ax.plot(.3111,-1,'k+',ms=5)
        if row==0:ax.set_title(labels[br])
        if column==0:ax.set_ylabel(mode+'\n$w$')
        if row==3:ax.set_xlabel(r'$\Omega_m$')
        ax.set_xlim(.19,.43);ax.set_ylim(-1.34,-.66)
fig.legend([Line2D([0],[0],color=colors[t],ls=styles[t]) for t in colors],['68.3%','95.4%','99.7%'],loc='lower center',ncol=3,frameon=False)
fig.tight_layout(rect=(0,.04,1,1));finish(fig,'fig__stage2a_four_case_boundary_diagnostic')

w=pd.read_csv(T/'eleven_wedge_areas.csv');r=pd.read_csv(T/'eleven_wedge_raster_areas.csv');st=pd.read_csv(T/'raster_800_1600_stability.csv')
fig,axs=plt.subplots(2,2,figsize=(11,8))
for method,style in [('linear_P','o-'),('linear_Q','s--')]:
    s=w[(w.target==.683)&(w.method==method)&w.branch.str.startswith('m')].sort_values('m')
    axs[0,0].plot(s.m,s.area_over_horizontal,style,label=method.replace('_','-'))
axs[0,0].axhline(1,color='.5',ls=':');axs[0,0].set_ylabel('68.3% area / horizontal-cut area');axs[0,0].legend()
for variant,label,style in [('no_gaussian','Unsmoothed','o-'),('gaussian','Smoothed','s--')]:
    s=r[(r.grid==800)&(r.target==.683)&(r.variant==variant)&r.branch.str.startswith('m')].sort_values('m')
    axs[0,1].plot(s.m,1e4*s.area,style,label=label)
axs[0,1].set_ylabel(r'$10^4 A_{0.683}$ (800 grid)');axs[0,1].legend()
for target in colors:
    for variant,marker in [('no_gaussian','o'),('gaussian','s')]:
        s=st[(st.target==target)&(st.variant==variant)&st.branch.str.startswith('m')].sort_values('m_800')
        axs[1,0].plot(s.m_800,100*s.relative_change,color=colors[target],ls=styles[target],marker=marker,ms=4,label=f'{100*target:g}%, '+('raw' if variant=='no_gaussian' else 'smooth'))
axs[1,0].set_ylabel('Area change: 800 to 1600 (%)');axs[1,0].legend(fontsize=8,ncol=2)
s=r[(r.grid==800)&(r.target==.683)&(r.variant=='gaussian')&r.branch.str.startswith('m')].sort_values('m')
axs[1,1].plot(s.m,s.smoothed_over_unsmoothed,'o-',color='#8746aa');axs[1,1].set_ylabel('Smoothed / unsmoothed 68.3% area')
for ax in axs.flat:ax.set_xlabel('Wedge slope m');ax.grid(alpha=.15)
fig.tight_layout();finish(fig,'fig__stage2a_eleven_slope_and_integration_diagnostic')

c=pd.read_csv(T/'weighted_cosine.csv');bins=pd.read_csv(T/'weighted_cosine_distance_bins.csv')
fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
for ax,(br,label) in zip(axs.flat,labels.items()):
    s=c[(c.branch==br)&(c['mode']=='joint')]
    ax.scatter(s.distance_nominal_steps,s.weighted_cosine,s=5,alpha=.25,color='#176b9a',rasterized=True)
    b=bins[(bins.branch==br)&(bins['mode']=='joint')]
    xx=(b.distance_lo+b.distance_hi)/2
    ax.plot(xx,b.cos_median,'o-',color='#a44725',ms=4);ax.fill_between(xx,b.cos_q05,b.cos_q95,color='#a44725',alpha=.12)
    ax.set_title(label);ax.set_ylim(.25,1.02);ax.set_xlim(0,55)
for ax in axs[1]:ax.set_xlabel('Distance from sim in nominal-step units')
for ax in axs[:,0]:ax.set_ylabel('Weighted cosine of d and I')
fig.tight_layout();finish(fig,'fig__stage2a_weighted_alignment_diagnostic')

s=pd.read_csv(T/'covariance_swap_areas.csv');s=s[(s.target==.683)&(s.method=='linear_P')]
fig,ax=plt.subplots(figsize=(8,4.8));positions=np.arange(2)
for k,cov in enumerate(['nofg','horizontal_only']):
    part=s[s.covariance_case==cov].set_index('residual_case').loc[['nofg','horizontal_only']]
    bars=ax.bar(positions+(k-.5)*.32,part.area,.32,label='Covariance: '+labels[cov],hatch='' if k==0 else '//')
    for bar,row in zip(bars,part.itertuples()):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.06,f'{row.area:.4g}'+('\nTouches hull' if row.region_touches_hull else ''),ha='center',va='bottom',fontsize=9)
ax.set_yscale('log');ax.set_ylim(5e-5,.04);ax.set_xticks(positions,['Residual: noFG','Residual: Horizontal cut']);ax.set_ylabel('68.3% area (linear-P)');ax.legend(loc='upper right');fig.tight_layout();finish(fig,'fig__stage2a_covariance_swap_diagnostic')
print('Four diagnostic PNGs rendered from saved CSV data.')
