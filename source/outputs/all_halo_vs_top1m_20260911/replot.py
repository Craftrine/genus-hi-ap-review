"""Publication-style matched-sample comparison; saved CSV/JSON inputs only."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from scipy.spatial import Delaunay,ConvexHull
OUT=Path(__file__).resolve().parent
BR=['nofg','horizontal_only','horizontal_m1','horizontal_m5']
TITLES=['No foreground cut','Horizontal cut',r'Horizontal + $m=0.294$',r'Horizontal + $m=1.692$']
COL={'top1m':'#277da8','all_halo':'#d47b24'};LAB={'top1m':'Top 1 million, HI weighted','all_halo':'All valid halos, HI weighted'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def read(name):
    r=list(csv.DictReader((OUT/name).open()))
    for row in r:
        for k,v in row.items():
            try:row[k]=float(v)
            except ValueError:pass
    return r
def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=185)
    plt.close(fig)
fid=read('fiducial_comparison.csv');am=read('amplitudes.csv');q=read('chisq.csv');resp=read('ap_response.csv');csens=read('covariance_sensitivity.csv')
sel=read('sample_selection.csv')
fig,axs=plt.subplots(2,4,figsize=(14,7),sharex=True)
for bi,b in enumerate(BR):
    for sample in ['top1m','all_halo']:
        r=[v for v in fid if v['sample']==sample and v['snapshot']==94 and v['branch']==b]
        x=np.array([v['nu_f'] for v in r]);y=np.array([v['g'] for v in r])*8000;err=np.array([v['sigma'] for v in r])*8000
        axs[0,bi].plot(x,y,color=COL[sample],lw=1.7,label=LAB[sample]);axs[0,bi].fill_between(x,y-err,y+err,color=COL[sample],alpha=.14,lw=0)
    r=[v for v in fid if v['sample']=='all_halo' and v['snapshot']==94 and v['branch']==b]
    y=np.array([v['all_minus_top'] for v in r])*8000;err=np.array([v['all_minus_top_paired_sigma'] for v in r])*8000
    axs[1,bi].plot(x,y,color='#74558e',lw=1.5);axs[1,bi].fill_between(x,y-err,y+err,color='#74558e',alpha=.18,lw=0)
    axs[0,bi].set_title(TITLES[bi],fontsize=11)
    for ax in axs[:,bi]:ax.axhline(0,color='.45',lw=.6);ax.grid(alpha=.15);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.tick_params(labelsize=8)
    axs[1,bi].set_xlabel(r'$\nu_f$')
axs[0,0].set_ylabel(r'$g(\nu_f)R^3$');axs[1,0].set_ylabel(r'$(g_{\rm all}-g_{\rm top})R^3$')
fig.suptitle(r'All-halo vs top-1M HI weighting: $z=0.59339$, $R=20\,h^{-1}\mathrm{Mpc}$'+'\nSame full-box mesh, RSD, filtering and genus estimator',fontsize=14)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False)
fig.text(.5,.018,'Bands: matched 64-block additive jackknife standard errors; difference bands include the correlation between samples.',ha='center',fontsize=9,color='.3')
fig.tight_layout(rect=(0,.12,1,.9),w_pad=1.1,h_pad=.6);save(fig,'fig__all_halo_vs_top1m__genus')

fig,axs=plt.subplots(4,4,figsize=(14,11),sharex=True)
for zi,snap in enumerate([118,112,105,94]):
    z=next(v['z'] for v in sel if v['snapshot']==snap)
    for bi,b in enumerate(BR):
        ax=axs[zi,bi]
        for sample in ['top1m','all_halo']:
            r=[v for v in fid if v['sample']==sample and v['snapshot']==snap and v['branch']==b]
            ax.plot([v['nu_f'] for v in r],np.array([v['g'] for v in r])*8000,c=COL[sample],lw=1.4)
        ax.axhline(0,c='.5',lw=.5);ax.grid(alpha=.12);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.tick_params(labelsize=8)
        if zi==0:ax.set_title(TITLES[bi],fontsize=11)
        if bi==0:ax.set_ylabel(f'z={z:.5f}\n'+r'$gR^3$')
        if zi==3:ax.set_xlabel(r'$\nu_f$')
fig.suptitle('Matched full curves across all four redshifts',fontsize=15)
fig.legend(handles=[Line2D([],[],c=COL[s],label=LAB[s]) for s in COL],loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False)
fig.tight_layout(rect=(0,.045,1,.96),w_pad=1,h_pad=.6);save(fig,'fig__all_halo_vs_top1m__all_redshifts')

fig,axs=plt.subplots(1,4,figsize=(14,4.2),sharex=True)
for bi,b in enumerate(BR):
    ax=axs[bi]
    for sample in ['top1m','all_halo']:
        r=sorted([v for v in am if v['sample']==sample and v['branch']==b],key=lambda v:v['z'])
        ax.errorbar([v['z'] for v in r],[v['AR3'] for v in r],yerr=[v['AR3_sigma'] for v in r],color=COL[sample],marker='o',ms=4,capsize=3,lw=1.3)
    ax.set_title(TITLES[bi],fontsize=11);ax.set_xlabel('Redshift z');ax.grid(alpha=.15);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.tick_params(labelsize=8)
axs[0].set_ylabel(r'Gaussian-template amplitude $AR^3$')
fig.suptitle('How sample selection changes genus amplitude and redshift evolution',fontsize=14)
fig.legend(handles=[Line2D([],[],c=COL[s],marker='o',label=LAB[s]) for s in COL],loc='lower center',bbox_to_anchor=(.5,.012),ncol=2,frameon=False)
fig.tight_layout(rect=(0,.1,1,.9),w_pad=1);save(fig,'fig__all_halo_vs_top1m__amplitude_evolution')

# Same covariance diagnostic isolates changed residuals from changed covariance.
fig,axs=plt.subplots(2,4,figsize=(14,6.8),sharex='row')
for ai,axis in enumerate(['omega_m','w']):
    for bi,b in enumerate(BR):
        ax=axs[ai,bi]
        for sample in ['top1m','all_halo']:
            for case,ls in [('own_covariance','-'),('top1m_covariance','--')]:
                if sample=='top1m' and case=='top1m_covariance':continue
                r=[v for v in csens if v['sample']==sample and v['branch']==b and v['covariance_case']==case and (v['w']==-1 if ai==0 else v['omega_m']==.3111)]
                r.sort(key=lambda v:v[axis]);ax.plot([v[axis]-(.3111 if ai==0 else -1) for v in r],[v['chi_squared'] for v in r],color=COL[sample],ls=ls,marker='o',ms=3,lw=1.3)
        ax.set_yscale('symlog',linthresh=.05);ax.grid(alpha=.13);ax.tick_params(labelsize=8)
        if ai==0:ax.set_title(TITLES[bi],fontsize=11)
        if bi==0:ax.set_ylabel('Joint Q')
        ax.set_xlabel(r'$\Omega_m-0.3111$ at $w=-1$' if ai==0 else r'$w+1$ at $\Omega_m=0.3111$',fontsize=9)
fig.suptitle('AP response: each sample uses its own intrinsic evolution\nSolid: own covariance; dashed orange: all-halo residuals with top-1M covariance (diagnostic only)',fontsize=13)
fig.legend(handles=[Line2D([],[],c=COL[s],label=LAB[s]) for s in COL],loc='lower center',bbox_to_anchor=(.5,.014),ncol=2,frameon=False)
fig.tight_layout(rect=(0,.08,1,.9),w_pad=1,h_pad=.7);save(fig,'fig__all_halo_vs_top1m__ap_response')

hpd={s:json.loads((OUT/f'hpd_{s}/integrated_hpd_summary.json').read_text()) for s in COL}
boundary=[]
for variant in ['intrinsic_subtracted','intrinsic_not_subtracted']:
    fig,axs=plt.subplots(1,4,figsize=(14,4.6),sharex=True,sharey=True)
    for bi,b in enumerate(BR):
        ax=axs[bi]
        for sample in COL:
            r=[v for v in q if v['sample']==sample and v['branch']==b and v['mode']=='joint' and v['variant']==variant]
            xy=np.array([[v['omega_m'],v['w']] for v in r]);val=np.array([v['delta_chi_squared'] for v in r])
            tri=Delaunay(xy);triang=mtri.Triangulation(xy[:,0],xy[:,1],tri.simplices)
            hr=next(v for v in hpd[sample]['results'] if v['variant']==variant and v['branch']==b and v['mode']=='joint');levels=hr['analytic']['deltaQ_levels']
            ax.tricontour(triang,val,levels=levels[:2],colors=COL[sample],linewidths=[2,1],linestyles=['-','--'])
            # Include all collinear boundary nodes, not just rectangle corners.
            boundary_nodes=np.unique(tri.convex_hull)
            bmin=float(val[boundary_nodes].min())
            rect=np.any(np.isclose(xy,xy.min(0),rtol=0,atol=1e-12)|np.isclose(xy,xy.max(0),rtol=0,atol=1e-12),axis=1)
            np.testing.assert_allclose(bmin,val[rect].min(),rtol=0,atol=1e-12)
            for prob,lev in zip([.683,.954,.997],levels):boundary.append(dict(sample=sample,variant=variant,branch=b,probability=prob,deltaQ_level=lev,touches_boundary=bmin<=lev,nodes_inside=int((val<=lev).sum()),boundary_min_deltaQ=bmin))
        ax.scatter(xy[:,0],xy[:,1],s=5,c='.55',alpha=.5);ax.plot(.3111,-1,'k*',ms=9)
        ax.set(xlim=(.3091,.3131),ylim=(-1.01,-.99),xticks=[.31,.3111,.312],yticks=[-1.01,-1,-.99],xlabel=r'$\Omega_m$',title=TITLES[bi]);ax.tick_params(labelsize=8);ax.grid(alpha=.13)
    axs[0].set_ylabel(r'$w$')
    fig.suptitle('Joint conditional regions on an identical local domain\n'+('Own intrinsic evolution subtracted; own reference covariance' if variant=='intrinsic_subtracted' else 'Intrinsic evolution retained; own reference covariance'),fontsize=13)
    fig.legend(handles=[Line2D([],[],c=COL[s],label=LAB[s]) for s in COL]+[Line2D([],[],c='.3',lw=2,label='68.3% local HPD'),Line2D([],[],c='.3',ls='--',label='95.4% local HPD')],loc='lower center',bbox_to_anchor=(.5,.06),ncol=4,frameon=False,fontsize=9)
    fig.text(.5,.018,'Probabilities normalized only over this shared local rectangle. Boundary-reaching regions do not yield global parameter errors.',ha='center',fontsize=9,color='.3')
    fig.tight_layout(rect=(0,.15,1,.86),w_pad=.7)
    save(fig,'fig__all_halo_vs_top1m__contours'+('' if variant=='intrinsic_subtracted' else '_raw'))
(OUT/'boundary_audit.json').write_text(json.dumps(boundary,indent=2))
print('Saved all comparison figures; local-domain boundary audit exported.')
