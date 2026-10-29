"""Redraw saved diagnostic CSVs without recomputing areas or measurements."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
TARGETS=[.683,.954,.997]
def read(n):return pd.read_csv(O/'tables'/n,float_precision='round_trip')
main=read('eleven_wedge_areas_domain_flags.csv')

allareas=read('triangulation_all_areas.csv');ranges=read('triangulation_ranges.csv')
fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for row,method in enumerate(['linear_P','linear_Q']):
    for col,target in enumerate(TARGETS):
        ax=axs[row,col];s=ranges[(ranges.method==method)&(ranges.target==target)&ranges.branch.str.startswith('m')].sort_values('m')
        a=main[(main.method==method)&(main.target==target)&main.branch.str.startswith('m')].sort_values('m')
        ax.fill_between(s.m,s.area_over_horizontal_min,s.area_over_horizontal_max,color='steelblue',alpha=.25,label='20 perturbed triangulations')
        ax.plot(a.m,a.area_over_horizontal,'o-',color='black',ms=3,label='Fixed diagonal rule')
        ax.axhline(1,ls=':',color='gray');ax.set_title(f'{method.replace("_","-")} | {target*100:.1f}%');ax.set_xlabel('m');ax.set_ylabel('Area / horizontal-cut area')
axs[0,0].legend(fontsize=8)
fig.savefig(O/'diagnostics/fig__triangulation_ranges.png',dpi=140);plt.close(fig)

s=read('covariance_swaps_side_by_side.csv');s=s[(s.run=='canonical_lexicographic')&(s.method=='linear_P')&(s.target==.683)&(s.residual_case!=s.weight_case)]
fig,axs=plt.subplots(1,2,figsize=(9,3.7),constrained_layout=True)
for ax,b in zip(axs,['nofg','horizontal_only']):
    a=s[s.residual_case==b].set_index('kind').loc[['full_covariance','correlation_only']]
    ax.bar(['Full covariance','Correlation only'],a.area_over_own_covariance,color=['gray','steelblue'])
    ax.axhline(1,ls=':',color='black');ax.set_title('Fixed residual: '+b);ax.set_ylabel('68.3% area / own-weight area')
fig.savefig(O/'diagnostics/fig__covariance_swap_comparison.png',dpi=140);plt.close(fig)

d=read('m1080_node_comparison.csv');s=d[d.relevant_any_Q_le20]
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,value,label in zip(axs,[s.deltaQ_vs_neighbor_mean,s.absolute_integral_weight],['Q(1.080) - mean of neighbouring slopes','Absolute weighted probability difference']):
    sc=ax.scatter(s.omega_m,s.w,c=value,s=22,cmap='coolwarm' if ax is axs[0] else 'viridis');fig.colorbar(sc,ax=ax);ax.set_xlabel('Omega_m');ax.set_ylabel('w');ax.set_title(label,fontsize=9)
fig.savefig(O/'diagnostics/fig__m1080_node_diagnostic.png',dpi=140);plt.close(fig)

