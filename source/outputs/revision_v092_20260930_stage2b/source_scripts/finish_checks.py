"""Checks and diagnostic plots for the explicitly stored canonical triangulation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path as MPath
from scipy.spatial import ConvexHull,cKDTree
from scipy.ndimage import gaussian_filter
from area_core import Domain,TARGETS

O=Path(__file__).resolve().parent;OLD=O.parent/'revision_v08_20260930_stage2a'
def read(n):return pd.read_csv(O/'tables'/n,float_precision='round_trip')
def save(d,n):pd.DataFrame(d).to_csv(O/'tables'/n,index=False,float_format='%.17g')
xy=read('nodes.csv')[['omega_m','w']].to_numpy();tri=read('main_triangles.csv')[['node0','node1','node2']].to_numpy(int)
table=pd.read_csv(OLD/'tables/eleven_wedge_nodal_Q.csv',float_precision='round_trip')
dom=Domain(xy,tri)
main=read('eleven_wedge_areas_domain_flags.csv')
branches=['nofg','horizontal_only']+[f'm{i:02d}' for i in range(1,12)]

# Detect the default CSV parser's effect and record, rather than silently hide it.
bad=pd.read_csv(OLD/'tables/eleven_wedge_nodal_Q.csv')
badxy=bad[bad.branch=='nofg'][['omega_m','w']].to_numpy()
q=table[table.branch=='horizontal_only'].Q.to_numpy()
result=Domain(badxy).surface(q,'linear_P').solve(.683,contours=False)[0]
(O/'qa/csv_roundtrip_check.json').write_text(json.dumps(dict(default_parser_changed_coordinate_entries=int((xy!=badxy).sum()),max_coordinate_difference=float(abs(xy-badxy).max()),default_parser_horizontal_683=result['area'],required_reading='pandas.read_csv(float_precision="round_trip") or archived NPZ coordinates; archived triangle indices define the main surface'),indent=2))

# Self-consistency, orientation and permutation checks for the tie rule.
ns={'__file__':str(O/'supplement.py')}
exec((O/'supplement.py').read_text().split('base=Delaunay')[0],ns)
permutation=np.random.default_rng(41).permutation(len(xy))
alt=permutation[ns['Delaunay'](xy[permutation]).simplices]
recovered,meta=ns['canonical_tri'](alt)
assert np.array_equal(recovered,tri)
uniform=dom.surface(np.zeros(len(xy)),'linear_P')
assert abs(uniform.total-dom.hull.volume)<1e-12

# Mark domain-limited four-case rows too, using the original audited surfaces.
four=pd.read_csv(OLD/'tables/four_cases_regions.csv')
four['domain_limited']=four.region_touches_hull|(four.boundary_band_probability>.01)
four['area_interpretation']='finite-domain conditional area; not a proven lower bound'
save(four,'four_cases_domain_flags_stage2a.csv')

# New fixed rule changes the surface; repeat raster checks on THIS surface.
rasters=[];hullpath=MPath(xy[ConvexHull(xy).vertices]);tree=cKDTree(xy)
sx=.0068836045056319675;sy=.01960330848343017
for n in [800,1600]:
    gx=np.linspace(.195,.425,n);gy=np.linspace(-1.325,-.67,n);X,Y=np.meshgrid(gx,gy)
    pix=(gx[1]-gx[0])*(gy[1]-gy[0]);points=np.c_[X.ravel(),Y.ravel()]
    inside=hullpath.contains_points(points).reshape(X.shape);nn=tree.query(points)[1].reshape(X.shape)
    for b in branches:
        s=table[table.branch==b];p=np.exp(-s.Q.to_numpy()/2)
        raw=mtri.LinearTriInterpolator(dom.mtri,p)(X,Y).filled(np.nan)
        raw=np.where(np.isnan(raw),p[nn],raw)
        smooth=gaussian_filter(raw,(sy/(gy[1]-gy[0]),sx/(gx[1]-gx[0])),mode='reflect')
        for variant,data in [('no_gaussian',raw),('gaussian',smooth)]:
            vals=np.sort(data[inside])[::-1];cum=np.cumsum(vals)
            for target in TARGETS:
                cutoff=vals[np.searchsorted(cum,target*cum[-1])]
                rasters.append(dict(branch=b,m=s.m.iloc[0],grid=n,variant=variant,target=target,area=float((vals>=cutoff).sum()*pix),sigma_omega_m=sx,sigma_w=sy))
        print('Canonical raster',n,b,flush=True)
r=pd.DataFrame(rasters)
for key,ix in r.groupby(['grid','variant','target']).groups.items():
    s=r.loc[ix];r.loc[ix,'area_over_horizontal']=s.area/s[s.branch=='horizontal_only'].area.iloc[0]
for key,ix in r.groupby(['branch','grid','target']).groups.items():
    s=r.loc[ix];r.loc[ix,'smooth_over_raw']=s[s.variant=='gaussian'].area.iloc[0]/s[s.variant=='no_gaussian'].area.iloc[0]
save(r,'canonical_raster_areas.csv')
c=r[r.grid==800].merge(r[r.grid==1600],on=['branch','variant','target'],suffixes=('_800','_1600'))
c['relative_change']=c.area_1600/c.area_800-1;save(c,'canonical_raster_stability.csv')
cmp=r[r.variant=='no_gaussian'].merge(main[main.method=='linear_P'][['branch','target','area']],on=['branch','target'],suffixes=('_raster','_triangle'))
cmp['relative_error']=cmp.area_raster/cmp.area_triangle-1;save(cmp,'canonical_raster_triangle_comparison.csv')

# Simple diagnostic plots; source data for every plotted point are saved CSVs.
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

# Check all protected inputs, including the accepted v0.8 manuscript.
ROOT=O.parents[1];hashes=json.loads((O/'qa/input_hashes.json').read_text())
badfiles=[n for n,h in hashes.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
assert not badfiles,badfiles
result=dict(completed=True,perturbations=20,all_area_rows=len(allareas),canonical_area_rows=len(main),
    covariance_swap_rows=len(read('covariance_swaps_side_by_side.csv')),canonical_raster_rows=len(r),
    max_new_raster_fraction_change=float(c.relative_change.abs().max()),
    max_new_raster_triangle_relative_error_1600=float(cmp[cmp.grid==1600].relative_error.abs().max()),
    main_triangulation_permutation_invariant=True,archived_input_files_unchanged=len(hashes),
    new_genus_measurements=False,stage2b_started=False,stage3_started=False,new_results_in_manuscript=False)
(O/'qa/final_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
