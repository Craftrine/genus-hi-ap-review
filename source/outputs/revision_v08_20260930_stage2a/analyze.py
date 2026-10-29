"""Stage 2a: saved-data postprocessing, no field construction/genus measurement."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MPath
from area_core import Domain,TARGETS,STEPS

O=Path(__file__).resolve().parent;P=O.parents[1]
D=P.parent/'genus_workbuddy/dense_band_20260917'
assert json.loads((O/'qa/input_gate.json').read_text())['passed']
assert (O/'qa/primary_area_gate.json').exists()
TABLES=O/'tables'
def save(frame,name):
    pd.DataFrame(frame).to_csv(TABLES/name,index=False,float_format='%.17g')
def note(msg):print(msg,flush=True)
def key(xy):return tuple(round(float(x),7) for x in xy)

four=pd.read_csv(O/'source_tables/dense_band_20260917__chisq_dense_A.csv')
five=pd.read_csv(O/'source_tables/wedge_all_m_20260920__joint_chisq.csv')
nine=pd.read_csv(O/'source_tables/m_turnover_full_20260923__joint_chisq.csv')
xy=five[five.branch=='nofg'][['omega_m','w']].to_numpy()
domain=Domain(xy)
BR=['nofg','horizontal_only','horizontal_m1','horizontal_m5']
MODES=['z2-z1','z3-z1','z4-z1','joint']
METHODS=['linear_P','linear_Q']
rows=[];components=[];vertices=[]
for br in BR:
    for mode in MODES:
        q=four[(four.branch==br)&(four['mode']==mode)].delta_chi_squared.to_numpy()
        for method in METHODS:
            surface=domain.surface(q,method)
            for target in TARGETS:
                meta=dict(branch=br,mode=mode,method=method,target=target)
                result,paths,verts=surface.solve(target)
                rows.append(dict(**meta,**result))
                components.extend(dict(**meta,**x) for x in paths)
                vertices.extend(dict(**meta,**x) for x in verts)
        note(f'Four cases: {br} {mode} complete')
save(rows,'four_cases_regions.csv');save(components,'contour_components.csv');save(vertices,'diagnostic_contour_vertices.csv')

# Eleven slopes = nine original turnover slopes + two lower saved slopes.
lo,hi=.2938445235861259,1.6921831690611855
m_old=np.linspace(lo,hi,5)
configs=[dict(branch='nofg',m=np.nan,source='nine',source_branch='nofg'),dict(branch='horizontal_only',m=0.,source='nine',source_branch='horizontal_only')]
wedge=[dict(m=m_old[0],source='five',source_branch='horizontal_m1'),dict(m=m_old[1],source='five',source_branch='horizontal_m2')]
for i in range(9):
    wedge.append(dict(m=float(nine[nine.branch==f'mgrid{i}'].m.iloc[0]),source='nine',source_branch=f'mgrid{i}'))
for i,c in enumerate(sorted(wedge,key=lambda x:x['m']),1):configs.append(dict(branch=f'm{i:02d}',**c))
assert len(configs)==13 and len(set(c['m'] for c in configs[2:]))==11
Q={};combined=[];wedge_rows=[]
for cfg in configs:
    src=five if cfg['source']=='five' else nine
    s=src[src.branch==cfg['source_branch']]
    assert np.array_equal(s[['omega_m','w']].to_numpy(),xy)
    q=s.delta_chi_squared.to_numpy();Q[cfg['branch']]=q
    combined.extend(dict(**cfg,omega_m=x,w=y,Q=v) for (x,y),v in zip(xy,q))
    for method in METHODS:
        surface=domain.surface(q,method)
        for target in TARGETS:
            r,_,_=surface.solve(target,contours=False)
            wedge_rows.append(dict(**cfg,method=method,target=target,**r))
    note('Eleven-slope triangles: '+cfg['branch'])
wedge_df=pd.DataFrame(wedge_rows)
for (method,target),indices in wedge_df.groupby(['method','target']).groups.items():
    part=wedge_df.loc[indices];h=float(part[part.branch=='horizontal_only'].area.iloc[0]);n=float(part[part.branch=='nofg'].area.iloc[0])
    wedge_df.loc[indices,'area_over_horizontal']=part.area/h
    wedge_df.loc[indices,'area_over_nofg']=part.area/n
save(wedge_df,'eleven_wedge_areas.csv');save(combined,'eleven_wedge_nodal_Q.csv');save(configs,'branch_map.csv')

# Restore only existing curves. Rounded keys align NPZ filenames to the exact
# archived CSV coordinates; triangulation always uses unrounded archived xy.
lookup={key(x):i for i,x in enumerate(xy)};assert len(lookup)==1285
g=np.full((4,1285,4,41),np.nan)
with np.load(P/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/curves.npz') as z:
    for pi,pt in enumerate(z['points']):g[:,lookup[key(pt)]]=z['curves'][:,pi].transpose(1,0,2)
source_hashes=[]
for path in sorted((D/'points_A').glob('om*.npz')):
    with np.load(path) as z:
        index=lookup[key([z['omega_m'],z['w']])]
        assert np.isnan(g[:,index]).all()
        g[:,index]=z['curves'][[0,1,2,6]]
    source_hashes.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert np.isfinite(g).all()
with np.load(P/'outputs/wedge_all_m_20260920/reference_covariances.npz') as z:
    C=z['covariances'][[0,1,2,6]];I=z['intrinsic'][:,[0,1,2,6]].transpose(1,0,2).reshape(4,123)
d=(g[:,:,1:]-g[:,:,:1]).reshape(4,1285,123);p=d-I[:,None,:]
np.savez_compressed(O/'qa/four_case_saved_residuals.npz',coordinates=xy,curves=g,d=d,intrinsic=I,residuals=p,covariances=C)
replay=[];cosine=[]
radius=np.sqrt(np.sum(((xy-[.3111,-1.])/STEPS)**2,axis=1))
for bi,br in enumerate(BR):
    for mi,mode in enumerate(MODES):
        sl=slice(0,123) if mode=='joint' else slice(41*mi,41*(mi+1))
        factor=cho_factor(C[bi,sl,sl],lower=True)
        pp=p[bi,:,sl];dd=d[bi,:,sl];ii=I[bi,sl]
        measured=np.einsum('ij,ij->i',pp,cho_solve(factor,pp.T).T)
        stored=four[(four.branch==br)&(four['mode']==mode)].delta_chi_squared.to_numpy()
        error=float(abs(measured-stored).max())
        replay.append(dict(branch=br,mode=mode,max_abs_Q_error=error))
        assert np.allclose(measured,stored,atol=1e-10,rtol=1e-12),'STOP: residual/covariance replay mismatch'
        invI=cho_solve(factor,ii);normI=float(ii@invI)
        normd=np.einsum('ij,ij->i',dd,cho_solve(factor,dd.T).T)
        cos=np.divide(dd@invI,np.sqrt(normd*normI),out=np.full(1285,np.nan),where=normd*normI>0)
        assert np.nanmax(abs(cos))<=1+1e-12
        for j,(x,y) in enumerate(xy):
            cosine.append(dict(branch=br,mode=mode,omega_m=x,w=y,delta_omega_m=x-.3111,delta_w=y+1,
                               distance_nominal_steps=radius[j],weighted_cosine=np.clip(cos[j],-1,1),
                               norm_d_squared=normd[j],norm_I_squared=normI,zero_norm=bool(normd[j]*normI<=0)))
save(replay,'residual_Q_replay_gate.csv');save(cosine,'weighted_cosine.csv')
(O/'qa/curve_source_hashes.json').write_text(json.dumps(source_hashes,indent=2))
cosdf=pd.DataFrame(cosine);bins=[0,1,2,4,8,16,32,64,128,256];summaries=[]
for (br,mode),s in cosdf.groupby(['branch','mode']):
    for left,right in zip(bins[:-1],bins[1:]):
        t=s[(s.distance_nominal_steps>=left)&(s.distance_nominal_steps<right)]
        if len(t):
            summaries.append(dict(branch=br,mode=mode,distance_lo=left,distance_hi=right,n=len(t),cos_min=t.weighted_cosine.min(),cos_q05=t.weighted_cosine.quantile(.05),cos_median=t.weighted_cosine.median(),cos_q95=t.weighted_cosine.quantile(.95),cos_max=t.weighted_cosine.max()))
save(summaries,'weighted_cosine_distance_bins.csv')
note('Saved-curve residual replay and weighted alignment complete')

swap=[];swap_Q=[]
for residual_index in [0,1]:
    for weight_index in [0,1]:
        pp=p[residual_index];factor=cho_factor(C[weight_index],lower=True)
        q=np.einsum('ij,ij->i',pp,cho_solve(factor,pp.T).T)
        meta=dict(residual_case=BR[residual_index],covariance_case=BR[weight_index])
        swap_Q.extend(dict(**meta,omega_m=x,w=y,Q=v) for (x,y),v in zip(xy,q))
        for method in METHODS:
            surface=domain.surface(q,method)
            for target in TARGETS:
                result,_,_=surface.solve(target,contours=False)
                swap.append(dict(**meta,method=method,target=target,**result))
swapdf=pd.DataFrame(swap)
for (br,method,target),idx in swapdf.groupby(['residual_case','method','target']).groups.items():
    s=swapdf.loc[idx];own=s[s.covariance_case==br].area.iloc[0]
    swapdf.loc[idx,'area_over_own_covariance']=s.area/own
save(swapdf,'covariance_swap_areas.csv');save(swap_Q,'covariance_swap_nodal_Q.csv')
note('Covariance-swap areas complete')

# Historical raster convention, with fixed physical kernel widths at both sizes.
settings=json.loads((P/'outputs/nine_m_smoothing_20260923/validation.json').read_text())
sx,sy=settings['sigma_omega_m'],settings['sigma_w']
hullpath=MPath(xy[ConvexHull(xy).vertices]);rasters=[]
for n in [800,1600]:
    gx=np.linspace(.195,.425,n);gy=np.linspace(-1.325,-.67,n);X,Y=np.meshgrid(gx,gy)
    pixel=(gx[1]-gx[0])*(gy[1]-gy[0]);inside=hullpath.contains_points(np.c_[X.ravel(),Y.ravel()]).reshape(X.shape)
    sigma=(sy/(gy[1]-gy[0]),sx/(gx[1]-gx[0]))
    for cfg in configs:
        q=Q[cfg['branch']];weights=np.exp(-q/2)
        raw=griddata(xy,weights,(X,Y),method='linear')
        near=griddata(xy,weights,(X,Y),method='nearest')
        raw=np.where(np.isnan(raw),near,raw)
        smoothed=gaussian_filter(raw,sigma=sigma,mode='reflect')
        for variant,prob in [('no_gaussian',raw),('gaussian',smoothed)]:
            vals=np.sort(prob[inside])[::-1];cum=np.cumsum(vals)
            for target in TARGETS:
                index=np.searchsorted(cum,target*cum[-1]);cut=vals[index];selected=vals>=cut
                rasters.append(dict(**cfg,grid=n,variant=variant,target=target,area=float(selected.sum()*pixel),
                                    threshold=float(cut),achieved_mass=float(vals[selected].sum()/cum[-1]),
                                    sigma_omega_m=sx if variant=='gaussian' else 0.,sigma_w=sy if variant=='gaussian' else 0.))
        note(f'Raster: {cfg["branch"]}, {n} completed')
raster=pd.DataFrame(rasters)
for (n,variant,target),idx in raster.groupby(['grid','variant','target']).groups.items():
    s=raster.loc[idx];h=s[s.branch=='horizontal_only'].area.iloc[0]
    raster.loc[idx,'area_over_horizontal']=s.area/h
for (br,n,target),idx in raster.groupby(['branch','grid','target']).groups.items():
    s=raster.loc[idx];ratio=s[s.variant=='gaussian'].area.iloc[0]/s[s.variant=='no_gaussian'].area.iloc[0]
    raster.loc[idx,'smoothed_over_unsmoothed']=ratio
save(raster,'eleven_wedge_raster_areas.csv')
coarse=raster[raster.grid==800];fine=raster[raster.grid==1600]
stability=coarse.merge(fine,on=['branch','variant','target'],suffixes=('_800','_1600'),validate='one_to_one')
stability['relative_change']=stability.area_1600/stability.area_800-1
save(stability,'raster_800_1600_stability.csv')
exact=raster[raster.variant=='no_gaussian'].merge(wedge_df[wedge_df.method=='linear_P'][['branch','target','area']],on=['branch','target'],suffixes=('_raster','_triangle'),validate='many_to_one')
exact['relative_error']=exact.area_raster/exact.area_triangle-1
save(exact,'raster_triangle_comparison.csv')
benchmark=float(raster[(raster.branch=='horizontal_only')&(raster.grid==800)&(raster.variant=='no_gaussian')&(raster.target==.683)].area.iloc[0])
assert abs(benchmark/2.435315734e-4-1)<1e-8,'STOP: historical raster benchmark mismatch'
note('800/1600 integration and smoothing comparisons complete')

# Every retained published table entry gets an explicit old/new comparison.
comparisons=[]
old_tri=pd.read_csv(P/'outputs/m_turnover_full_20260923/areas.csv')
old_raster=pd.read_csv(P/'outputs/nine_m_smoothing_20260923/areas_and_ratios.csv')
for row in wedge_df.to_dict('records'):
    if row['source']!='nine':continue
    old=old_tri[(old_tri.branch==row['source_branch'])&(old_tri.method==row['method'])&(old_tri.target==row['target'])].iloc[0]
    for metric in ['area','area_over_horizontal']:
        comparisons.append(dict(table='triangular',branch=row['branch'],source_branch=row['source_branch'],method=row['method'],target=row['target'],grid=0,variant='no_gaussian',metric=metric,old_value=old[metric],new_value=row[metric],absolute_difference=row[metric]-old[metric],relative_difference=row[metric]/old[metric]-1))
for row in raster.to_dict('records'):
    if row['source']!='nine':continue
    old=old_raster[(old_raster.branch==row['source_branch'])&(old_raster.grid==row['grid'])&(old_raster.variant==row['variant'])&(old_raster.target==row['target'])].iloc[0]
    for metric in ['area','area_over_horizontal']:
        comparisons.append(dict(table='raster',branch=row['branch'],source_branch=row['source_branch'],method='linear_P',target=row['target'],grid=row['grid'],variant=row['variant'],metric=metric,old_value=old[metric],new_value=row[metric],absolute_difference=row[metric]-old[metric],relative_difference=row[metric]/old[metric]-1))
compare=pd.DataFrame(comparisons);save(compare,'comparison_with_current_manuscript_sources.csv')
assert compare.relative_difference.abs().max()<1e-8,'STOP: manuscript source replay differs'
for source in json.loads((O/'qa/input_gate.json').read_text())['source_manifest']+source_hashes:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256'],'STOP: source changed'
result=dict(completed=True,points=1285,wedge_slopes=11,cases_including_controls=13,four_case_rows=len(rows),contour_components=len(components),wedge_area_rows=len(wedge_df),raster_rows=len(raster),cosine_rows=len(cosdf),covariance_swap_rows=len(swapdf),
            max_Q_replay_error=max(x['max_abs_Q_error'] for x in replay),max_manuscript_relative_difference=float(compare.relative_difference.abs().max()),
            horizontal_683_raster800=benchmark,max_raster_fraction_change=float(stability.relative_change.abs().max()),
            max_raster_fraction_change_wedges=float(stability[stability.branch.str.startswith('m')].relative_change.abs().max()),
            source_hashes_unchanged=True,new_genus_measurements=False,manuscript_results_modified=False)
(O/'qa/analysis_validation.json').write_text(json.dumps(result,indent=2))
note(json.dumps(result,indent=2))
