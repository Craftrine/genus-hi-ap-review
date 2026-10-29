"""Saved-array postprocessing only; no density construction, FFT or genus."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay,ConvexHull,cKDTree
from scipy.linalg import cho_factor,cho_solve
from area_core import Domain,TARGETS,STEPS,areas

O=Path(__file__).resolve().parent;ROOT=O.parents[1]
OLD=O.parent/'revision_v08_20260930_stage2a'
def save(rows,name):pd.DataFrame(rows).to_csv(O/'tables'/name,index=False,float_format='%.17g')
def note(s):print(s,flush=True)
tab=pd.read_csv(OLD/'tables/eleven_wedge_nodal_Q.csv',float_precision='round_trip')
branches=['nofg','horizontal_only']+[f'm{i:02d}' for i in range(1,12)]
xy=tab[tab.branch=='nofg'][['omega_m','w']].to_numpy()
Q={b:tab[tab.branch==b].Q.to_numpy() for b in branches}
M={b:float(tab[tab.branch==b].m.iloc[0]) for b in branches}
assert all(np.array_equal(xy,tab[tab.branch==b][['omega_m','w']].to_numpy()) for b in branches)
N=len(xy)

def ordered(tri):
    tri=np.sort(tri,axis=1)
    return tri[np.lexsort(tri.T[::-1])]

def canonical_tri(start):
    """For convex cocircular quadrilaterals, choose lexicographically smaller diagonal.

    Coordinates are original physical (Omega_m,w), not scaled. Cocircularity is
    tested after local translation and isotropic scaling, tolerance 1e-10.
    Strict convexity excludes collinear boundary configurations. Lexicographic
    vertex ranks do not depend on input row order. Archived triangles are final.
    """
    tri=np.array(start,copy=True)
    tri=tri[areas(xy[tri])>1e-15]
    rank=np.empty(N,int); rank[np.lexsort(xy.T[::-1])]=np.arange(N)
    flips=0;maxdet=0.
    for sweep in range(100):
        lookup={}
        for ti,t in enumerate(tri):
            for a,b in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]:
                lookup.setdefault(tuple(sorted((int(a),int(b)))),[]).append(ti)
        changed=0;used=set()
        for edge,ts in sorted(lookup.items(),key=lambda x:tuple(sorted(rank[list(x[0])]))):
            if len(ts)!=2 or any(t in used for t in ts):continue
            a,b=edge;c=int(next(v for v in tri[ts[0]] if v not in edge));d=int(next(v for v in tri[ts[1]] if v not in edge))
            oldkey=tuple(sorted(rank[[a,b]]));newkey=tuple(sorted(rank[[c,d]]))
            if oldkey<=newkey:continue
            pts=xy[[a,b,c,d]];pts=(pts-pts.mean(0))/np.ptp(pts,axis=0).max()
            cross=lambda u,v:u[0]*v[1]-u[1]*v[0]
            if cross(pts[1]-pts[0],pts[2]-pts[0])*cross(pts[1]-pts[0],pts[3]-pts[0])>=-1e-15:continue
            if cross(pts[3]-pts[2],pts[0]-pts[2])*cross(pts[3]-pts[2],pts[1]-pts[2])>=-1e-15:continue
            det=abs(np.linalg.det(np.column_stack([pts,np.sum(pts**2,axis=1),np.ones(4)])))
            if det>1e-10:continue
            tri[ts[0]]=[c,d,a];tri[ts[1]]=[d,c,b]
            used.update(ts);changed+=1;maxdet=max(maxdet,det)
        flips+=changed
        if not changed:break
    else:raise AssertionError('tie flips failed to terminate')
    return ordered(tri),dict(flips=flips,sweeps=sweep+1,max_flipped_normalized_determinant=maxdet)

base=Delaunay(xy).simplices
main,meta=canonical_tri(base)
save([dict(triangle=i,node0=int(a),node1=int(b),node2=int(c)) for i,(a,b,c) in enumerate(main)],'main_triangles.csv')
save([dict(node=i,omega_m=x,w=y) for i,(x,y) in enumerate(xy)],'nodes.csv')
tri_hash=hashlib.sha256(main.tobytes()).hexdigest()
all_rows=[];geometry=[];main_dom=None
for run in range(-2,20):
    if run==-2:tri=base;label='stage2a_archived';seed=None
    elif run==-1:tri=main;label='canonical_lexicographic';seed=None
    else:
        seed=20260930+run;rng=np.random.default_rng(seed)
        pert=xy*(1+1e-13*rng.uniform(-1,1,xy.shape))
        tri=Delaunay(pert).simplices;label=f'jitter_{run:02d}'
        if run<3:
            check,cmeta=canonical_tri(tri)
            assert np.array_equal(check,main),'Canonical tie rule differs across initial triangulations'
    domain=Domain(xy,tri)
    if run==-1:main_dom=domain
    geometry.append(dict(run=label,seed=seed,n_triangles=len(domain.tri),hull_area=domain.hull.volume,
                         integrated_triangle_area=float(domain.area.sum()),triangles_sha256=hashlib.sha256(ordered(domain.tri).tobytes()).hexdigest()))
    for b in branches:
        for method in ['linear_P','linear_Q']:
            s=domain.surface(Q[b],method)
            for target in TARGETS:
                r,_,_=s.solve(target,contours=False)
                rows=dict(run=label,branch=b,m=M[b],method=method,target=target,**r)
                rows['domain_limited']=r['region_touches_hull'] or r['boundary_band_probability']>.01
                rows['area_interpretation']='finite-domain conditional area; not a proven lower bound'
                all_rows.append(rows)
    note(f'{label}: both interpolations completed')
df=pd.DataFrame(all_rows)
for (run,method,target),ix in df.groupby(['run','method','target']).groups.items():
    s=df.loc[ix];h=s[s.branch=='horizontal_only'].area.iloc[0]
    df.loc[ix,'area_over_horizontal']=s.area/h
save(df,'triangulation_all_areas.csv')
save(geometry,'triangulation_geometry.csv')
main_df=df[df.run=='canonical_lexicographic'].copy();save(main_df,'eleven_wedge_areas_domain_flags.csv')
old=pd.read_csv(OLD/'tables/eleven_wedge_areas.csv')
replay=df[df.run=='stage2a_archived'].merge(old,on=['branch','method','target'],suffixes=('_now','_old'))
assert np.max(abs(replay.area_now/replay.area_old-1))<1e-8
cmp=main_df.merge(old[['branch','method','target','area','area_over_horizontal']],on=['branch','method','target'],suffixes=('_new','_stage2a'))
cmp['area_relative_change']=cmp.area_new/cmp.area_stage2a-1
cmp['ratio_relative_change']=cmp.area_over_horizontal_new/cmp.area_over_horizontal_stage2a-1
save(cmp,'main_vs_stage2a.csv')
jitter=df[df.run.str.startswith('jitter')]
ranges=[]
for (b,method,target),s in jitter.groupby(['branch','method','target']):
    r=dict(branch=b,m=M[b],method=method,target=target,n_runs=len(s))
    for col in ['area','area_over_horizontal']:
        r.update({col+'_min':s[col].min(),col+'_max':s[col].max(),col+'_median':s[col].median(),col+'_relative_span':s[col].max()/s[col].min()-1})
    ranges.append(r)
save(ranges,'triangulation_ranges.csv')
mins=[]
for (run,method,target),s in df[df.branch.str.startswith('m')].groupby(['run','method','target']):
    r=s.loc[s.area.idxmin()];mins.append(dict(run=run,method=method,target=target,branch=r.branch,m=r.m,area=r.area,ratio=r.area_over_horizontal))
save(mins,'sampled_minima.csv')
(O/'qa/triangulation_rule.json').write_text(json.dumps(dict(rule=canonical_tri.__doc__,canonical_hash=tri_hash,metadata=meta,initial_jitter_rules_verified=3,jitter_runs=20,jitter_formula='xy * (1 + 1e-13 * uniform(-1,1)), seed=20260930+run; integrate on original xy',relative_precision_warning='ranges describe tested triangulations, not confidence intervals'),indent=2))

# Keep own variances; exchange only regularized correlation matrices.
with np.load(OLD/'qa/four_case_saved_residuals.npz') as z:
    C=z['covariances'];p=z['residuals'];curves=z['curves'];I=z['intrinsic'];assert np.array_equal(xy,z['coordinates'])
std=np.sqrt(np.diagonal(C,axis1=1,axis2=2));corr=C/std[:,:,None]/std[:,None,:]
swap=[];swapnodal=[];validation=[]
for ri,rname in enumerate(branches[:2]):
    for wi,wname in enumerate(branches[:2]):
        for kind in ['full_covariance','correlation_only']:
            cov=C[wi] if kind=='full_covariance' else std[ri,:,None]*corr[wi]*std[ri,None,:]
            assert np.allclose(cov,cov.T)
            factor=cho_factor(cov,lower=True)
            q=np.einsum('ij,ij->i',p[ri],cho_solve(factor,p[ri].T).T)
            validation.append(dict(residual_case=rname,weight_case=wname,kind=kind,min_eigenvalue=float(np.linalg.eigvalsh(cov).min()),max_relative_variance_change=float(abs(np.diag(cov)/np.diag(C[ri])-1).max())))
            swapnodal.extend(dict(residual_case=rname,weight_case=wname,kind=kind,node=j,omega_m=x,w=y,Q=v) for j,((x,y),v) in enumerate(zip(xy,q)))
            for dn,dom in [('stage2a_archived',Domain(xy,base)),('canonical_lexicographic',main_dom)]:
                for method in ['linear_P','linear_Q']:
                    s=dom.surface(q,method)
                    for target in TARGETS:
                        r,_,_=s.solve(target,contours=False)
                        swap.append(dict(run=dn,residual_case=rname,weight_case=wname,kind=kind,method=method,target=target,domain_limited=r['region_touches_hull'] or r['boundary_band_probability']>.01,**r))
swap=pd.DataFrame(swap)
for keys,ix in swap.groupby(['run','kind','residual_case','method','target']).groups.items():
    s=swap.loc[ix];ref=s[s.weight_case==keys[2]].area.iloc[0]
    swap.loc[ix,'area_over_own_covariance']=s.area/ref
save(swap,'covariance_swaps_side_by_side.csv');save(swapnodal,'covariance_swaps_nodal_Q.csv');save(validation,'covariance_validation.csv')
note('Correlation-only and original swaps completed')

# Weak wedge: all differences, not only rounded values.
dq=Q['m01']-Q['horizontal_only'];dc=curves[2]-curves[1]
save([dict(node=i,omega_m=x,w=y,Q_horizontal=Q['horizontal_only'][i],Q_weak=Q['m01'][i],delta_Q=dq[i],equal_Q=bool(dq[i]==0),max_abs_curve_difference=float(abs(dc[i]).max())) for i,(x,y) in enumerate(xy)],'weak_wedge_pointwise.csv')
np.savez_compressed(O/'qa/weak_wedge_curve_difference.npz',coordinates=xy,curve_difference=dc)
(O/'qa/weak_wedge_comparison.json').write_text(json.dumps(dict(equal_Q_count=int((dq==0).sum()),n_points=N,max_abs_delta_Q=float(abs(dq).max()),unequal_curve_points=int(np.any(dc!=0,axis=(1,2)).sum()),covariance_exact_equal=bool(np.array_equal(C[1],C[2])),template_exact_equal=bool(np.array_equal(I[1],I[2]))),indent=2))

# The m=1.0804 bump: node anomalies and their spatial distribution.
qlo,qmid,qhi=[Q[b] for b in ['m03','m04','m05']]
anom=qmid-(qlo+qhi)/2
nearest=cKDTree(xy/STEPS).query(xy/STEPS,k=9)[1][:,1:]
local=anom-np.median(anom[nearest],axis=1)
prob=[np.exp(-q/2) for q in [qlo,qmid,qhi]]
delta_prob=prob[1]-(prob[0]+prob[2])/2
nodeweights=np.zeros(N)
for t,a in zip(main_dom.tri,main_dom.area):nodeweights[t]+=a/3
frames=pd.DataFrame(dict(node=np.arange(N),omega_m=xy[:,0],w=xy[:,1],Q_m03=qlo,Q_m04=qmid,Q_m05=qhi,
    deltaQ_m04_m03=qmid-qlo,deltaQ_m04_m05=qmid-qhi,deltaQ_vs_neighbor_mean=anom,
    local_deltaQ_anomaly=local,delta_P=delta_prob,absolute_integral_weight=np.abs(delta_prob)*nodeweights,
    relevant_any_Q_le20=np.minimum.reduce([qlo,qmid,qhi])<=20))
frames['rank_abs_deltaQ']=frames.deltaQ_vs_neighbor_mean.abs().rank(method='first',ascending=False).astype(int)
frames['rank_integral_impact']=frames.absolute_integral_weight.rank(method='first',ascending=False).astype(int)
save(frames,'m1080_node_comparison.csv')
save(frames.sort_values('rank_abs_deltaQ').head(30),'m1080_top_Q_nodes.csv')
save(frames.sort_values('rank_integral_impact').head(30),'m1080_top_probability_nodes.csv')
concentration=[]
for subset,s in [('all',frames),('any_Q_le20',frames[frames.relevant_any_Q_le20])]:
    for metric in ['deltaQ_vs_neighbor_mean','local_deltaQ_anomaly','absolute_integral_weight']:
        vals=np.sort(np.abs(s[metric]))[::-1];cum=np.cumsum(vals)
        row=dict(subset=subset,metric=metric,n_nodes=len(s),n50=int(np.searchsorted(cum,.5*cum[-1])+1),n90=int(np.searchsorted(cum,.9*cum[-1])+1))
        for count in [1,5,10,20,50]:row[f'top{count}_fraction']=float(vals[:count].sum()/vals.sum())
        concentration.append(row)
save(concentration,'m1080_concentration.csv')
counts=[]
for b in branches:
    for target in TARGETS:
        r=main_df[(main_df.branch==b)&(main_df.method=='linear_P')&(main_df.target==target)].iloc[0]
        counts.append(dict(branch=b,m=M[b],target=target,nodes_Q_lt2p3=int((Q[b]<2.3).sum()),nodes_above_linearP_HPD_threshold=int((np.exp(-Q[b]/2)>=r.threshold).sum())))
save(counts,'region_node_counts.csv')
note('All supplement areas and node diagnostics complete')
