"""Read-only source gates. Writes only new audit files in this directory."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent
P=O.parents[1]
D=P.parent/'genus_workbuddy/dense_band_20260917'
BR=['nofg','horizontal_only','horizontal_m1','horizontal_m5']
MODES=['z2-z1','z3-z1','z4-z1','joint']
T=O/'source_tables'
a=pd.read_csv(T/'dense_band_20260917__chisq_dense_A.csv')
b=pd.read_csv(T/'wedge_all_m_20260920__joint_chisq.csv')
c=pd.read_csv(T/'m_turnover_full_20260923__joint_chisq.csv')
keys=['branch','mode','omega_m','w']
assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
assert len(a)==4*4*1285 and len(b)==7*1285
for br in BR:
    for mode in MODES:
        s=a[(a.branch==br)&(a['mode']==mode)]
        assert len(s)==1285 and np.isfinite(s.delta_chi_squared).all()
join=a[a['mode']=='joint'].merge(b[b.branch.isin(BR)],on=keys,how='outer',indicator=True,suffixes=('_dense','_wedge'),validate='one_to_one')
assert (join['_merge']=='both').all()
join['absolute_difference']=abs(join.delta_chi_squared_dense-join.delta_chi_squared_wedge)
# Numerical equality: tolerance is stated in advance, not widened after inspection.
join['tolerance']=1e-10+1e-12*abs(join.delta_chi_squared_dense)
join['passes']=join.absolute_difference<=join.tolerance
join.to_csv(O/'tables/joint_Q_pointwise_gate.csv',index=False,float_format='%.17g')
rows=join.groupby('branch').agg(n_points=('passes','size'),max_abs_difference=('absolute_difference','max'),all_pass=('passes','all')).reset_index()
rows.to_csv(O/'tables/joint_Q_gate_summary.csv',index=False,float_format='%.17g')
assert join.passes.all(), 'STOP: shared joint Q mismatch'
reference=b[b.branch=='nofg'][['omega_m','w']].to_numpy()
for table in [a,b,c]:
    for _,s in table.groupby(['branch','mode'],sort=False):
        assert np.array_equal(s[['omega_m','w']].to_numpy(),reference),'STOP: coordinates/order mismatch'
# All original 233 points: archived geometry and the source path use full-box measurements.
geo=json.loads((P/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/geometry.json').read_text())
assert len(geo)==233*4 and all(g['objects']==1000000 for g in geo)
assert all(np.isclose(np.prod(np.asarray(g['shape'])*g['spacing']),g['volume'],rtol=1e-13) for g in geo)
# All 1052 added points: the saved contract must be A; A calls the full-box geometry function.
points=sorted((D/'points_A').glob('om*.npz'))
assert len(points)==1052
geometry_rows=[]
for path in points:
    with np.load(path) as z:
        assert str(z['contract'])=='A' and z['curves'].shape==(7,4,41)
        assert np.isfinite(z['curves']).all()
        meta=json.loads(str(z['meta']))
        assert len(meta)==4
        for g in meta:
            assert len(g['shape'])==len(g['spacing'])==3 and min(g['spacing'])>0
            geometry_rows.append(dict(omega_m=float(z['omega_m']),w=float(z['w']),snapshot=g['snap'],contract='A',volume=np.prod(np.asarray(g['shape'])*g['spacing'])))
pd.DataFrame(geometry_rows).to_csv(O/'tables/full_box_added_point_audit.csv',index=False,float_format='%.17g')
source=(O/'source_scripts/scan_sim_nearest_mesh.py').exists()
geometry_source=(O/'source_scripts/full_hi_pipeline__scan_sim_nearest_mesh.py').read_text()
assert 'retained_lengths_scan=lengths.tolist()' in geometry_source and 'discarded_widths_scan=[0.,0.,0.]' in geometry_source
assert "if contract == 'A':" in (O/'source_scripts/dense_band_20260917__run_dense_fast.py').read_text()
# Nine-slope source must retain exactly the overlapping old-branch Q values.
overlap=[]
for old,new in [('nofg','nofg'),('horizontal_only','horizontal_only'),('horizontal_m3','mgrid0'),('horizontal_m4','mgrid4'),('horizontal_m5','mgrid8')]:
    x=b[b.branch==old].delta_chi_squared.to_numpy();y=c[c.branch==new].delta_chi_squared.to_numpy()
    error=float(np.max(abs(x-y)));overlap.append(dict(old_branch=old,new_branch=new,max_abs_difference=error))
    assert np.allclose(x,y,rtol=1e-12,atol=1e-10),'STOP: nine-slope overlap mismatch'
pd.DataFrame(overlap).to_csv(O/'tables/nine_slope_overlap_gate.csv',index=False,float_format='%.17g')
manifest=[]
for path in [D/'chisq_dense_A.csv',P/'outputs/wedge_all_m_20260920/joint_chisq.csv',P/'outputs/m_turnover_full_20260923/joint_chisq.csv',P/'outputs/wedge_all_m_20260920/reference_covariances.npz',P/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/curves.npz',P/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/geometry.json']:
    manifest.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
result=dict(passed=True,shared_rows=len(join),max_abs_difference=float(join.absolute_difference.max()),atol=1e-10,rtol=1e-12,old_geometry_records=len(geo),new_contract_A_points=len(points),full_box_provenance='All old geometry retains 1e6 objects and recorded full volume; all added points are contract A. Copied source explicitly retains full AP lengths and zero discarded widths. Both Q pipelines use these same full-box sources.',source_manifest=manifest)
(O/'qa/input_gate.json').write_text(json.dumps(result,indent=2))
print(rows.to_string(index=False));print('INPUT GATES PASSED',flush=True)
