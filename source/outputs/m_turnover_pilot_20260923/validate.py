from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent
cfg=json.loads((O/'plan.json').read_text());records=[];manifest={};errors=[]
for p in cfg['points']:
    for snap in cfg['snapshots']:
        fp=O/'measurements'/f'{p["id"]}_snap{snap}.npz'
        with np.load(fp) as f:
            meta=json.loads(str(f['metadata']));g=meta['geometry'];vol=np.prod(np.array(g['grid_shape_scan'])*np.array(g['spacing_scan']))
            assert f['curves'].shape==(10,41) and f['counts'].shape==(10,41)
            assert np.isfinite(f['curves']).all()
            err=float(np.max(abs(f['counts']/vol-f['curves'])));errors.append(err)
            assert err<1e-18
            assert np.all(np.diff(meta['removed_rfft_bins'])>=0)
            assert g['object_count_before']==g['object_count_after']==1000000
            assert g['discarded_widths_scan']==[0.,0.,0.]
            if p['id']=='sim':np.testing.assert_allclose(f['contributions'].sum(1),f['curves'],rtol=0,atol=1e-20)
            for b in range(10):records.append(dict(point_id=p['id'],snapshot=snap,branch=b,
                nx=g['grid_shape_scan'][0],ny=g['grid_shape_scan'][1],nz=g['grid_shape_scan'][2],
                dx=g['spacing_scan'][0],dy=g['spacing_scan'][1],dz=g['spacing_scan'][2],
                removed_rfft_bins=meta['removed_rfft_bins'][b],field_std=f['sigma'][b],curve_from_counts_max_abs_error=err))
        manifest[str(fp)]=hashlib.sha256(fp.read_bytes()).hexdigest()
q=pd.read_csv(O/'point_Q.csv');assert len(q)==130
assert np.all(q[q.point_id=='sim'].Q==0)
pd.DataFrame(records).to_csv(O/'geometry_and_modes.csv',index=False)
(O/'measurement_manifest.json').write_text(json.dumps(manifest,indent=2))
val=json.loads((O/'validation.json').read_text())
val.update(files_verified=52,curves_verified=520,max_count_density_error=max(errors),all_objects_retained=True,
    increasing_m_removed_bin_counts_nondecreasing=True,sim_Q_zero=True)
(O/'validation.json').write_text(json.dumps(val,indent=2));print(json.dumps(val,indent=2))
