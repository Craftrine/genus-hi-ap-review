"""Keep the four-case tables on the same archived main triangulation."""
from pathlib import Path
import pandas as pd
import numpy as np
from area_core import Domain,TARGETS
O=Path(__file__).resolve().parent;OLD=O.parent/'revision_v08_20260930_stage2a'
xy=pd.read_csv(O/'tables/nodes.csv',float_precision='round_trip')[['omega_m','w']].to_numpy()
tri=pd.read_csv(O/'tables/main_triangles.csv')[['node0','node1','node2']].to_numpy(int)
domain=Domain(xy,tri)
data=pd.read_csv(OLD/'source_tables/dense_band_20260917__chisq_dense_A.csv',float_precision='round_trip')
rows=[];components=[]
for b in ['nofg','horizontal_only','horizontal_m1','horizontal_m5']:
    for mode in ['z2-z1','z3-z1','z4-z1','joint']:
        s=data[(data.branch==b)&(data['mode']==mode)]
        assert np.allclose(xy,s[['omega_m','w']],rtol=0,atol=1e-14)
        for method in ['linear_P','linear_Q']:
            surface=domain.surface(s.delta_chi_squared.to_numpy(),method)
            for target in TARGETS:
                result,paths,_=surface.solve(target)
                meta=dict(branch=b,mode=mode,method=method,target=target)
                rows.append(dict(**meta,**result,domain_limited=result['region_touches_hull'] or result['boundary_band_probability']>.01))
                components.extend(dict(**meta,**path) for path in paths)
pd.DataFrame(rows).to_csv(O/'tables/four_cases_main_domain_flags.csv',index=False,float_format='%.17g')
pd.DataFrame(components).to_csv(O/'tables/four_cases_main_contour_components.csv',index=False)
print('96 main-rule four-case area/closure rows written',flush=True)
