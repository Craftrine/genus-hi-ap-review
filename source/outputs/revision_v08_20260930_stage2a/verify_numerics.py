"""Independent small numerical checks, not new genus measurements."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from area_core import Domain,meanexp
O=Path(__file__).resolve().parent
d=Domain(np.array([[0.,0.],[.1,0.],[0.,.1]]))
# A linear density with two equal low vertices has an analytic cap integral.
q=-2*np.log([.2,.2,1.])
expected_cut=brentq(lambda t:((1-t)/.8)**2*(1+2*t)/1.4-.683,.2,1)
expected_area=.005*((1-expected_cut)/.8)**2
r,_,_=d.surface(q,'linear_P').solve(.683,contours=False)
assert abs(r['area']/expected_area-1)<1e-10
# A constant density gives boundary probability = boundary geometric area / area.
constant=np.full(3,-2*np.log(.4))
expected_band=d.band_areas.sum()/d.area.sum()
uniform_checks={}
for method in ['linear_P','linear_Q']:
    value=d.surface(constant,method).band_mass
    assert abs(value-expected_band)<1e-12
    uniform_checks[method]=value
# Independent 100x100 Gauss-Legendre quadrature of representative exponential triangles.
rng=np.random.default_rng(41);values=np.sort(rng.uniform(0,650,(20,3)),axis=1);values-=values[:,0,None]
values=np.vstack([values,[[0,0,0],[0,2,2],[0,0,100],[0,.001,.002]]])
nodes,weights=np.polynomial.legendre.leggauss(100);nodes=(nodes+1)/2;weights/=2
quadrature=np.zeros(len(values))
for x,wx in zip(nodes,weights):
    for y,wy in zip(nodes,weights):
        quadrature+=2*wx*wy*(1-x)*np.exp(-.5*(values[:,1]*x+values[:,2]*(1-x)*y))
analytic=meanexp(values)
assert np.allclose(analytic,quadrature,rtol=1e-8,atol=1e-12)
s=pd.read_csv(O/'source_tables/wedge_all_m_20260920__joint_chisq.csv');s=s[s.branch=='horizontal_only']
domain=Domain(s[['omega_m','w']].to_numpy());q=s.delta_chi_squared.to_numpy();primary={}
for method,expected in [('linear_P',2.433190797e-4),('linear_Q',9.679326726e-5)]:
    r,_,_=domain.surface(q,method).solve(.683,contours=False);primary[method]=r
    assert abs(r['area']/expected-1)<1e-8
(O/'qa/primary_area_gate.json').write_text(json.dumps(primary,indent=2))
result=dict(analytic_linear_P_area=expected_area,measured_linear_P_area=d.surface(-2*np.log([.2,.2,1.]),'linear_P').solve(.683,False)[0]['area'],
            uniform_boundary_mass=uniform_checks,exponential_quadrature_max_relative_error=float(np.max(abs(analytic/quadrature-1))),
            boundary_integration='Direct sum of disjoint clipped boundary pieces; no subtraction of nearly equal total and interior masses.',passed=True)
(O/'qa/independent_numerical_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
