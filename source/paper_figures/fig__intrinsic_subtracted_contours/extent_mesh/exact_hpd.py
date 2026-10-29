"""Exact exponential integral on linear-Q triangles, including clipped HPD sets."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.optimize import brentq
OUT=Path(__file__).resolve().parent;rows=list(csv.DictReader((OUT/'before_after_chisq.csv').open()));targets=[.683,.954,.997]
gn,gw=np.polynomial.legendre.leggauss(16);u=(gn+1)/2;wt=gw/2

def meanexp(values):
 q=np.sort(np.asarray(values),axis=1);a=q[:,0];B=(q[:,1]-a)/2;C=(q[:,2]-a)/2;out=np.empty(len(a));small=C<.01
 # Stable bounded-range independent quadrature handles coincident vertices.
 if np.any(small):
  bs=B[small];cs=C[small];rr=np.zeros(len(bs))
  for x,wx in zip(u,wt):
   for y,wy in zip(u,wt):rr+=2*wx*wy*(1-x)*np.exp(-bs*x-cs*(1-x)*y)
  out[small]=np.exp(-a[small]/2)*rr
 normal=~small
 if np.any(normal):
  b=B[normal];c=C[normal];same=abs(c-b)<1e-8*np.maximum(1,c);v=np.empty(len(b))
  v[same]=2*(1-(1+b[same])*np.exp(-b[same]))/(b[same]**2)
  other=~same;bo=b[other];co=c[other];fb=np.ones(len(bo));nz=bo>1e-12;fb[nz]=-np.expm1(-bo[nz])/bo[nz];fc=-np.expm1(-co)/co
  v[other]=2*(fb-fc)/(co-bo);out[normal]=np.exp(-a[normal]/2)*v
 return out
results=[]
for var in ['intrinsic_subtracted','intrinsic_not_subtracted']:
 for branch in ['nofg','horizontal_only','horizontal_m1','horizontal_m5']:
  for mode in ['z2-z1','z3-z1','z4-z1','joint']:
   rr=[r for r in rows if r['variant']==var and r['branch']==branch and r['mode']==mode];xy=np.array([[float(r['omega_m']),float(r['w'])] for r in rr]);q=np.array([float(r['delta_chi_squared']) for r in rr]);tr=Delaunay(xy).simplices;v=xy[tr];area=abs((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1])-(v[:,1,1]-v[:,0,1])*(v[:,2,0]-v[:,0,0]))/2
   assert abs(area.sum()-ConvexHull(xy).volume)<1e-12
   vals=np.sort(q[tr],axis=1);a,b,c=vals.T;full=area*meanexp(vals);total=full.sum()
   def mass(t):
    out=np.where(c<=t,full,0.)
    lo=(a<t)&(t<b)
    if np.any(lo):
     frac=(t-a[lo])**2/((b[lo]-a[lo])*(c[lo]-a[lo]));out[lo]=area[lo]*frac*meanexp(np.column_stack([a[lo],np.full(lo.sum(),t),np.full(lo.sum(),t)]))
    hi=(b<=t)&(t<c)&(a<t)
    if np.any(hi):
     frac=(c[hi]-t)**2/((c[hi]-a[hi])*(c[hi]-b[hi]));out[hi]=full[hi]-area[hi]*frac*meanexp(np.column_stack([np.full(hi.sum(),t),np.full(hi.sum(),t),c[hi]]))
    return out.sum()/total
   cut=[brentq(lambda t:mass(t)-target,float(q.min())-1e-9,float(q.max())+1e-9,xtol=1e-10) for target in targets];ach=[mass(t) for t in cut]
   assert max(abs(np.array(ach)-targets))<1e-8
   results.append(dict(variant=var,branch=branch,mode=mode,targets=targets,analytic=dict(deltaQ_levels=cut,integrated_mass=ach,normalization_integral=float(total))))
   print(var,branch,mode,np.round(cut,4),flush=True)
# Cross-check integral against Gauss-Legendre quadrature for representative wide triangles.
rng=np.random.default_rng(41);test=np.sort(rng.uniform(0,650,(40,3)),axis=1);test-=test[:,0,None];ana=meanexp(test);nodes,weights=np.polynomial.legendre.leggauss(100);nodes=(nodes+1)/2;weights=weights/2;quad=np.zeros(40)
for x,wx in zip(nodes,weights):
 for y,wy in zip(nodes,weights):quad+=2*wx*wy*(1-x)*np.exp(-.5*(test[:,1]*x+test[:,2]*(1-x)*y))
np.testing.assert_allclose(ana,quad,rtol=1e-8,atol=1e-12)
(OUT/'integrated_hpd_summary.json').write_text(json.dumps(dict(prior='uniform in Omega_m,w on sampled convex hull; uniform in Omega_m,w',density='exp(-DeltaQ/2)',definition='HPD with analytic exponential integrals on clipped piecewise-linear Q triangles',targets=targets,quadrature_crosscheck_passed=True,results=results),indent=2))
with (OUT/'integrated_hpd_levels.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['variant','branch','mode','target_mass','deltaQ_level','achieved_mass'])
 for r in results:
  for i,t in enumerate(targets):w.writerow([r['variant'],r['branch'],r['mode'],t,r['analytic']['deltaQ_levels'][i],r['analytic']['integrated_mass'][i]])
