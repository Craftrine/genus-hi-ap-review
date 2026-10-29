from pathlib import Path
import csv,json
import numpy as np
from scipy.spatial import Delaunay
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;D=ROOT/'paper_figures/fig__intrinsic_subtracted_contours/dense_mesh';fine=json.loads((D/'integrated_hpd_summary.json').read_text());coarse=json.loads((D/'coarse_same_domain/integrated_hpd_summary.json').read_text());fr=list(csv.DictReader((D/'before_after_chisq.csv').open()));cr=list(csv.DictReader((D/'coarse_same_domain/before_after_chisq.csv').open()));results=[]
def area(rows,var,b,mode,t):
 rr=[r for r in rows if r['variant']==var and r['branch']==b and r['mode']==mode];xy=np.array([[float(r['omega_m']),float(r['w'])] for r in rr]);q=np.array([float(r['delta_chi_squared']) for r in rr]);idx=Delaunay(xy).simplices;v=xy[idx];A=abs((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1])-(v[:,1,1]-v[:,0,1])*(v[:,2,0]-v[:,0,0]))/2;a,bb,c=np.sort(q[idx],axis=1).T;f=(c<=t).astype(float);lo=(a<t)&(t<bb);f[lo]=(t-a[lo])**2/((bb[lo]-a[lo])*(c[lo]-a[lo]));hi=(bb<=t)&(t<c)&(a<t);f[hi]=1-(c[hi]-t)**2/((c[hi]-a[hi])*(c[hi]-bb[hi]));contacts=[]
 for axis,bound,name in [(0,.3071,'Omega low'),(0,.3151,'Omega high'),(1,-1.018,'w low'),(1,-.982,'w high')]:
  if np.any(np.isclose(xy[:,axis],bound,atol=1e-10)&(q<=t)):contacts.append(name)
 return float(A@f),contacts,int(np.sum(q<=t))
for r in fine['results']:
 var,b,mode=r['variant'],r['branch'],r['mode'];c=next(v for v in coarse['results'] if v['variant']==var and v['branch']==b and v['mode']==mode);entry=dict(variant=var,branch=b,mode=mode,regions=[])
 for i,p in enumerate(fine['targets']):
  af,contacts,n=area(fr,var,b,mode,r['analytic']['deltaQ_levels'][i]);ac,_,_=area(cr,var,b,mode,c['analytic']['deltaQ_levels'][i]);entry['regions'].append(dict(probability=p,fine_area=af,coarse_area=ac,area_change_percent=100*(af/ac-1),boundary_contacts=contacts,n_measured_nodes_inside=n))
 results.append(entry)
( D/'grid_sensitivity.json').write_text(json.dumps(results,indent=2))
for r in results:
 if r['variant']=='intrinsic_subtracted' and r['mode']=='joint':print(r)
