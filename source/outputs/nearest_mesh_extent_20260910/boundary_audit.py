from pathlib import Path
import csv,json
import numpy as np
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh';rows=list(csv.DictReader((D/'before_after_chisq.csv').open()));hp=json.loads((D/'integrated_hpd_summary.json').read_text());out=[]
for h in hp['results']:
 rr=[v for v in rows if v['variant']==h['variant'] and v['branch']==h['branch'] and v['mode']==h['mode']];xy=np.array([[float(v['omega_m']),float(v['w'])] for v in rr]);q=np.array([float(v['delta_chi_squared']) for v in rr]);hull=ConvexHull(xy);v=np.flatnonzero(np.any(abs(xy@hull.equations[:,:2].T+hull.equations[:,2])<1e-10,axis=1));records=[]
 for p,t in zip(hp['targets'],h['analytic']['deltaQ_levels']):
  bad=v[q[v]<=t];records.append(dict(probability=p,deltaQ_level=t,touches_sampled_boundary=bool(len(bad)),boundary_points=xy[bad].tolist(),boundary_Q_min=float(min(q[v])),measured_nodes_inside=int(sum(q<=t))))
 out.append(dict(variant=h['variant'],branch=h['branch'],mode=h['mode'],regions=records))
(D/'boundary_audit.json').write_text(json.dumps(out,indent=2))
for v in out:
 if v['variant']=='intrinsic_subtracted' and v['mode']=='joint':print(v)
