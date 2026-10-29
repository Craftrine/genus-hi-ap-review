"""Area/boundary audit using the exact same triangulated-Q definition."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.spatial import Delaunay
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;D=ROOT/'paper_figures/fig__intrinsic_subtracted_contours/dense_mesh';allrows=list(csv.DictReader((D/'before_after_chisq.csv').open()));plan=json.loads((OUT/'plan.json').read_text())
# A same-domain nested coarser subset for quadrature/response sensitivity; it is not used as the main result.
coarse={(round(o,7),round(w,7)) for o in plan['omega_m'][::2] for w in plan['w'][::2]};coarse.add((.3111,-1.))
C=D/'coarse_same_domain';C.mkdir(exist_ok=True)
selected=[r for r in allrows if (round(float(r['omega_m']),7),round(float(r['w']),7)) in coarse]
with (C/'before_after_chisq.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(selected)
s=(D/'exact_hpd.py').read_text();(C/'exact_hpd.py').write_text(s)
print('Prepared same-domain coarse subset',len(coarse))
