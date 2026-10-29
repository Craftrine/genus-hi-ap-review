import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from full_hi_pipeline import scan_sim_nearest_mesh as n,scan_sim_fixed_cell as old
rng=np.random.default_rng(513);p=rng.uniform(0,40,(1000,3));w=rng.uniform(.1,2,1000)
x,wt,g=n.geometry(p,w,[1,1,1],target=4,box=40);d,c=n.deposit(x,wt,g['grid_shape_scan'],g['spacing_scan']);d0,_=old.deposit(p,w,[10,10,10],4)
np.testing.assert_array_equal(d,d0)
x,wt,g=n.geometry(p,w,[.999,1.001,.998],target=4,box=40);dd,c=n.deposit(x,wt,g['grid_shape_scan'],g['spacing_scan']);np.testing.assert_allclose(dd,d0,atol=1e-6);assert g['object_count_before']==g['object_count_after'];assert c['relative_mass_error']<1e-12
np.testing.assert_allclose(np.array(g['grid_shape_scan'])*g['spacing_scan'],g['box_lengths_scan'],atol=1e-12)
a=n.prepare_spectrum(d,[4,4,4],4);b=old.prepare_spectrum(d,4,4);np.testing.assert_array_equal(a[0],b[0])
for x,y in zip(a[1],b[1]):np.testing.assert_array_equal(x,y)
nu=np.linspace(-2.5,2.5,41);f=old.filtered_field(*a,8,0);m=n.measure(f,[4,4,4],nu);m0=old.measure(f,4,nu)
for k in ['curve','thresholds','negative_euler_counts']:np.testing.assert_array_equal(m[k],m0[k])
print('PASS: full-volume tiling, object retention, mass conservation, anisotropic CIC scaling, fiducial FFT and genus equivalence')
