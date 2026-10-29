import itertools
import numpy as np
from full_hi_pipeline import scan_sim_fixed_cell as s


def test_crop_strict_spacing_and_no_return_of_discarded_objects():
    p=np.array([[.1,.1,.1],[9.5,2,2],[10.2,2,2]])
    q,w,m=s.crop_geometry(p,np.ones(3),np.ones(3),3.,10.)
    assert m['grid_shape_scan']==[3,3,3]
    assert m['retained_lengths_scan']==[9.,9.,9.]
    assert len(q)==2
    np.testing.assert_allclose(q[:,0],[.1,.2])


def test_cic_edge_renormalization_and_centers():
    p=np.array([[0.,0.,0.],[3.999,3.999,3.999],[1.,1.,1.]])
    weights=np.array([2.,3.,8.]); mass=np.zeros((4,4,4))
    s._cic(p,weights,1.,mass)
    assert mass.sum()==13.
    assert mass[0,0,0]==3.
    np.testing.assert_allclose(mass[-1,-1,-1],3.,rtol=1e-14)
    assert np.count_nonzero(mass)==9


def brute_negative_euler(active):
    total=0
    for mask in itertools.product([0,1],repeat=3):
        cells=np.ones(tuple(n-d for n,d in zip(active.shape,mask)),bool)
        for offset in itertools.product(*[range(d+1) for d in mask]):
            cells &= active[tuple(slice(o,o+n) for o,n in zip(offset,cells.shape))]
        total+=(-1)**sum(mask)*int(cells.sum())
    return -total


def test_open_counts_independent_enumeration_and_no_join():
    rng=np.random.default_rng(71);field=rng.normal(size=(4,5,6)).astype(np.float32)
    thresholds=np.array([-.8,0.,.6])
    np.testing.assert_array_equal(s.negative_euler(field,thresholds),
        [brute_negative_euler(field>t) for t in thresholds])
    field=np.zeros((4,4,4),np.float32);field[0,1,1]=field[-1,1,1]=1
    assert s.negative_euler(field,np.array([.5]))[0]==-2
    assert s.negative_euler(np.ones((4,5,6),np.float32),np.array([0.]))[0]==-1


def test_foreground_union_not_sum_and_absolute_k():
    axes=(np.array([0.,.1,1.]),np.array([0.]),np.array([-.2,-.05,0.,.05,.2]))
    horizontal=s.removal_mask(axes,1)
    np.testing.assert_array_equal(horizontal.ravel(),[False,True,True,True,False])
    previous=np.broadcast_to(horizontal,(3,1,5))
    for b in range(2,7):
        mask=s.removal_mask(axes,b)
        assert np.all(mask|~previous)
        assert not mask[1,0,-1]  # 0.2 > max(0.1, 1.6922*0.1)
        previous=mask
    assert s.removal_mask(axes,0) is None


def test_ap_identity_and_rsd_entire_position():
    np.testing.assert_allclose(s.ap_scan(.3,s.OMEGA_M_SIM,s.W_SIM),1.)
    p=np.ones((2,3));v=np.zeros((2,3));v[:,2]=100
    q=s.add_rsd_sim(p,v,.3)
    assert np.all(q[:,2]>p[:,2])
    np.testing.assert_array_equal(q[:,:2],p[:,:2])


def test_gaussian_amplitude_full_frozen_threshold_window():
    from scipy.special import ndtr
    f=np.random.default_rng(21).normal(size=(9,10,11)).astype(np.float32)
    nu=np.linspace(-2.5,2.5,41);result=s.measure(f,4.,nu)
    np.testing.assert_allclose(result['thresholds'],np.quantile(f,ndtr(nu)))
    t=(1-nu*nu)*np.exp(-nu*nu/2)
    np.testing.assert_allclose(result['amplitude_gaussian'],result['curve']@t/(t@t))


def test_fourier_gaussian_gain_and_horizontal_removal():
    # An analytic LOS mode tests filtering, not periodic genus.
    n=32;spacing=4.;indices=np.arange(n)
    k=2*np.pi/(n*spacing)
    field=np.broadcast_to(np.cos(2*np.pi*indices/n),(n,n,n)).copy().astype(np.float32)
    spectrum,axes,shape_fft,slices=s.prepare_spectrum(field,spacing,0)
    output=s.filtered_field(spectrum,axes,shape_fft,slices,5.,0)
    expected=field*np.exp(-.5*(5*k)**2)/np.sinc(1/n)**2
    np.testing.assert_allclose(output,expected,atol=3e-7)
    removed=s.filtered_field(spectrum,axes,shape_fft,slices,5.,1)
    assert abs(removed).max()<1e-7


def test_zero_extension_does_not_invent_signal():
    field=np.zeros((7,9,11),np.float32)
    spectrum,axes,shape_fft,slices=s.prepare_spectrum(field,4.,8)
    for branch in range(7):
        result=s.filtered_field(spectrum,axes,shape_fft,slices,15.,branch)
        np.testing.assert_array_equal(result,field)
