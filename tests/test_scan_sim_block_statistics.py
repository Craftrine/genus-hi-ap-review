import numpy as np
from full_hi_pipeline.scan_sim_block_statistics import block_counts,weighted_covariance
from full_hi_pipeline.scan_sim_fixed_cell import negative_euler


def test_exact_block_closure_with_unequal_shapes():
    field=np.random.default_rng(1).normal(size=(9,10,11)).astype(np.float32)
    thresholds=np.array([-.8,0.,.7])
    for n in [2,4]:
        blocks,sizes=block_counts(field,thresholds,n)
        np.testing.assert_array_equal(blocks.sum(axis=0),negative_euler(field,thresholds))
        assert sizes.sum()==field.size and np.all(sizes>0)


def test_weighted_covariance_equal_block_jackknife_identity():
    values=np.random.default_rng(2).normal(size=(64,4));f=np.full(64,1/64)
    leave=(values.sum(0)-values)/(1-f[:,None]);center=leave-leave.mean(0)
    expected=63/64*(center.T@center)
    np.testing.assert_allclose(weighted_covariance(values,f),expected,rtol=1e-13)


def test_constant_density_contributions_have_zero_covariance():
    f=np.array([.1,.2,.3,.4]);total=np.array([1.,2.])
    np.testing.assert_allclose(weighted_covariance(f[:,None]*total,f),0.,atol=1e-28)
