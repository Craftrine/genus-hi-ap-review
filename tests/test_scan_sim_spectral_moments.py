import numpy as np
from scipy.fft import rfftn,irfftn,fftn,fftfreq
from full_hi_pipeline.scan_sim_spectral_moments import spectral_moments,derivative_axes,gaussian_amplitude


def test_rfft_matches_full_complex_modes_even_and_odd():
    for shape in [(8,10,12),(7,9,11)]:
        x=np.random.default_rng(17).normal(size=shape);x-=x.mean();F=fftn(x)
        axes=[2*np.pi*fftfreq(n,2.) for n in shape]
        for i,n in enumerate(shape):
            if n%2==0:axes[i][n//2]=0
        k=np.meshgrid(*axes,indexing='ij');p=abs(F)**2/x.size**2;p[0,0,0]=0
        expected=np.array([[np.sum(p*k[i]*k[j]) for j in range(3)] for i in range(3)])
        m=spectral_moments(rfftn(x),shape,2.)
        np.testing.assert_allclose(m['variance'],np.var(x),rtol=1e-13)
        np.testing.assert_allclose(m['gradient_covariance'],expected,rtol=1e-12,atol=1e-14)


def test_three_cosine_analytic_moments_and_rescaling():
    shape=(24,30,40);spacing=2.;indices=np.indices(shape)
    x=sum(np.cos(2*np.pi*indices[i]/shape[i]) for i in range(3))
    k=2*np.pi/(np.array(shape)*spacing);F=rfftn(x)
    m=spectral_moments(F,shape,spacing)
    np.testing.assert_allclose(m['variance'],1.5,rtol=1e-13)
    np.testing.assert_allclose(m['gradient_covariance'],np.diag(k*k/2),atol=1e-14)
    np.testing.assert_allclose(m['amplitude'],np.prod(k)/(3**1.5*4*np.pi**2),rtol=1e-13)
    np.testing.assert_allclose(spectral_moments(7*F,shape,spacing)['amplitude'],m['amplitude'],rtol=1e-13)


def test_derivative_parseval_with_mixed_modes():
    shape=(12,14,16);x=np.random.default_rng(27).normal(size=shape);F=rfftn(x)
    axes=derivative_axes(shape,1.)
    derivatives=[]
    for i in range(3):
        view=[1,1,1];view[i]=len(axes[i])
        derivatives.append(irfftn(F*(1j*axes[i].reshape(view)),s=shape))
    real=np.array([[np.mean(a*b) for b in derivatives] for a in derivatives])
    np.testing.assert_allclose(spectral_moments(F,shape,1.)['gradient_covariance'],real,atol=1e-13)


def test_isotropic_limit_and_anisotropic_trace_upper_bound():
    m=gaussian_amplitude(2.,np.eye(3)*6.)
    np.testing.assert_allclose(m['amplitude'],3**1.5/(4*np.pi**2))
    x=gaussian_amplitude(1.,np.diag([1.,2.,9.]))
    assert x['isotropic_trace_amplitude']>x['amplitude']
