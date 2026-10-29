"""Anisotropic Gaussian moment prediction; see SPECTRAL_COMPARISON_CONTRACT.

Does not compute the realized topology. No binning, no repeated filters,
no shot-noise subtraction. rFFT multiplicities and real derivative symbols.
"""
import numpy as np
from scipy.fft import fftfreq,rfftfreq


def derivative_axes(shape,spacing):
    axes=[2*np.pi*fftfreq(shape[0],spacing),2*np.pi*fftfreq(shape[1],spacing),
          2*np.pi*rfftfreq(shape[2],spacing)]
    for i,n in enumerate(shape):
        if n%2==0:axes[i][n//2]=0.
    return axes


def gaussian_amplitude(variance,gradient_covariance):
    if not np.isfinite(variance) or variance<=0:raise ValueError('Positive variance required')
    matrix=np.asarray(gradient_covariance,float)/variance
    eig=np.linalg.eigvalsh(matrix)
    if eig.min() < -1e-12*max(float(eig.max()),1e-30):raise ValueError('Gradient covariance not PSD')
    amplitude=float(np.sqrt(np.prod(np.maximum(eig,0)))/(4*np.pi**2))
    return dict(amplitude=amplitude,normalized_gradient_covariance=matrix,
                gradient_eigenvalues=eig,isotropic_trace_amplitude=float((np.trace(matrix)/3)**1.5/(4*np.pi**2)))


def spectral_moments(spectrum,shape,spacing):
    if spectrum.shape!=(shape[0],shape[1],shape[2]//2+1):raise ValueError('rFFT shape mismatch')
    axes=derivative_axes(shape,spacing)
    multiplicity=np.full(spectrum.shape[2],2.);multiplicity[0]=1.
    if shape[2]%2==0:multiplicity[-1]=1.
    total=0.;gradient=np.zeros((3,3));nyquist=0.
    ky=axes[1][:,None];kz=axes[2][None,:]
    for ix,kx in enumerate(axes[0]):
        slab=spectrum[ix];power=(slab.real.astype(float)**2+slab.imag.astype(float)**2)*multiplicity[None,:]
        if ix==0:power[0,0]=0.
        if not np.isfinite(power).all():raise ValueError('Nonfinite spectrum')
        total+=power.sum()
        ks=[kx,ky,kz]
        for i in range(3):
            for j in range(i,3):gradient[i,j]+=np.sum(power*ks[i]*ks[j])
        mask=np.zeros(power.shape,bool)
        if shape[0]%2==0 and ix==shape[0]//2:mask[:]=True
        if shape[1]%2==0:mask[shape[1]//2,:]=True
        if shape[2]%2==0:mask[:,-1]=True
        nyquist+=power[mask].sum()
    gradient=gradient+gradient.T-np.diag(np.diag(gradient))
    normalization=float(np.prod(shape))**2
    variance=float(total/normalization);gradient/=normalization
    return dict(variance=variance,gradient_covariance=gradient,
        nyquist_power_fraction=float(nyquist/total),**gaussian_amplitude(variance,gradient))
