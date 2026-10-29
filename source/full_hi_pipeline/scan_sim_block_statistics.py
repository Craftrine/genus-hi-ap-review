"""Additive open-Euler block contributions at fixed full-field thresholds.

This is conditional spatial resampling, NOT catalogue deletion/re-filtering or
the topology of a newly cut domain. Each stencil belongs to its low-index node.
"""
import numpy as np
from numba import njit, prange
from .fast_counting import _encode_ranks


@njit(cache=True,parallel=True)
def _hist(r,n,nt):
    nx,ny,nz=r.shape;h=np.zeros((n**3,nt+1),np.int64);sizes=np.zeros(n**3,np.int64)
    for b in prange(n**3):
        bx=b//(n*n);by=(b//n)%n;bz=b%n
        ilo=(bx*nx+n-1)//n;ihi=((bx+1)*nx+n-1)//n
        jlo=(by*ny+n-1)//n;jhi=((by+1)*ny+n-1)//n
        klo=(bz*nz+n-1)//n;khi=((bz+1)*nz+n-1)//n
        sizes[b]=(ihi-ilo)*(jhi-jlo)*(khi-klo)
        for i in range(ilo,ihi):
            for j in range(jlo,jhi):
                for k in range(klo,khi):
                    a=r[i,j,k];h[b,a]-=1
                    if i+1<nx:h[b,min(a,r[i+1,j,k])]+=1
                    if j+1<ny:h[b,min(a,r[i,j+1,k])]+=1
                    if k+1<nz:h[b,min(a,r[i,j,k+1])]+=1
                    if i+1<nx and j+1<ny:h[b,min(a,r[i+1,j,k],r[i,j+1,k],r[i+1,j+1,k])]-=1
                    if i+1<nx and k+1<nz:h[b,min(a,r[i+1,j,k],r[i,j,k+1],r[i+1,j,k+1])]-=1
                    if j+1<ny and k+1<nz:h[b,min(a,r[i,j+1,k],r[i,j,k+1],r[i,j+1,k+1])]-=1
                    if i+1<nx and j+1<ny and k+1<nz:
                        h[b,min(a,r[i+1,j,k],r[i,j+1,k],r[i,j,k+1],r[i+1,j+1,k],r[i+1,j,k+1],r[i,j+1,k+1],r[i+1,j+1,k+1])]+=1
    return h,sizes


def block_counts(field,thresholds,n):
    if min(field.shape)<n or len(thresholds)>254:raise ValueError('Invalid partition')
    ranks=np.empty(field.shape,np.uint8)
    if _encode_ranks(field,np.asarray(thresholds),ranks):raise ValueError('Invalid field')
    hist,sizes=_hist(ranks,n,len(thresholds))
    return np.cumsum(hist[:,::-1],axis=1,dtype=np.int64)[:,::-1][:,1:],sizes


def weighted_covariance(block_totals, fractions):
    """Weighted spatial block covariance for a sum of local contributions.

Pseudo-values are block_totals / volume_fraction. For equal blocks this is
exactly the usual delete-one jackknife covariance. Weights handle 62/63-cell
blocks without pretending their volumes are equal. Blocks across redshifts
and scales share the same sim spatial index; independence remains approximate.
"""
    fractions=np.asarray(fractions,float)
    pseudo=block_totals/fractions[:,None]
    center=pseudo-(fractions[:,None]*pseudo).sum(axis=0)
    b=len(fractions)
    return (center.T@(fractions[:,None]*center))/(b-1)
