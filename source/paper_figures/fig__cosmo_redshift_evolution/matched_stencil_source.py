import numpy as np
from numba import njit
@njit(cache=False)
def matched_hist(r,edges,nt):
    n=4
    nx,ny,nz=r.shape;h=np.zeros((n**3,nt+1),np.int64);sizes=np.zeros(n**3,np.int64)
    for b in range(n**3):
        bx=b//(n*n);by=(b//n)%n;bz=b%n
        ilo=edges[0,bx];ihi=edges[0,bx+1]
        jlo=edges[1,by];jhi=edges[1,by+1]
        klo=edges[2,bz];khi=edges[2,bz+1]
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
