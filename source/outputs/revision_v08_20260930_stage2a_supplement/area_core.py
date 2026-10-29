"""Postprocessing only. Equations ported from the preserved solve/meanexp scripts.

Adds cached triangulation, an explicitly defined boundary band and contour audits.
The band is <=1 Euclidean unit from the convex-hull boundary after coordinates
are divided by (0.00275, 0.01). No density field or genus is measured here.
"""
from pathlib import Path
import ast
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

O=Path(__file__).resolve().parent
gn,gw=np.polynomial.legendre.leggauss(16)
ns=dict(np=np,u=(gn+1)/2,wt=gw/2)
source=O/'source_scripts/extent_mesh__exact_hpd.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='meanexp')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
meanexp=ns['meanexp']
STEPS=np.array([.00275,.01])
TARGETS=[.683,.954,.997]


def areas(v):
    return abs((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1])-(v[:,1,1]-v[:,0,1])*(v[:,2,0]-v[:,0,0]))/2


def clip_halfplane(points,normal,offset):
    """Clip a convex polygon; any trailing columns are interpolated attributes."""
    if len(points)==0:return np.empty((0,points.shape[1]))
    result=[];prev=points[-1];hp=float(prev[:2]@normal+offset)
    for cur in points:
        hc=float(cur[:2]@normal+offset)
        if (hp<=0)!=(hc<=0):
            t=hp/(hp-hc);result.append(prev+t*(cur-prev))
        if hc<=0:result.append(cur)
        prev,hp=cur,hc
    return np.asarray(result).reshape(-1,points.shape[1])


class Domain:
    def __init__(self,xy,triangles=None):
        self.xy=np.asarray(xy,float)
        self.dt=Delaunay(self.xy)
        self.tri=self.dt.simplices if triangles is None else np.asarray(triangles,int)
        # Ignore floating-point slivers on nominally collinear hull edges.
        self.tri=self.tri[areas(self.xy[self.tri])>1e-15]
        self.v=self.xy[self.tri]
        self.area=areas(self.v)
        self.hull=ConvexHull(self.xy)
        edges=np.sort(np.concatenate([self.tri[:,[0,1]],self.tri[:,[1,2]],self.tri[:,[2,0]]]),axis=1)
        edge,counts=np.unique(edges,axis=0,return_counts=True)
        assert counts.max()<=2
        self.boundary_nodes=np.unique(edge[counts==1])
        self.mtri=mtri.Triangulation(*self.xy.T,self.tri)
        self.scaled_planes=ConvexHull(self.xy/STEPS).equations
        self.inner_polygon=(self.xy/STEPS)[self.hull.vertices]
        for a,b,c in self.scaled_planes:
            self.inner_polygon=clip_halfplane(self.inner_polygon,np.array([a,b]),c+1)
        weights=[];ids=[];vertices=[]
        band_weights=[];band_ids=[];band_vertices=[]
        def add_fan(points,ti,vertices,weights,ids):
            for j in range(1,len(points)-1):
                sub=points[[0,j,j+1],:2]*STEPS
                if areas(sub[None])[0]<1e-22:continue
                vertices.append(sub);weights.append(points[[0,j,j+1],2:]);ids.append(ti)
        # Intersect every triangle with the inner hull once. Polygon fans provide
        # exact linear-P integration and analytic linear-Q integration afterward.
        for ti,v in enumerate(self.v):
            if self.area[ti]<1e-20:continue
            points=np.column_stack([v/STEPS,np.eye(3)])
            for a,b,c in self.scaled_planes:
                # Disjoint outside pieces avoid subtracting nearly equal masses
                # when the boundary-band probability is extremely small.
                outside=clip_halfplane(points,-np.array([a,b]),-c-1)
                add_fan(outside,ti,band_vertices,band_weights,band_ids)
                points=clip_halfplane(points,np.array([a,b]),c+1)
                if not len(points):break
            add_fan(points,ti,vertices,weights,ids)
        self.inner_weights=np.asarray(weights)
        self.inner_ids=np.asarray(ids)
        self.inner_areas=areas(np.asarray(vertices))
        self.band_weights=np.asarray(band_weights);self.band_ids=np.asarray(band_ids)
        self.band_areas=areas(np.asarray(band_vertices))
        assert abs(self.area.sum()-self.hull.volume)<1e-12
        vv=self.inner_polygon*STEPS
        inner_area=abs(np.sum(vv[:,0]*np.roll(vv[:,1],-1)-vv[:,1]*np.roll(vv[:,0],-1)))/2
        assert abs(self.inner_areas.sum()-inner_area)<1e-12
        assert abs(self.inner_areas.sum()+self.band_areas.sum()-self.area.sum())<1e-12

    def surface(self,q,method):
        return Surface(self,np.asarray(q),method)


class Surface:
    def __init__(self,domain,q,method):
        self.d=domain;self.q=q;self.method=method
        self.nodal=q if method=='linear_Q' else np.exp(-q/2)
        self.values=np.sort(self.nodal[domain.tri],axis=1)
        self.a,self.b,self.c=self.values.T
        self.full=domain.area*(meanexp(self.values) if method=='linear_Q' else self.values.mean(1))
        self.total=float(self.full.sum())
        inner_values=np.einsum('nij,nj->ni',domain.inner_weights,self.nodal[domain.tri[domain.inner_ids]])
        inner_mass=np.sum(domain.inner_areas*(meanexp(inner_values) if method=='linear_Q' else inner_values.mean(1)))
        band_values=np.einsum('nij,nj->ni',domain.band_weights,self.nodal[domain.tri[domain.band_ids]])
        band_mass=np.sum(domain.band_areas*(meanexp(band_values) if method=='linear_Q' else band_values.mean(1)))
        assert abs((inner_mass+band_mass)/self.total-1)<1e-10
        self.band_mass=float(band_mass/self.total)

    def integral(self,t):
        a,b,c=self.a,self.b,self.c;area=self.d.area
        frac=np.zeros(len(area));mass=np.zeros(len(area))
        if self.method=='linear_Q':
            yes=c<=t;frac[yes]=1;mass[yes]=self.full[yes]
            lo=(a<t)&(t<b)
            frac[lo]=(t-a[lo])**2/((b[lo]-a[lo])*(c[lo]-a[lo]))
            mass[lo]=area[lo]*frac[lo]*meanexp(np.column_stack([a[lo],np.full(lo.sum(),t),np.full(lo.sum(),t)]))
            hi=(b<=t)&(t<c)&(a<t)
            ex=(c[hi]-t)**2/((c[hi]-a[hi])*(c[hi]-b[hi]))
            frac[hi]=1-ex
            mass[hi]=self.full[hi]-area[hi]*ex*meanexp(np.column_stack([np.full(hi.sum(),t),np.full(hi.sum(),t),c[hi]]))
        else:
            yes=a>=t;frac[yes]=1;mass[yes]=self.full[yes]
            lo=(a<t)&(t<b)
            ex=(t-a[lo])**2/((b[lo]-a[lo])*(c[lo]-a[lo]))
            frac[lo]=1-ex;mass[lo]=self.full[lo]-area[lo]*ex*(a[lo]+2*t)/3
            hi=(b<=t)&(t<c)&(a<t)
            frac[hi]=(c[hi]-t)**2/((c[hi]-a[hi])*(c[hi]-b[hi]))
            mass[hi]=area[hi]*frac[hi]*(c[hi]+2*t)/3
        return float(mass.sum()/self.total),float((area*frac).sum())

    def solve(self,target,contours=True):
        bounds=(float(self.q.min()),float(self.q.max())) if self.method=='linear_Q' else (0.,float(self.nodal.max()))
        threshold=brentq(lambda t:self.integral(t)[0]-target,*bounds,xtol=1e-11 if self.method=='linear_Q' else 1e-13)
        mass,area=self.integral(threshold)
        assert abs(mass-target)<1e-8
        boundary_values=self.nodal[self.d.boundary_nodes]
        touches=bool(boundary_values.min()<=threshold) if self.method=='linear_Q' else bool(boundary_values.max()>=threshold)
        result=dict(area=area,threshold=threshold,achieved_mass=mass,normalization_integral=self.total,
                    region_touches_hull=touches,boundary_band_probability=self.band_mass,hull_area=self.d.hull.volume)
        paths=[];vertices=[]
        if contours:
            fig,ax=plt.subplots()
            cs=ax.tricontour(self.d.mtri,self.nodal,levels=[threshold])
            for j,segment in enumerate(cs.allsegs[0]):
                if len(segment)<2:continue
                closed=bool(np.linalg.norm((segment[0]-segment[-1])/STEPS)<1e-8)
                margins=(segment/STEPS)@self.d.scaled_planes[:,:2].T+self.d.scaled_planes[:,2]
                contact=bool(np.min(abs(margins))<1e-8)
                paths.append(dict(component=j,n_vertices=len(segment),closed=closed,touches_hull=contact,closed_inside_hull=closed and not contact))
                vertices.extend(dict(component=j,vertex=k,omega_m=float(x),w=float(y)) for k,(x,y) in enumerate(segment))
            plt.close(fig)
            result.update(n_contours=len(paths),all_contours_closed=bool(paths) and all(x['closed'] for x in paths),
                          contours_touch_hull=any(x['touches_hull'] for x in paths),
                          all_contours_closed_inside=bool(paths) and all(x['closed_inside_hull'] for x in paths))
        return result,paths,vertices
