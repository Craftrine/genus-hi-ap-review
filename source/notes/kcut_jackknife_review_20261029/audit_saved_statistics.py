"""Author: Wells. Audit saved blocks + small counting examples; no production execution."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import ast,csv,json,hashlib,itertools
import numpy as np
from sklearn.covariance import oas
import sklearn
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
sources={}
def load(rel):
 p=R/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return np.load(p)
def extract(rel,names):
 p=R/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest();ns={'np':np,'prange':range}
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
 for n in nodes:n.decorator_list=[]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);return ns
ns=extract('full_hi_pipeline/fast_counting.py',['_left_rank','_encode_ranks'])
match=extract('paper_figures/fig__cosmo_redshift_evolution/matched_stencil_source.py',['matched_hist'])['matched_hist']
wcov=extract('full_hi_pipeline/scan_sim_block_statistics.py',['weighted_covariance'])['weighted_covariance']
def independent(field,threshold,edges):
 occupied=field>threshold;parts=np.zeros(64,dtype=np.int64)
 # Enumerate cubical elements by dimension; assign each to the low-index vertex.
 for dim in range(4):
  for axes in itertools.combinations(range(3),dim):
   limits=[field.shape[i]-(i in axes) for i in range(3)]
   for anchor in np.ndindex(*limits):
    vertices=[tuple(anchor[k]+(bits[axes.index(k)] if k in axes else 0) for k in range(3)) for bits in itertools.product([0,1],repeat=dim)]
    if all(occupied[v] for v in vertices):
     bb=[np.searchsorted(edges[k],anchor[k],side='right')-1 for k in range(3)]
     parts[bb[0]*16+bb[1]*4+bb[2]]+=(-1 if dim%2==0 else 1)
 return parts
rng=np.random.default_rng(629);small_tests=0
for shape in [(4,4,4),(5,6,7),(8,8,8)]:
 edges=np.array([np.clip(np.ceil(np.arange(5)*n/4-.5).astype(int),0,n) for n in shape])
 fields=[rng.integers(-2,3,size=shape).astype(float),np.ones(shape),np.zeros(shape)]
 for field in fields:
  tt=np.array([-2.,0.,0.,1.,3.]);ranks=np.empty(shape,np.uint8)
  assert ns['_encode_ranks'](field,tt,ranks)==0
  assert np.array_equal(ranks,np.searchsorted(tt,field,side='left'))
  hist,sizes=match(ranks,edges,len(tt));counts=np.cumsum(hist[:,::-1],axis=1)[:,::-1][:,1:]
  assert sizes.sum()==field.size and (sizes>0).all()
  for j,t in enumerate(tt):np.testing.assert_array_equal(counts[:,j],independent(field,t,edges));small_tests+=1
# Exact equal-block normalization identity on synthetic additive totals.
t=rng.normal(size=(64,7));frac=np.full(64,1/64);g=t.sum(0);lv=(g-t)/(1-frac[:,None]);X=lv-lv.mean(0)
np.testing.assert_allclose(63/64*X.T@X,wcov(t,frac),rtol=1e-12)
SN=[118,112,105,94];BR=['nofg','horizontal_only','horizontal_m1','horizontal_m5'];amp_rows=list(csv.DictReader((R/'paper/english_revision_v0.20_full_redshift_table_20261029/data/sample_amplitudes.csv').open()))
nu=np.linspace(-2.5,2.5,41);tpl=(1-nu**2)*np.exp(-nu**2/2)
records=[];amp_errors=[];curve_errors=[];difference_errors=[];max_add=0.;frac_ranges=[]
arrays=load('outputs/nearest_mesh_figure_refresh_20260910/refreshed_arrays.npz')
# Main 48 curves / 36 redshift differences, matching their saved error columns.
global_curves={}
for ci in range(3):
 for zi,sn in enumerate(SN):
  a=load(f'outputs/nearest_mesh_figure_refresh_20260910/c{ci}_snap{sn}.npz');g=a['curves'];t=a['contributions'];f=a['fractions']
  max_add=max(max_add,float(abs(t.sum(1)-g).max()));np.testing.assert_allclose(t.sum(1),g,atol=1e-20,rtol=1e-13)
  lv=(g[:,None,:]-t)/(1-f[None,:,None]);np.testing.assert_allclose(lv,arrays['loo'][ci,zi],atol=0,rtol=0)
  for bi,b in enumerate(BR):global_curves[ci,sn,b]=lv[bi]
  frac_ranges.append({'ci':ci,'snapshot':sn,'min_fraction':float(f.min()),'max_fraction':float(f.max()),'max_relative_to_equal':float(abs(f*64-1).max())})
for row in csv.DictReader((R/'paper_figures/fig__cosmo_redshift_evolution/data.csv').open()):
 lv=global_curves[int(row['cosmology_id']),int(row['snapshot']),row['branch']];j=int(round((float(row['nu_f'])+2.5)*8));e=np.sqrt(63/64*np.sum((lv[:,j]-lv[:,j].mean())**2));curve_errors.append(abs(e-float(row['sigma_g'])))
for row in csv.DictReader((R/'paper_figures/fig__cosmo_redshift_evolution/differences.csv').open()):
 ci=int(row['cosmology_id']);sn=int(row['snapshot']);br=row['branch'];lv=global_curves[ci,sn,br]-global_curves[ci,118,br];j=int(round((float(row['nu_f'])+2.5)*8));e=np.sqrt(63/64*np.sum((lv[:,j]-lv[:,j].mean())**2));difference_errors.append(abs(e-float(row['sigma_delta_g'])))
# Both samples' amplitude error bars, including all intermediate snapshots.
for sample in ['top1m','all_halo']:
 for sn in SN:
  rel=f'outputs/nearest_mesh_figure_refresh_20260910/c0_snap{sn}.npz' if sample=='top1m' else f'outputs/all_halo_vs_top1m_20260911/snap{sn}/fiducial_blocks.npz'
  a=load(rel);lv=(a['curves'][:,None,:]-a['contributions'])/(1-a['fractions'][None,:,None])
  for bi,b in enumerate(BR):
   aa=lv[bi]@tpl/(tpl@tpl)*8000;sig=np.sqrt(63/64*np.sum((aa-aa.mean())**2));row=next(r for r in amp_rows if r['sample']==sample and int(r['snapshot'])==sn and r['branch']==b);amp_errors.append(abs(sig-float(row['AR3_sigma'])))
# Seven-case reference plus nine-slope pilot = all eleven distinct wedge slopes + controls.
def audit_group(label,files,refpath,ms):
 dat=[load(x) for x in files];g=np.array([a['curves'] for a in dat]);t=np.array([a['contributions'] for a in dat]);f=np.array([a['fractions'] for a in dat]);saved=load(refpath)
 assert np.array_equal(f,np.broadcast_to(f[0],f.shape));np.testing.assert_allclose(t.sum(2),g,atol=1e-20)
 lv=(g[:,:,None,:]-t)/(1-f[:,None,:,None]);np.testing.assert_allclose(g[1:]-g[:1],saved['intrinsic'],atol=0,rtol=0)
 for bi,m in enumerate(ms):
  samples=(lv[1:,bi]-lv[0,bi]).transpose(1,0,2).reshape(64,123);X=samples-samples.mean(0);raw=63/64*X.T@X;sd=np.sqrt(np.diag(raw));Z=X/np.sqrt(np.mean(X*X,axis=0));corr,rho=oas(Z,assume_centered=True);C=corr*sd[:,None]*sd[None,:]
  np.testing.assert_allclose(C,saved['covariances'][bi],atol=1e-28,rtol=1e-12)
  np.testing.assert_allclose(C,(1-rho)*raw+rho*np.diag(np.diag(raw)),atol=1e-28,rtol=1e-12)
  tt=(t[1:,bi]-t[0,bi]).transpose(1,0,2).reshape(64,123);cw=wcov(tt,f[0]);ratio=np.sqrt(np.diag(cw)/np.diag(raw))-1
  crosses=[]
  for iz in range(1,4):
   x0=lv[0,bi]-lv[0,bi].mean(0);xi=lv[iz,bi]-lv[iz,bi].mean(0)
   direct=63/64*(xi-x0).T@(xi-x0);expanded=63/64*(xi.T@xi+x0.T@x0-xi.T@x0-x0.T@xi)
   np.testing.assert_allclose(direct,expanded,atol=1e-26,rtol=1e-12);crosses.append(float(np.median(np.diag(direct)/np.diag(63/64*(xi.T@xi+x0.T@x0)))))
  records.append(dict(group=label,branch_index=bi,m=m,rho=float(rho),rank=int(np.linalg.matrix_rank(raw)),condition=float(np.linalg.cond(C)),minimum_eigenvalue=float(np.linalg.eigvalsh(C).min()),saved_cov_max_abs_error=float(abs(C-saved['covariances'][bi]).max()),oas_diagonal_relative_error=float(abs(np.diag(C)/np.diag(raw)-1).max()),weighted_sigma_relative_min=float(ratio.min()),weighted_sigma_relative_max=float(ratio.max()),paired_variance_over_unpaired_median=crosses))
# Build the seven-case arrays from separate measured caches as in the production archive.
seven=[]
for sn in SN:
 a=load(f'outputs/nearest_mesh_figure_refresh_20260910/c0_snap{sn}.npz');b=load(f'outputs/wedge_all_m_20260920/c0_snap{sn}.npz')
 q={key:np.array([a[key][0],a[key][1],a[key][2],b[key][1],b[key][2],b[key][3],a[key][3]]) for key in ['curves','contributions']};q['fractions']=a['fractions']
 path=O/f'seven_snap{sn}.npz';np.savez_compressed(path,**q);seven.append(str(path.relative_to(R)))
audit_group('seven_reference',seven,'outputs/wedge_all_m_20260920/reference_covariances.npz',['nofg','horizontal']+list(np.linspace(.2938445235861259,1.6921831690611855,5)))
ms=np.linspace(np.linspace(.2938445235861259,1.6921831690611855,5)[2],1.6921831690611855,9)
audit_group('nine_slope_pilot',[f'outputs/m_turnover_pilot_20260923/measurements/sim_snap{sn}.npz' for sn in SN],'outputs/m_turnover_pilot_20260923/references.npz',['horizontal']+list(ms))
assert max(curve_errors)<1e-18 and max(difference_errors)<1e-18 and max(amp_errors)<1e-15
summary=dict(author='Wells',new_field_measurements=0,small_counting_tests=small_tests,curve_error_entries=len(curve_errors),curve_sigma_max_abs_error=max(curve_errors),difference_error_entries=len(difference_errors),difference_sigma_max_abs_error=max(difference_errors),amplitude_errors_checked=len(amp_errors),amplitude_sigma_max_abs_error=max(amp_errors),block_sum_max_abs_error=max_add,reference_cases_replayed=len(records),unique_wedge_slopes=11,weighted_sigma_relative_extrema=[min(r['weighted_sigma_relative_min'] for r in records),max(r['weighted_sigma_relative_max'] for r in records)],sklearn_version=sklearn.__version__,status='Arithmetic and saved-data replay pass; unequal-block and conditional-estimator limitations remain',fractions=frac_ranges,covariances=records)
(O/'AUDIT_RESULTS.json').write_text(json.dumps(summary,indent=2));(O/'SOURCE_HASHES.json').write_text(json.dumps(sources,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['fractions','covariances']},indent=2))
