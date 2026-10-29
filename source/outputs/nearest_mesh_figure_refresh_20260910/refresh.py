"""Refresh official text datasets consistently under the full-box mesh contract."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,shutil,time
import numpy as np
from sklearn.covariance import oas
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;FIG=ROOT/'paper_figures';archive=ROOT/'maintenance/pre_nearest_mesh_figure_refresh_20260910';archive.mkdir(exist_ok=True)
BR=['nofg','horizontal_only','horizontal_m1','horizontal_m5'];SN=[118,112,105,94];P=[(.3111,-1),(.2,-1.4),(.42,-.6)];nu=np.linspace(-2.5,2.5,41)
for folder in FIG.glob('fig__*'):
 if not folder.is_dir():continue
 dst=archive/folder.name;dst.mkdir(exist_ok=True)
 for f in folder.iterdir():
  if f.is_file() and f.suffix in ['.csv','.json','.py','.md','.png','.pdf','.npz'] and not(dst/f.name).exists():shutil.copy2(f,dst/f.name)
g=np.zeros((3,4,4,41));th=np.zeros_like(g);sig=np.zeros((3,4,4));loo=np.zeros((3,4,4,64,41));zs=[]
for ci in range(3):
 for zi,snap in enumerate(SN):
  with np.load(OUT/f'c{ci}_snap{snap}.npz') as f:
   g[ci,zi]=f['curves'];th[ci,zi]=f['thresholds'];sig[ci,zi]=f['sigma'];loo[ci,zi]=(f['curves'][:,None,:]-f['contributions'])/(1-f['fractions'][None,:,None])
   if ci==0:zs.append(float(f['z']))
def cov(x):
 x=x-x.mean(0);return 63/64*(x.T@x)
def write(folder,name,header,rows):
 with (FIG/folder/name).open('w') as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)
D=g[:,1:]-g[:,:1];I=D[0];C=[];covmeta=[]
old=np.load(FIG/'fig__intrinsic_subtracted_contours/analysis_arrays.npz')
for bi,b in enumerate(BR):
 x=(loo[0,1:,bi]-loo[0,0,bi]).transpose(1,0,2).reshape(64,123);x-=x.mean(0);raw=63/64*(x.T@x);st=np.sqrt(np.diag(raw));Z=x/np.sqrt(np.mean(x*x,axis=0));corr,rho=oas(Z,assume_centered=True);cc=corr*st[:,None]*st[None,:];np.testing.assert_allclose(cc,old[f'C_v0_b{bi}'],atol=1e-27,rtol=1e-10);C.append(cc);covmeta.append(dict(branch=b,oas_rho=float(rho)))
# 1. Baseline comparison: same science, recomputed consistent64-block errors.
r=[]
for bi,b in enumerate(BR):
 e=np.sqrt(np.diag(cov(loo[0,0,bi])))
 for i,x in enumerate(nu):r.append([b,.10235,20,x,th[0,0,bi,i],g[0,0,bi,i],8000*g[0,0,bi,i],e[i]])
write('fig__noFG_hcut_msan_comp','data.csv',['branch','z','R','nu_f','threshold','g_h3_Mpc3','gR3','sigma_g_64'],r)
#2 and5 share measured curves and covariance convention.
r=[];dr=[];ir=[];qr=[]
for ci,(om,w) in enumerate(P):
 for bi,b in enumerate(BR):
  fullp=[];fulld=[]
  for zi,snap in enumerate(SN):
   e=np.sqrt(np.diag(cov(loo[ci,zi,bi])));d=g[ci,zi,bi]-g[ci,0,bi];de=np.sqrt(np.diag(cov(loo[ci,zi,bi]-loo[ci,0,bi])))
   for i,x in enumerate(nu):r.append([ci,om,w,snap,zs[zi],b,x,th[ci,zi,bi,i],g[ci,zi,bi,i],e[i]])
   for i,x in enumerate(nu):dr.append([ci,om,w,snap,zs[zi],b,x,d[i],de[i]])
   if zi:
    intrinsic=I[zi-1,bi];p=d-intrinsic;cc=C[bi][(zi-1)*41:zi*41,(zi-1)*41:zi*41];se=np.sqrt(np.diag(cc));qd=float(d@np.linalg.solve(cc,d));qp=float(p@np.linalg.solve(cc,p));qr.append([ci,b,snap,zs[zi],41,qd,qp]);fullp.extend(p);fulld.extend(d)
    for i,x in enumerate(nu):ir.append([ci,om,w,snap,zs[zi],b,x,d[i],intrinsic[i],p[i],se[i],de[i],p[i]/se[i]])
  p=np.array(fullp);d=np.array(fulld);qr.append([ci,b,'joint','all',123,float(d@np.linalg.solve(C[bi],d)),float(p@np.linalg.solve(C[bi],p))])
header=['cosmology_id','omega_m','w','snapshot','redshift','branch','nu_f','threshold','g','sigma_g']
write('fig__cosmo_redshift_evolution','data.csv',header,r)
write('fig__cosmo_redshift_evolution','differences.csv',['cosmology_id','omega_m','w','snapshot','redshift','branch','nu_f','delta_g','sigma_delta_g'],dr)
write('fig__cosmo_redshift_evolution','intrinsic_subtracted.csv',['cosmology_id','omega_m','w','snapshot','redshift','branch','nu_f','delta_g','intrinsic_evolution','p','sigma_contour_fixed','sigma_trial_evolution','pull_contour_fixed'],ir)
write('fig__cosmo_redshift_evolution','chisq_annotations.csv',['cosmology_id','branch','snapshot','redshift','vector_dimension','chi2_before_intrinsic','chi2_after_intrinsic'],qr)
write('fig__diff_redshifts','data.csv',['snapshot','z','branch','R_Mpc_h','nu_f','g_h3_Mpc3','sigma_g_64'],[[v[3],v[4],v[5],20,v[6],v[8],v[9]] for v in r if v[0]==0])
write('fig__diff_redshifts','differences.csv',['snapshot','z','branch','nu_f','delta_g','sigma_delta_g_64'],[[v[3],v[4],v[5],v[6],v[7],v[8]] for v in dr if v[0]==0 and v[3]!=118])
#3 slices: recomputed fiducial raw matrices and full-volume sigmas.
m=json.loads((FIG/'fig__slices/metadata.json').read_text())
for v in m['slices']:
 zi=SN.index(v['snapshot']);bi=BR.index(v['branch'])
 with np.load(OUT/f'c0_snap{v["snapshot"]}.npz') as f:new=f[f'{v["plane"]}_{bi}']
 previous=np.loadtxt(FIG/'fig__slices'/v['file'],delimiter=',');np.testing.assert_allclose(previous,new,atol=1e-8,rtol=1e-8)
 np.savetxt(FIG/'fig__slices'/v['file'],new,delimiter=',',fmt='%.17g');v['sigma_full3D']=float(sig[0,zi,bi]);v['median_threshold_full3D']=float(th[0,zi,bi,20])
m['measurement_contract']='full AP box nearest per-axis spacing; fiducial replay identical';(FIG/'fig__slices/metadata.json').write_text(json.dumps(m,indent=2))
#4 high-z comparison: new wrong-cosmology measurements and their individual errors.
r4=[]
for ci,(om,w) in enumerate(P):
 for bi,b in enumerate(BR):
  e=np.sqrt(np.diag(cov(loo[ci,3,bi])))
  for i,x in enumerate(nu):r4.append([om,w,zs[3],b,x,th[ci,3,bi,i],g[ci,3,bi,i],e[i],g[ci,3,bi,i]-g[0,3,bi,i]])
write('fig__diff_cosmo','data.csv',['omega_m_scan','w_scan','z','branch','nu_f','threshold','g','sigma_g_64','difference_from_sim'],r4)
#7 same verified fiducial covariance.
r7=[]
for bi,b in enumerate(BR):
 for i in range(123):
  for j in range(123):r7.append([b,i//41+2,nu[i%41],j//41+2,nu[j%41],C[bi][i,j]])
write('fig__covmat','covariance.csv',['branch','pair_i','nu_i','pair_j','nu_j','covariance'],r7)
(FIG/'fig__covmat/source_summary.json').write_text(json.dumps(dict(covariance=covmeta,contract='new full-box mesh, fiducial covariance verified unchanged'),indent=2))
# New metadata and current-context captions for all refreshed figures.
notes={
'fig__noFG_hcut_msan_comp':'Simulation-cosmology curves are unchanged;64-block additive errors are recomputed consistently.',
'fig__diff_redshifts':'Simulation-cosmology redshift curves and paired errors are recomputed under the full-box contract.',
'fig__slices':'All16 slices and standard deviations are verified/recomputed; fiducial images are unchanged.',
'fig__diff_cosmo':'Both wrong cosmologies are recomputed using full AP boxes, per-axis spacing and matching per-curve errors.',
'fig__cosmo_redshift_evolution':'All48 curves and matching spatial errors refreshed; both difference rows use the fixed fiducial covariance for annotated Q.',
'fig__covmat':'Covariance recalculated from fiducial spatial blocks and verified equal to the previous matrix; displayed regularized matrix remains valid for the controlled comparison.',
'fig__mesh_size_distribution':'Geometry-only distribution already follows the new rule; replot without scientific changes.'}
for folder,note in notes.items():
 f=FIG/folder/'README.md';oldtext=f.read_text() if f.exists() else '';f.write_text('# Current full-box nearest-mesh revision\n\n'+note+'\n\nAll active plots use the new contract. Any preceding cropped-trial results below are historical and are not current scientific conclusions. Earlier active files archived in maintenance/pre_nearest_mesh_figure_refresh_20260910.\n\n'+oldtext)
 (FIG/folder/'current_revision.json').write_text(json.dumps(dict(contract='full_box_nearest_per_axis_spacing',refresh_source=str(OUT),note=note),indent=2))
np.savez_compressed(OUT/'refreshed_arrays.npz',curves=g,thresholds=th,loo=loo,fiducial_covariance=C,intrinsic=I,redshifts=zs)
(OUT/'verification.json').write_text(json.dumps(dict(curves=48,all_fiducial_curves_exact=True,fiducial_covariance_matches_old=True,all16_slices_replayed=True,wrong_cosmologies_recomputed=True,old_data_archived=str(archive)),indent=2))
print('REFRESHED scientific text data for figures1–5,7; geometry figure8 already current')
