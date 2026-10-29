"""Apply the established presentation smoothing to saved parameter statistics only.
No field construction, FFT, genus or covariance measurement. Archived inputs immutable.
"""
from pathlib import Path
import hashlib,json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path as MP
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter,binary_erosion
O=Path(__file__).resolve().parent;R=O.parents[1];SRC=R/'outputs/revision_v093_20260930_stage2b_fixes/tables';T=O/'tables'
FILES=['four_cases_nodal_Q.csv','eleven_wedge_nodal_Q.csv','nodes.csv','main_triangles.csv','publication_hull.csv','canonical_raster_areas.csv','publication_contour_vertices.csv','publication_smoothing_thresholds.csv','branch_map.csv','canonical_raster_stability.csv']
manifest={f:hashlib.sha256((SRC/f).read_bytes()).hexdigest() for f in FILES}
for f in FILES:shutil.copy2(SRC/f,T/f)
def read(n):return pd.read_csv(T/n,float_precision='round_trip')
xy=read('nodes.csv')[['omega_m','w']].to_numpy();tri=read('main_triangles.csv')[['node0','node1','node2']].to_numpy(int);hull=read('publication_hull.csv')[['omega_m','w']].to_numpy();d=read('four_cases_nodal_Q.csv');orig=read('canonical_raster_areas.csv');eleven=read('eleven_wedge_nodal_Q.csv')
x=np.linspace(.195,.425,800);y=np.linspace(-1.325,-.670,800);X,Y=np.meshgrid(x,y);points=np.c_[X.ravel(),Y.ravel()];inside=MP(hull).contains_points(points).reshape(X.shape);edge=inside&~binary_erosion(inside);nn=cKDTree(xy).query(points)[1].reshape(X.shape);pix=(x[1]-x[0])*(y[1]-y[0]);mt=mtri.Triangulation(*xy.T,triangles=tri);sx=.0068836045056319675;sy=.01960330848343017
rows=[];stats=[];checks=[];mapping={'nofg':'nofg','horizontal_only':'horizontal_only','horizontal_m1':'m01','horizontal_m5':'m11'}
def export(cs,branch,mode,target,kind):
 for j,(seg,codes) in enumerate(zip(cs.allsegs[0],cs.allkinds[0])):
  if len(seg)==0:continue
  for k,((xx,yy),code) in enumerate(zip(seg,codes)):
   rows.append(dict(branch=branch,mode=mode,target=target,kind=kind,component=j,vertex=k,omega_m=xx,w=yy,code=int(code)))
for (b,mode),s in d.groupby(['branch','mode'],sort=False):
 dist,idx=cKDTree(s[['omega_m','w']].to_numpy()).query(xy)
 assert len(set(idx))==len(xy) and dist.max()<1e-14,(b,mode,'node correspondence')
 checks.append(dict(branch=b,mode=mode,node_coordinate_max_difference=float(dist.max()),note='one-to-one match to canonical coordinates; archived CSV roundoff only'))
 q=s.delta_chi_squared.to_numpy()[idx];p=np.exp(-q/2)
 if mode=='joint':
  eq=eleven[eleven.branch==mapping[b]].Q.to_numpy();assert np.max(abs(q-eq))<1e-9,(b,'Q mismatch')
 raw=mtri.LinearTriInterpolator(mt,p)(X,Y).filled(np.nan);raw=np.where(np.isnan(raw),p[nn],raw)
 smooth=gaussian_filter(raw,(sy/(y[1]-y[0]),sx/(x[1]-x[0])),mode='reflect');vals=np.sort(smooth[inside])[::-1];cum=np.cumsum(vals);masked=np.ma.masked_where(~inside,smooth)
 for target in [.683,.954,.997]:
  cutoff=vals[np.searchsorted(cum,target*cum[-1])];selected=inside&(smooth>=cutoff);area=float(selected.sum()*pix)
  stats.append(dict(branch=b,mode=mode,target=target,threshold=cutoff,area=area,achieved_weight_fraction=float(smooth[selected].sum()/cum[-1]),touches_hull=bool(np.any(selected&edge)),sigma_omega_m=sx,sigma_w=sy,grid=800))
  if mode=='joint':
   saved=orig[(orig.branch==mapping[b])&(orig.variant=='gaussian')&(orig.grid==800)&np.isclose(orig.target,target)].area.iloc[0];checks.append(dict(branch=b,target=target,area=area,archived_area=saved,absolute_error=abs(area-saved)));assert abs(area-saved)<1e-12,(b,target,area,saved)
  f,ax=plt.subplots();cs=ax.contour(X,Y,masked,levels=[cutoff]);export(cs,b,mode,target,'line')
  if target==.683:
   cs=ax.contourf(X,Y,masked,levels=[cutoff,float(smooth.max())+1]);export(cs,b,mode,target,'fill')
  plt.close(f)
 print('Prepared',b,mode,flush=True)
pd.DataFrame(rows).to_csv(T/'four_cases_smoothed_vertices.csv',index=False,float_format='%.17g');pd.DataFrame(stats).to_csv(T/'four_cases_smoothed_stats.csv',index=False,float_format='%.17g')
# All joint areas retained. Only the area-trend display excludes the last slope.
a=orig[(orig.grid==800)&(orig.variant=='gaussian')].copy();a['included_in_area_trend']=a.branch.str.startswith('m')&(a.m<1.65);a.to_csv(T/'joint_smoothed_areas_all.csv',index=False,float_format='%.17g');a[a.included_in_area_trend].to_csv(T/'area_trend_display.csv',index=False,float_format='%.17g')
cv=read('publication_contour_vertices.csv');cv=cv[(cv.family=='eleven')&(cv.variant=='gaussian')];cv.to_csv(T/'eleven_smoothed_vertices.csv',index=False,float_format='%.17g')
assert manifest=={f:hashlib.sha256((SRC/f).read_bytes()).hexdigest() for f in FILES}
(O/'VALIDATION.json').write_text(json.dumps({'saved_parameter_statistics_only':True,'new_field_measurements':False,'new_genus_measurements':False,'new_covariance_estimation':False,'postprocessing_new':'12 single-pair smoothed parameter surfaces; 4 joint replays for verification','max_joint_area_replay_error':max(c['absolute_error'] for c in checks if 'absolute_error' in c),'checks':checks,'input_sha256':manifest,'kernel':{'sigma_omega_m':sx,'sigma_w':sy,'source':'historical 2.5 times median nearest-neighbour spacing in unit rectangle','independently_calibrated':False},'area_display_omission':'m=1.692183 retained in underlying results, overview contours and example panels; omitted only from trend display as requested'},indent=2))
