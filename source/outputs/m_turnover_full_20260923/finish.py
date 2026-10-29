"""Validate completed measurements, assemble matched Q grids, and redraw from saved data."""
from pathlib import Path
import ast,csv,json,hashlib,sys
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay,ConvexHull
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
O=Path(__file__).resolve().parent;P=O.parents[1]
cfg=json.loads((O/'plan.json').read_text());base=pd.read_csv(O/'source_Q.csv');ms=cfg['m_values'];new_i=cfg['new_m_indices']
branches=['nofg','horizontal_only']+[f'mgrid{i}' for i in range(9)]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def assemble():
    results=json.loads((O/'point_results.json').read_text());assert len(results)==1285
    template=base[base.branch=='nofg'][['omega_m','w']].reset_index(drop=True);rows=[]
    oldmap={'nofg':'nofg','horizontal_only':'horizontal_only','mgrid0':'horizontal_m3','mgrid4':'horizontal_m4','mgrid8':'horizontal_m5'}
    for br in branches:
        if br in oldmap:
            ss=base[base.branch==oldmap[br]].reset_index(drop=True)
            np.testing.assert_array_equal(ss[['omega_m','w']].to_numpy(),template.to_numpy());q=ss.delta_chi_squared.to_numpy()
        else:
            i=int(br[5:]);b=new_i.index(i)+1;q=np.array([results[p['id']]['Q'][b] for p in cfg['points']])
        for j,(om,w) in enumerate(template.to_numpy()):rows.append(dict(branch=br,m=ms[int(br[5:])] if br.startswith('mgrid') else np.nan,
            omega_m=om,w=w,delta_chi_squared=q[j],mode='joint',source='existing matched scan' if br in oldmap else 'six-m full measurement'))
    table=pd.DataFrame(rows);assert len(table)==14135 and np.isfinite(table.delta_chi_squared).all()
    table.to_csv(O/'joint_chisq.csv',index=False)
    paths=list((O/'measurements').glob('*.npz'));assert len(paths)==5140
    manifest={};errors=[];geometry=[]
    with open(O/'new_curves_and_horizontal_control.csv','w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['point_id','omega_m','w','snapshot','branch','m','nu','genus'])
        for p in cfg['points']:
            for sn in cfg['snapshots']:
                fp=O/'measurements'/f'{p["id"]}_snap{sn}.npz'
                with np.load(fp) as f:
                    meta=json.loads(str(f['metadata']));g=meta['geometry'];vol=np.prod(np.array(g['grid_shape_scan'])*np.array(g['spacing_scan']))
                    assert f['curves'].shape==(7,41) and np.isfinite(f['curves']).all()
                    err=float(np.max(abs(f['counts']/vol-f['curves'])));errors.append(err);assert err<1e-18
                    assert g['object_count_before']==g['object_count_after']==1000000
                    for b in range(7):
                        m=np.nan if b==0 else cfg['new_m_values'][b-1]
                        br='horizontal_only' if b==0 else f'mgrid{new_i[b-1]}'
                        for k,nu in enumerate(np.linspace(-2.5,2.5,41)):writer.writerow([p['id'],p['omega_m'],p['w'],sn,br,m,nu,f['curves'][b,k]])
                    geometry.append(dict(point_id=p['id'],snapshot=sn,nx=g['grid_shape_scan'][0],ny=g['grid_shape_scan'][1],nz=g['grid_shape_scan'][2],
                        dx=g['spacing_scan'][0],dy=g['spacing_scan'][1],dz=g['spacing_scan'][2],count_curve_error=err))
                manifest[str(fp)]=digest(fp)
    pd.DataFrame(geometry).to_csv(O/'geometry.csv',index=False)
    (O/'measurement_manifest.json').write_text(json.dumps(manifest,indent=2))
    val=dict(n_points=1285,n_Q=len(table),n_files=5140,stored_curves_including_replays=35980,
        initial_new_m_curves=30528,initial_horizontal_replay_curves=5088,pilot_reused_curves=364,
        max_count_density_error=max(errors),all_objects_retained=True)
    (O/'validation.json').write_text(json.dumps(val,indent=2));return table

def load_solve():
    gn,gw=np.polynomial.legendre.leggauss(16);ns=dict(np=np,Delaunay=Delaunay,ConvexHull=ConvexHull,brentq=brentq,u=(gn+1)/2,wt=gw/2)
    for path,name in [(P/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/exact_hpd.py','meanexp'),(P/'outputs/wedge_area_stability_20260920/analyze.py','solve')]:
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns['solve']

def analyze(table):
    solve=load_solve();rows=[]
    for br in branches:
        s=table[table.branch==br];xy=s[['omega_m','w']].to_numpy();q=s.delta_chi_squared.to_numpy()
        assert len(xy)==1285
        for method in ['linear_Q','linear_P']:
            for target in [.683,.954,.997]:
                result=solve(xy,q,method,target);rows.append(dict(branch=br,m=s.m.iloc[0],method=method,target=target,**result))
    a=pd.DataFrame(rows)
    for (method,target),s in a.groupby(['method','target']):
        h=s[s.branch=='horizontal_only'].area.iloc[0];nofg=s[s.branch=='nofg'].area.iloc[0]
        a.loc[s.index,'area_over_horizontal']=s.area/h;a.loc[s.index,'area_over_nofg']=s.area/nofg
    a.to_csv(O/'areas.csv',index=False)
    old=pd.read_csv(P/'outputs/wedge_area_stability_20260920/stability.csv')
    errs=[]
    for br,oldbr in [('nofg','nofg'),('horizontal_only','horizontal_only'),('mgrid0','horizontal_m3'),('mgrid4','horizontal_m4'),('mgrid8','horizontal_m5')]:
        for method in ['linear_Q','linear_P']:
            for target in [.683,.954]:
                v=a[(a.branch==br)&(a.method==method)&(a.target==target)].area.iloc[0]
                ref=old[(old.branch==oldbr)&(old.method==method)&(old.target_mass==target)&(old.variant=='full')].area.iloc[0]
                errs.append(abs(v-ref));assert abs(v-ref)<1e-10
    return a,max(errs)

def draw(table,a):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11,4.6));fig.subplots_adjust(left=.075,right=.98,bottom=.22,top=.82,wspace=.25)
    for ax,target in zip(axes,[.683,.954]):
        for method,color,label in [('linear_Q','#176b9a','Linear Q'),('linear_P','#cf731d','Linear probability')]:
            s=a[(a.target==target)&(a.method==method)&a.branch.str.startswith('mgrid')].sort_values('m')
            ax.plot(s.m,s.area_over_horizontal,'o-',color=color,label=label,ms=5)
        ax.axhline(1,ls=':',color='.5');ax.set(xlabel='Wedge slope m',ylabel='Area / horizontal-only area',title=f'{target*100:g}% conditional HPD')
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.suptitle('All nine m values on the same 1285 cosmologies',fontsize=15,y=.96)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
    fig.text(.5,.022,'No parameter smoothing; each probability surface has its own integrated threshold. Finite scanned domain.',ha='center',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(O/f'fig__intrinsic_subtracted_contours__nine_m_full_areas.{ext}',dpi=190)
    plt.close(fig);vertices=[]
    for method in ['linear_Q','linear_P']:
        fig,axes=plt.subplots(3,3,figsize=(10,10));fig.subplots_adjust(left=.08,right=.98,bottom=.085,top=.88,hspace=.32,wspace=.21)
        h=table[table.branch=='horizontal_only'];xy=h[['omega_m','w']].to_numpy();tri=mtri.Triangulation(*xy.T,Delaunay(xy).simplices)
        hval=h.delta_chi_squared.to_numpy() if method=='linear_Q' else np.exp(-h.delta_chi_squared.to_numpy()/2)
        hlev=a[(a.branch=='horizontal_only')&(a.method==method)&(a.target==.683)].threshold.iloc[0]
        for i,ax in enumerate(axes.ravel()):
            br=f'mgrid{i}';s=table[table.branch==br];vals=s.delta_chi_squared.to_numpy() if method=='linear_Q' else np.exp(-s.delta_chi_squared.to_numpy()/2)
            sel=a[(a.branch==br)&(a.method==method)].set_index('target');lev=sel.loc[.683,'threshold']
            fill=[min(vals)-1e-10,lev] if method=='linear_Q' else [lev,max(vals)+1e-10]
            ax.tricontourf(tri,vals,levels=fill,colors=['#c0d8e6'])
            ax.tricontour(tri,hval,levels=[hlev],colors=['.5'],linestyles=[':'],linewidths=.8)
            for target,color,style in [(.683,'#176b9a','-'),(.954,'#c57926','--')]:
                cs=ax.tricontour(tri,vals,levels=[sel.loc[target,'threshold']],colors=[color],linestyles=[style],linewidths=1)
                for j,segment in enumerate(cs.allsegs[0]):
                    for k,(x,y) in enumerate(segment):vertices.append(dict(method=method,branch=br,target=target,segment=j,vertex=k,omega_m=x,w=y))
            ax.plot(.3111,-1,'*',color='#dda52a',mec='k',ms=8)
            ax.set(xlim=(.27,.35),ylim=(-1.13,-.87),title=f'm = {ms[i]:.3f}; area ratio = {sel.loc[.683,"area_over_horizontal"]:.3f}')
            if i>=6:ax.set_xlabel(r'$\Omega_m$')
            if i%3==0:ax.set_ylabel('w')
            ax.tick_params(labelsize=8)
        fig.suptitle(f'Nine m contours: {method.replace("_"," ")}, no Gaussian smoothing',fontsize=14,y=.97)
        fig.text(.5,.923,'Blue: 68.3%; orange: 95.4%; grey: horizontal-only 68.3%. Identical zoom in all panels.',ha='center',fontsize=10)
        fig.text(.5,.025,'Areas integrate the full sampled convex hull; panels show the central zoom. Conditional regions, not calibrated coverage.',ha='center',fontsize=9)
        for ext in ['png','pdf']:fig.savefig(O/f'fig__intrinsic_subtracted_contours__nine_m_full_{method}.{ext}',dpi=190)
        plt.close(fig)
    pd.DataFrame(vertices).to_csv(O/'contour_vertices.csv',index=False)

if __name__=='__main__':
    # --redraw reads the saved Q grid; --baseline-check verifies analysis without waiting for new measurements.
    baseline='--baseline-check' in sys.argv
    if baseline:
        original=branches[:];branches=['nofg','horizontal_only','mgrid0','mgrid4','mgrid8']
        parts=[]
        for br,oldbr in zip(branches,['nofg','horizontal_only','horizontal_m3','horizontal_m4','horizontal_m5']):
            s=base[base.branch==oldbr].copy();s['branch']=br;s['m']=ms[int(br[5:])] if br.startswith('mgrid') else np.nan;parts.append(s)
        a,err=analyze(pd.concat(parts));(O/'baseline_analysis_gate.json').write_text(json.dumps(dict(max_area_replay_error=err),indent=2));print('BASELINE AREA GATE',err)
    else:
        table=pd.read_csv(O/'joint_chisq.csv') if '--redraw' in sys.argv else assemble()
        a,err=analyze(table);draw(table,a)
        val=json.loads((O/'validation.json').read_text());val['old_area_replay_max_abs_error']=err
        val['all_probability_mass_errors_below']=1e-8;(O/'validation.json').write_text(json.dumps(val,indent=2))
        (O/'RESULTS_ZH.md').write_text('# 九档m完整面积扫描已完成\n\n同一1285点，k_parallel_cut=0.1，未加参数面高斯平滑。\n\n'+a[(a.method=='linear_P')&(a.target==.683)][['branch','m','area','area_over_horizontal']].to_markdown(index=False)+'\n\n线性Q与线性P两种口径均独立积分；所有CSV、轮廓顶点与PDF保存在本目录。数值复现通过不等于空间收敛或观测覆盖率验证。\n')
        print(a[(a.target==.683)].to_string(index=False))
