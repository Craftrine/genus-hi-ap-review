"""Matched 84-curve comparison; no changes to production measurements."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import json,time,csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.fft import irfftn
from full_hi_pipeline import scan_sim_fixed_cell as s
from full_hi_pipeline.scan_sim_spectral_moments import spectral_moments,derivative_axes,gaussian_amplitude
from full_hi_pipeline.particle_first_density import valid_objects
from full_hi_pipeline.genus_grid import _atomic_json,_atomic_npz,sha256_file
from run_scan_sim_fixed_cell import ROOT,SOURCE,SNAPS,SCALES

BASE=ROOT/'outputs/scan_sim_fixed_cell_20260909_v1'
OUT=BASE/'spectral_topology_comparison';OUT.mkdir(exist_ok=True)


def mean_product(a,b):
    return float(np.einsum('ijk,ijk->',a,b,dtype=np.float64,optimize=False)/a.size)


def skew_kurt(field):
    mean=float(field.mean(dtype=np.float64));flat=field.ravel();mom=np.zeros(3)
    for start in range(0,len(flat),1000000):
        v=flat[start:start+1000000].astype(float)-mean;v2=v*v
        mom += [v2.sum(),(v2*v).sum(),(v2*v2).sum()]
    mom/=len(flat)
    return float(mom[1]/mom[0]**1.5),float(mom[2]/mom[0]**2-3)


def snapshot(snap):
    started=time.time();tag=f'snap{snap}_om0.3111000_w-1.0000000_d4_p160'
    directories={branch:BASE/('four_branch_campaign' if branch in [0,1,2,6] else 'remaining_three_campaign')/'points'/tag for branch in range(7)}
    meta=json.loads((directories[0]/'summary.json').read_text());source=SOURCE/f'snap{snap}_selection.npz'
    assert sha256_file(source)==meta['config']['source_sha256']
    assert sha256_file(Path(s.__file__))==meta['config']['core_sha256']
    with np.load(source) as f:records=f['records']
    pos,vel,weights,_,_=valid_objects(records);del records
    rsd=s.add_rsd_sim(pos,vel,meta['config']['z'])
    positions_scan,weights_scan,geo=s.crop_geometry(rsd,weights,np.ones(3),4.)
    delta,conservation=s.deposit(positions_scan,weights_scan,geo['grid_shape_scan'],4.)
    del pos,vel,weights,rsd,positions_scan,weights_scan
    spectrum,axes,shape,slices=s.prepare_spectrum(delta,4.,160);del delta
    daxes=derivative_axes(shape,4.);nu=np.linspace(-2.5,2.5,41);rows=[];arrays={}
    for ri,radius in enumerate(SCALES):
        for branch,label in enumerate(s.LABELS):
            _atomic_json(OUT/f'snap{snap}_status.json',dict(status='RUNNING',radius=radius,branch=label,completed=len(rows),target=21))
            # Same multiplication order and float32 windows as production filtered_field.
            work=spectrum.copy()
            for axis,k in enumerate(axes):
                view=[1,1,1];view[axis]=len(k)
                work*=np.exp(-.5*(radius*k)**2).astype(np.float32).reshape(view)
            mask=s.removal_mask(axes,branch)
            if mask is not None:work*=~mask
            del mask;work[0,0,0]=0
            moments=spectral_moments(work,shape,4.)
            padded=irfftn(work,s=shape,workers=1)
            field=np.ascontiguousarray(padded[slices])
            variance_real=float(padded.var(dtype=np.float64));del padded
            variance_error=abs(variance_real/moments['variance']-1)
            if variance_error>3e-6:raise ArithmeticError('Variance Parseval failed')
            measured=s.measure(field,4.,nu)
            stored=directories[branch]/f'R{radius:g}_{label}.npz'
            with np.load(stored) as f:
                np.testing.assert_array_equal(measured['curve'],f['curve'])
                np.testing.assert_array_equal(measured['thresholds'],f['thresholds'])
                np.testing.assert_array_equal(measured['amplitude_gaussian'],f['amplitude_gaussian'])
            retained_variance=float(field.var(dtype=np.float64));skew,kurt=skew_kurt(field)
            del field
            derivatives=[]
            for axis,k in enumerate(daxes):
                view=[1,1,1];view[axis]=len(k)
                temp=work.copy();temp*=(1j*k).astype(np.complex64).reshape(view)
                derivatives.append(irfftn(temp,s=shape,workers=1));del temp
            global_gradient=np.array([[mean_product(a,b) for b in derivatives] for a in derivatives])
            gradient_error=float(np.max(abs(global_gradient-moments['gradient_covariance']))/np.max(np.diag(moments['gradient_covariance'])))
            if gradient_error>3e-6:raise ArithmeticError('Gradient Parseval failed')
            cropped=[a[slices] for a in derivatives]
            means=np.array([a.mean(dtype=np.float64) for a in cropped])
            gradient_retained=np.array([[mean_product(a,b) for b in cropped] for a in cropped])-np.outer(means,means)
            local=gaussian_amplitude(retained_variance,gradient_retained)
            del cropped,derivatives,work
            direct=measured['amplitude_gaussian']*radius**3
            key=f'R{radius:g}_{label}'
            arrays[key+'_M_padded']=moments['normalized_gradient_covariance']
            arrays[key+'_M_retained']=local['normalized_gradient_covariance']
            arrays[key+'_direct_curve']=measured['curve']
            arrays[key+'_predicted_gaussian_curve']=moments['amplitude']*(1-nu*nu)*np.exp(-nu*nu/2)
            row=dict(snapshot=snap,z=meta['config']['z'],R=radius,branch=label,
                topology_gaussian_AR3=direct,topology_hermite_AR3=measured['amplitude_hermite']*radius**3,
                spectral_padded_AR3=moments['amplitude']*radius**3,
                gradient_retained_AR3=local['amplitude']*radius**3,
                isotropic_trace_AR3=moments['isotropic_trace_amplitude']*radius**3,
                spectral_vs_topology_percent=100*(moments['amplitude']*radius**3/direct-1),
                retained_vs_topology_percent=100*(local['amplitude']*radius**3/direct-1),
                variance_padded=moments['variance'],variance_retained=retained_variance,
                nyquist_power_fraction=moments['nyquist_power_fraction'],variance_parseval_relative_error=variance_error,
                gradient_parseval_relative_error=gradient_error,skewness_retained=skew,excess_kurtosis_retained=kurt,
                production_exact_replay=True,production_sha256=sha256_file(stored))
            rows.append(row);print(snap,radius,label,'spectral/direct %',row['spectral_vs_topology_percent'],flush=True)
    _atomic_npz(OUT/f'snap{snap}_arrays.npz',nu=nu,**arrays)
    _atomic_json(OUT/f'snap{snap}_summary.json',dict(rows=rows,source_sha256=sha256_file(source),conservation=conservation,
        fft_shape_scan=list(shape),retained_shape_scan=geo['grid_shape_scan'],elapsed_seconds=time.time()-started))
    _atomic_json(OUT/f'snap{snap}_status.json',dict(status='COMPLETE',completed=21,target=21))
    return rows


def report(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=sorted(rows,key=lambda r:(SNAPS.index(r['snapshot']),SCALES.index(r['R']),s.LABELS.index(r['branch'])))
    with (OUT/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    x=np.arange(7);names=['noFG','H only']+[f'H+m{m:.3f}' for m in s.M_SCAN]
    for ax,radius in zip(axes,SCALES):
        selected=[r for r in rows if r['snapshot']==118 and r['R']==radius]
        for key,label,color,style in [('topology_gaussian_AR3','Direct Euler + Gaussian fit','#222222','-'),
            ('spectral_padded_AR3','FFT moments: padded domain','#c43b38','--'),
            ('gradient_retained_AR3','Spectral gradients: retained region','#2578b0',':')]:
            ax.plot(x,[r[key] for r in selected],style+'o',color=color,label=label,ms=4)
        ax.set(title=f'R={radius:g} Mpc/h',ylabel=r'$A R^3$');ax.set_xticks(x,names,rotation=45,ha='right');ax.grid(alpha=.18)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('z=0.10235, scan=sim: realized topology vs anisotropic Gaussian moment predictions',fontsize=13)
    fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(OUT/'lowz_amplitude_comparison.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,radius in zip(axes,SCALES):
        for bi,label in enumerate(s.LABELS):
            selected=[r for r in rows if r['R']==radius and r['branch']==label]
            ax.plot([r['z'] for r in selected],[r['spectral_vs_topology_percent'] for r in selected],'o-',label=names[bi],ms=4)
        ax.set(title=f'R={radius:g} Mpc/h',xlabel='Redshift',ylabel='100 (A_FFT / A_topology - 1) [%]');ax.axhline(0,color='gray',lw=.8);ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'redshift_fractional_difference.png',dpi=165);plt.close(fig)
    maximum=lambda key:max(r[key] for r in rows)
    summary=dict(rows=rows,curves=84,all_production_replays_exact=all(r['production_exact_replay'] for r in rows),
        max_variance_parseval_error=maximum('variance_parseval_relative_error'),max_gradient_parseval_error=maximum('gradient_parseval_relative_error'),
        max_nyquist_power_fraction=maximum('nyquist_power_fraction'))
    _atomic_json(OUT/'summary.json',summary)
    lines=['# FFT谱矩预测与直接genus振幅：84条同场比较','',
        '四红移、三个尺度、七前景，均为scan=sim。全部生产曲线、阈值与Gaussian振幅逐元素复现。旧结果未改动。',
        '', '主谱预测为补零FFT域的三维模式矩M=S/sigma0²，A=sqrt(det M)/(4pi²)，保留所有各向异性分量。',
        '另用同一谱导数在保留区域计算经验梯度矩作为窗口诊断。两者均为Gaussian二阶矩预测，不是直接计算该非Gaussian有限区域的拓扑。',
        '', '## 最低红移 z=0.10235','',
        '|R|分支|直接Gaussian AR³|FFT谱矩 AR³|谱矩相对直接差异|保留区域梯度预测 AR³|',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['snapshot']==118:lines.append(f"|{r['R']:g}|{r['branch']}|{r['topology_gaussian_AR3']:.7f}|{r['spectral_padded_AR3']:.7f}|{r['spectral_vs_topology_percent']:+.2f}%|{r['gradient_retained_AR3']:.7f}|")
    lines+=['',f"方差Parseval最大相对误差={summary['max_variance_parseval_error']:.3g}；梯度协方差Parseval最大归一误差={summary['max_gradient_parseval_error']:.3g}；Nyquist平面最大功率占比={summary['max_nyquist_power_fraction']:.3g}。",'',
        '未做球平均分箱或shot-noise单边扣除，未重复平滑/切割。FFT核、半谱倍数、梯度符号通过解析余弦场、完整复FFT和Parseval独立检验。',
        '差异中可能包含非Gaussian/相位、有限样本、窗口边界与离散拓扑效应；Parseval通过不能证明Gaussian近似对真实HI成立。保留区域梯度诊断不是严格窗口修正。',
        '', '![最低红移比较](lowz_amplitude_comparison.png)','', '![红移差异](redshift_fractional_difference.png)',
        '', '[完整CSV](comparison.csv) · [全部数值与验证](summary.json)',
        '', '依据：[计算约定](../../../SPECTRAL_COMPARISON_CONTRACT_20260909.md) · [原始公式登记](../../../GENUS_SPECTRAL_AMPLITUDE_NOTE_20260909.md)']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    started=time.time();_atomic_json(OUT/'status.json',dict(status='RUNNING',target_curves=84))
    try:
        rows=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(snapshot,snap) for snap in SNAPS]):rows.extend(future.result())
        report(rows);_atomic_json(OUT/'status.json',dict(status='COMPLETE',curves=84,elapsed_seconds=time.time()-started))
    except Exception as exc:_atomic_json(OUT/'status.json',dict(status='FAILED',error=repr(exc)));raise
