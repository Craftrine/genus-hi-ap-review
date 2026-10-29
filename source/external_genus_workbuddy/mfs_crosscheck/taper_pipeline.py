"""taper 掩膜版 nearest-mesh 管线（复刻生产链，新增平滑 taper 前景切割）。

设计：
- 硬掩膜路径完全复用 old.removal_mask（用于复现存档行为）；
- taper 路径：保留权重 W = sigmoid((|k∥| − cut)/w)，cut = max(0.1, m·k⊥)（wedge 分支）
  或 cut = 0.1（horizontal_only），w = taper_cells × dk（dk 取三轴格波数最小值）；
- 其余（几何、CIC、补零、平滑、测量）与生产逐行相同。
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/LOCAL_USER/Documents/ChatGPT/genus')
from full_hi_pipeline import scan_sim_fixed_cell as old, scan_sim_nearest_mesh as n


def taper_weight(axes, branch, taper_cells=4.0, k_parallel_cut=0.1):
    """平滑保留权重 ∈(0,1]；branch 语义与 old.removal_mask 相同（0=noFG 返回 None）。"""
    if branch == 0:
        return None
    kx, ky, kz = axes
    # 物理格波数：axes 已是物理 k；dk = 各轴非零最小间隔
    dk_list = []
    for k in (kx, ky, kz):
        ks = np.sort(np.abs(np.unique(k)))
        dk_list.append(ks[1] - ks[0] if len(ks) > 1 else ks[0])
    dk_min = min(dk_list)
    w = taper_cells * dk_min
    kz3 = np.abs(kz)[None, None, :]
    low = 0.5*(1+np.tanh((kz3 - k_parallel_cut)/w))            # 横切 taper
    if branch == 1:
        return low.astype(np.float32)
    kp = np.sqrt(kx[:, None]**2 + ky[None, :]**2)[:, :, None]
    cut = np.maximum(k_parallel_cut, old.M_SCAN[branch-2]*kp)
    wgt = 0.5*(1+np.tanh((kz3 - cut)/w))
    return wgt.astype(np.float32)


def filtered_field_taper(spectrum, axes, shape_fft, slices, radius, branch, taper_cells=4.0):
    """old.filtered_field 的 taper 版（逐行对应，仅掩膜换成乘权重）。"""
    work = spectrum.copy()
    for axis, k in enumerate(axes):
        view = [1, 1, 1]; view[axis] = len(k)
        work *= np.exp(-.5*(radius*k)**2).astype(np.float32).reshape(view)
    wgt = taper_weight(axes, branch, taper_cells)
    if wgt is not None:
        work *= wgt
    work[0, 0, 0] = 0
    from scipy.fft import irfftn
    padded = irfftn(work, s=shape_fft, workers=1)
    return np.ascontiguousarray(padded[slices])


def build_field(snap, om=.3111, w=-1.):
    """复刻 run.py worker 的建场段（top-1M 选样文件）。"""
    import json as jm
    from pathlib import Path
    from full_hi_pipeline.particle_first_density import valid_objects
    ROOT = Path('/Users/LOCAL_USER/Documents/ChatGPT/genus')
    src = ROOT/f'outputs/top_hi_1m_weighted_rsd_all13_20260908_v1/snap{snap}_selection.npz'
    z = jm.loads((src.parent/f'snap{snap}.json').read_text())['redshift']
    with np.load(src) as f:
        rec = f['records']
    p, v, wt, _, _ = valid_objects(rec); del rec
    p = old.add_rsd_sim(p, v, z); del v
    alpha = old.ap_scan(z, om, w)
    p, wt, g = n.geometry(p, wt, alpha)
    delta, cons = n.deposit(p, wt, g['grid_shape_scan'], g['spacing_scan']); del p, wt
    spec, axes, shape, slices = n.prepare_spectrum(delta, g['spacing_scan'], 160); del delta
    return dict(spec=spec, axes=axes, shape=shape, slices=slices, geo=g, z=z, alpha=alpha)


def measure_branch(built, branch, radius=20., taper_cells=None, nu=None):
    """taper_cells=None → 硬掩膜（生产行为）；否则 taper。"""
    if nu is None:
        nu = np.linspace(-2.5, 2.5, 41)
    if taper_cells is None:
        field = old.filtered_field(built['spec'], built['axes'], built['shape'], built['slices'], radius, branch)
    else:
        field = filtered_field_taper(built['spec'], built['axes'], built['shape'], built['slices'], radius, branch, taper_cells)
    return n.measure(field, built['geo']['spacing_scan'], nu), field
