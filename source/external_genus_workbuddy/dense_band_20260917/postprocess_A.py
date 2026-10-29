"""A 合同（nearest+hard）后处理：233 存档点 + 1052 新点 → ΔQ → contour。

ΔQ 定义与官方完全一致（extent_mesh 口径）：
  p_pair(θ) = [g_zj(θ)−g_z1(θ)] − intrinsic[pair, branch]   （41 维）
  joint p = concat(p_z2z1, p_z3z1, p_z4z1)                    （123 维）
  ΔQ = pᵀ C_v0_b⁻¹ p（Cholesky）；per-pair 用 C 的 41×41 对角块（高斯边缘精确）
HPD：三角剖分线性插值 + exact_hpd.meanexp 解析积分（与官方同法）。
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import Delaunay

PROJ = Path('/Users/LOCAL_USER/Documents/ChatGPT/genus')
OUT = Path('/Users/LOCAL_USER/Documents/ChatGPT/genus_workbuddy/dense_band_20260917')
sys.path.insert(0, str(PROJ/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh'))
from exact_hpd import meanexp

BR_LABELS = ['nofg', 'horizontal_only', 'horizontal_m1', 'horizontal_m5']
BR_IDX = [0, 1, 2, 6]          # npz 内 7 分支中的索引
MODES = ['z2-z1', 'z3-z1', 'z4-z1', 'joint']
TARGETS = [0.683, 0.954, 0.997]

arr = np.load(PROJ/'paper_figures/fig__intrinsic_subtracted_contours/analysis_arrays.npz')
INTR = arr['intrinsic']                    # (3 pair, 4 branch, 41)
CHOL = {b: cho_factor(arr[f'C_v0_b{b}'], lower=True) for b in range(4)}
BLOCK = {b: [cho_factor(arr[f'C_v0_b{b}'][41*k:41*(k+1), 41*k:41*(k+1)], lower=True)
             for k in range(3)] for b in range(4)}

def deltaQ(curves74x41, bi):
    """curves: (7分支,4红移,41)；bi: BR_LABELS 索引。返回 4 个 mode 的 ΔQ。"""
    c = curves74x41[BR_IDX[bi]]            # (4,41)
    pps = [(c[k+1]-c[0]) - INTR[k, bi] for k in range(3)]
    out = []
    for k in range(3):
        out.append(float(pps[k] @ cho_solve(BLOCK[bi][k], pps[k])))
    pj = np.concatenate(pps)
    out.append(float(pj @ cho_solve(CHOL[bi], pj)))
    return out

# ---- 新点 ----
rows = []
files = sorted((OUT/'points_A').glob('om*.npz'))
for fp in files:
    with np.load(fp) as z:
        om, w = float(z['omega_m']), float(z['w'])
        curves = z['curves']
    for bi in range(4):
        dq = deltaQ(curves, bi)
        for mi, md in enumerate(MODES):
            rows.append(dict(branch=BR_LABELS[bi], mode=md, omega_m=om, w=w,
                             delta_chi_squared=dq[mi]))
new = pd.DataFrame(rows)
new['variant'] = 'intrinsic_subtracted'
new['chi_squared'] = new.delta_chi_squared   # 新点上 fiducial 差即全量（模板已扣）
print(f'新点 {len(files)}，行 {len(new)}')

# ---- 旧 233 点（官方 ΔQ）----
old = pd.read_csv(PROJ/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/before_after_chisq.csv')
old = old[old.variant == 'intrinsic_subtracted'][['branch','mode','omega_m','w','chi_squared','delta_chi_squared','variant']]

# ---- 门 G2：重叠区连续性——新点与最近旧点的 ΔQ 差 ----
from scipy.spatial import cKDTree
scale = np.array([0.22, 0.8])
o_pts = old[['omega_m','w']].drop_duplicates().to_numpy()
tree = cKDTree((o_pts-[0.2,-1.4])/scale)
jnew = new[(new.branch=='nofg')&(new['mode']=='joint')]
d, idx = tree.query((jnew[['omega_m','w']].to_numpy()-[0.2,-1.4])/scale)
ref = old[(old.branch=='nofg')&(old['mode']=='joint')].set_index(['omega_m','w'])
diffs = []
for (om, w, dq), ii in zip(jnew[['omega_m','w','delta_chi_squared']].to_numpy(), idx):
    key = tuple(o_pts[ii]);  dq_old = ref.loc[key, 'delta_chi_squared']
    diffs.append(abs(dq - dq_old))
diffs = np.array(diffs)
print(f'G2 连续性（新点 vs 最近旧点 joint noFG |ΔΔQ|）：中位 {np.median(diffs):.2f}，95% {np.percentile(diffs,95):.2f}')

merged = pd.concat([old, new[old.columns]], ignore_index=True)
merged.to_csv(OUT/'chisq_dense_A.csv', index=False)
print(f'合并点集 {merged[["omega_m","w"]].drop_duplicates().shape[0]} 点 → chisq_dense_A.csv')

# ---- HPD（官方 exact_hpd 向量化方法，逐行移植）----
from scipy.optimize import brentq
from scipy.spatial import ConvexHull

def hpd_levels_tri(pts, dq):
    xy = np.asarray(pts); q = np.asarray(dq, float)
    tr = Delaunay(xy).simplices
    v = xy[tr]
    area = abs((v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1])-(v[:,1,1]-v[:,0,1])*(v[:,2,0]-v[:,0,0]))/2
    assert abs(area.sum()-ConvexHull(xy).volume) < 1e-9*ConvexHull(xy).volume
    vals = np.sort(q[tr], axis=1); a, b, c = vals.T
    full = area*meanexp(vals); total = full.sum()
    def mass(t):
        out = np.where(c <= t, full, 0.)
        lo = (a < t) & (t < b)
        if np.any(lo):
            frac = (t-a[lo])**2/((b[lo]-a[lo])*(c[lo]-a[lo]))
            out[lo] = area[lo]*frac*meanexp(np.column_stack([a[lo], np.full(lo.sum(), t), np.full(lo.sum(), t)]))
        hi = (b <= t) & (t < c) & (a < t)
        if np.any(hi):
            frac = (c[hi]-t)**2/((c[hi]-a[hi])*(c[hi]-b[hi]))
            out[hi] = full[hi]-area[hi]*frac*meanexp(np.column_stack([np.full(hi.sum(), t), np.full(hi.sum(), t), c[hi]]))
        return out.sum()/total
    lv = {}
    for tg in TARGETS:
        lv[tg] = brentq(lambda t: mass(t)-tg, 0.0, q.max())
    return lv

# 门 G3：先用官方 233 点复算，必须与 integrated_hpd_levels.csv 一致
off = pd.read_csv(PROJ/'paper_figures/fig__intrinsic_subtracted_contours/extent_mesh/integrated_hpd_levels.csv')
off = off[off.variant == 'intrinsic_subtracted']
g3ok = True
for br in BR_LABELS:
    sub = old[(old.branch == br) & (old['mode'] == 'joint')]
    pts = sub[['omega_m', 'w']].to_numpy(); dq = sub.delta_chi_squared.to_numpy()
    lv = hpd_levels_tri(pts, dq)
    for t in TARGETS:
        o = float(off[(off.branch == br) & (off['mode'] == 'joint') & (off.target_mass == t)].deltaQ_level.iloc[0])
        rel = abs(lv[t]-o)/max(o, 1e-12)
        if rel > 1e-6:
            g3ok = False
            print(f'G3 偏差 {br} {t}: mine {lv[t]:.6f} vs official {o:.6f} (rel {rel:.1e})')
print('G3 ' + ('通过：我的 HPD 复算与官方逐位一致' if g3ok else '有偏差，见上'))

levels_all = {}
for bi, br in enumerate(BR_LABELS):
    sub = merged[(merged.branch == br) & (merged['mode'] == 'joint')]
    pts = sub[['omega_m', 'w']].to_numpy(); dq = sub.delta_chi_squared.to_numpy()
    lv = hpd_levels_tri(pts, dq)
    levels_all[br] = lv
    print(f'{br} joint（合并 1285 点）HPD 层级: ' + ', '.join(f'{t}={lv[t]:.2f}' for t in TARGETS))
json.dump({b: {str(t): float(v) for t, v in lv.items()} for b, lv in levels_all.items()},
          (OUT/'hpd_levels_dense_A.json').open('w'), indent=1)
print('已存 hpd_levels_dense_A.json')
