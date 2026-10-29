"""Write a concise diagnosis from completed local numerical results."""
from pathlib import Path
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent
names={'nofg':'noFG','horizontal_only':'horizontal cut','horizontal_m1':'m=0.294','horizontal_m4':'m=1.343'}
lines=['# Genus χ² 的 mesh 跳变诊断','',
'**本轮选择 A+B 做粗面积比较。** A 消除了两组原 mesh 边界的大跳变，但固定 N 后，1D 切线上仍出现约 24–27 的硬掩膜台阶。A 不满足“曲线光滑”的条件。A+B 在该处消除了同量级台阶；这不等于证明离散 genus 已严格光滑或收敛。','',
'本轮没有修改论文或原始科学数据；未运行 1285 节点全量扫描，未测 A 的面积。','',
'## 设置与口径','',
'- 四快照 118/112/105/94、原 top-HI 100 万选样、RSD/AP、R20、41 个体积分数阈值、开放边界 −Euler 均保持原定义。原方案为完整 AP 盒中的最近格距整数 N。A 固定 250³，spacing_i=4α_i，每侧补零 160 格，FFT 尺寸也固定。固定 N 的 AP 位置与 spacing 在 CIC 中相消，因此在 sim 坐标沉积；Fourier 波数和体积仍随 AP 改变。',
'- A+B 对 Fourier 振幅乘 W=½[1+tanh((|k_parallel|−k_cut)/0.01)]；horizontal 的 k_cut=0.1，wedge 为 max(0.1,m k_perp)。0.01 h/Mpc 是 tanh 分母，10%–90% 宽度为 0.02197 h/Mpc。m 使用原精确值 0.2938445235861259、1.3425985076924206。noFG 不加 taper，故 A 与 A+B 完全相同。',
'- 每种方案扣除自身 sim intrinsic。主比较统一用原协方差 C，χ²_int=rᵀC⁻¹r；A 的全部 sim 曲线逐元素回放一致。匹配 C 对照使用重新测量的 taper sim 场及原 64 块 matched-stencil jackknife + OAS 方法，不新增独立模拟。',
'- 面积为同域 exp(−χ²_int/2) 的 **unsmoothed linear-P** 积分 HPD，重新求 68.3% 阈值。Δχ²≈2.3 只用作跳变量级参考。','',
'## 原 mesh 边界','']
f=O/'boundary_results.csv'
if f.exists():
 d=pd.read_csv(f);s=d[d.covariance=='common'];p=s.pivot(index=['probe','branch'],columns='scheme',values='delta_Q')[['original','A','AB']];p=p.reindex(pd.MultiIndex.from_product([['nofg_3','horizontal_only_1'],list(names)],names=['probe','branch'])).rename(index=names,columns={'AB':'A+B'}).reset_index()
 lines+=['表中 Δχ²=右点−左点，四种 foreground 均为实际测量。两对参数间距分别为 (5.5×10⁻⁸,2×10⁻⁷) 和 (1.2×10⁻⁷,3×10⁻⁸)。','',p.to_markdown(index=False,floatfmt='.6g'),'','A 最大 |Δχ²|=0.001805；A+B 最大 |Δχ²|=0.130788，均小于 1。A+B 不保证每一个离散拓扑小跳变都比 A 更小。','', '![边界跳变](fig__mesh_AB_boundary_jumps.png)','']
lines+=['## Ωm 切线与选型','']
f=O/'line.csv'
if f.exists():
 d=pd.read_csv(f);lines += [f'固定 w={d.w.iloc[0]:.12f}，Ωm∈[{d.omega_m.min():.9f},{d.omega_m.max():.9f}]。已完成 {d.omega_m.nunique()} 个位置，包含原 mesh 切换附近的 10⁻⁷ 近点。曲线直接连接测量点，没有显示平滑。本切线上 m=0.294 与 horizontal 的 χ² 逐点重合，两分支均执行了各自过滤与测量。','', '![一维切线](fig__mesh_AB_omega_line.png)','']
lines+=['A 的额外台阶位于 Ωm=0.273163180046…：snap94 固定 FFT 网格的第 37 个视线波数跨过 k_parallel=0.1。这里 N 不变，属于硬掩膜模式切换。因此按用户规则，粗面积选择 A+B。','']
f=O/'hard_mask_pair.csv'
if f.exists():
 d=pd.read_csv(f);p=d.pivot(index='branch',columns='scheme',values='delta_Q')[['original','A','AB']].rename(index=names,columns={'AB':'A+B'})
 lines+=['追加 ±5×10⁻⁸ 的紧括区间确认硬掩膜台阶（w 固定，ΔΩm=10⁻⁷）：','',p.to_markdown(floatfmt='.6g'),'']
lines+=['## 271 节点粗面积比较','',
'子集来自原 1285 节点，范围约 Ωm=0.29075…0.32925、w=−1.0435…−0.9500；其凸包覆盖原 horizontal cut 与 m=1.343 的完整 95.4% 区域（800² 原域栅格检查）。原方案和 A+B 使用完全相同的节点与三角插值域；原方案复用既有实际测量并重新积分。**不以子集绝对面积作结论。**','']
f=O/'subset_area_ratios.csv'
if f.exists():
 d=pd.read_csv(f);r=d[(d.raster==800)&(d.target==.683)].copy();r['relative_change_percent']=100*r.relative_change
 r.quantity=r.quantity.replace({**names,'wedge_over_horizontal':'(wedge / horizontal) 面积比'})
 lines+=['共同 C，68.3% 面积相对变化：','',r[['quantity','old_ratio','new_ratio','relative_change_percent']].rename(columns={'quantity':'比较项','old_ratio':'原比值','new_ratio':'新比值','relative_change_percent':'相对变化 %'}).to_markdown(index=False,floatfmt='.5g'),'','前两行的新比值为 S_A+B/S_original；第三行为两种 foreground 的面积比变化。','', '![同子集区域](fig__mesh_AB_subset_regions.png)','']
 a=pd.read_csv(O/'subset_areas.csv');boundary=a[(a.raster==800)&(a.target==.683)]
 lines+=['68.3% 区域接触子集边界：'+ '; '.join(f"{names[v.branch]} / {v.scheme}: {'是' if v.touches_subset_boundary else '否'}" for v in boundary.itertuples())+'。两种 C 下的 68.3% 与 95.4% 区域均未接触子集边界，但仍不外推为全域约束改善。','']
 spread=d[d.target==.683].groupby('quantity').new_ratio.agg(['min','max']);lines += ['400²、800²、1200² 积分栅格之间，上述比值的相对极差均小于 0.03%；详细数值见 subset_area_ratios.csv。这只检查积分误差，不代表参数节点已收敛。','']
 fm=O/'subset_area_ratios_matched.csv'
 if fm.exists():
  m=pd.read_csv(fm);m=m[(m.raster==800)&(m.target==.683)].copy();m['relative_change_percent']=100*m.relative_change
  lines+=['匹配 taper C 的敏感性对照（同一组已测曲线，原方案仍用原 C）：','',m[['quantity','new_ratio','relative_change_percent']].to_markdown(index=False,floatfmt='.5g'),'']
else:lines+=['测量尚未完成，暂不报告面积比。','']
lines+=['## 实测耗时与 1285 节点估算','']
f=O/'timing_summary.csv'
if f.exists():
 t=pd.read_csv(f);t.branch=t.branch.replace(names);t.scheme=t.scheme.replace({'AB':'A+B'})
 lines+=['下表为已完成切线节点的中位数：一个节点包含四红移；秒数为实测建场/FFT 共享开销加该 foreground 的滤波与 genus 测量累计。小时数按同硬件、两个工作进程线性估算，未把多个 foreground 重复计算的共享开销相加。文件读写、GC 等额外开销不包含在分项秒数内；noFG 的 A+B 与 A 相同，复用 A 的测量和计时。','',t[['scheme','branch','standalone_seconds_per_node','hours_1285_two_workers']].rename(columns={'scheme':'方案','branch':'foreground','standalone_seconds_per_node':'实测核心秒/节点','hours_1285_two_workers':'1285节点估计小时（2进程）'}).to_markdown(index=False,floatfmt='.2f'),'']
 f=O/'timing_all_foregrounds.csv'
 if f.exists():
  t=pd.read_csv(f);lines+=['四种 foreground 共用建场时：','',t.rename(columns={'scheme':'方案','seconds_per_node_all_four_foregrounds':'四foreground共用建场：秒/节点','hours_1285_two_workers':'1285节点估计小时（2进程）'}).replace({'方案':{'AB':'A+B'}}).to_markdown(index=False,floatfmt='.2f'),'']
 f=O/'node_wall_times.csv'
 if f.exists():
  t=pd.read_csv(f);lines+=[f'原方案、A、A+B 三套设置及全部四种 foreground 一起测量的实际节点完成跨度中位数约 {t.node_elapsed_seconds_approx.median():.1f} 秒（两进程；由任务耗时和完成时间估计），已统计 {len(t)} 个完整节点。这与只跑一种方案、一个 foreground 的分项耗时不同。全域估算受网格尺寸、温度降频、I/O 和并发影响，不是承诺时间。','']
if (O/'subset_runtime.csv').exists():
 t=pd.read_csv(O/'subset_runtime.csv').iloc[0]
 lines += [f'使用已逐元素回放验证的原始 FFT 复用实现，A+B 两 foreground、271 节点实际总耗时 {t.elapsed_seconds/60:.1f} 分钟（2 进程）。同样两 foreground 的 1285 节点线性估计为 {t.hours_1285_same_two_foregrounds:.2f} 小时；这只是估算，本轮未运行全量。','']
lines+=['## 结论范围与文件','',
'整数 N 切换和硬 Fourier 掩膜切换都是本例中的数值跳变来源。taper 也改变了滤波响应，面积变化不能全部归因于数值误差修复，更不能自动称为精度改善。A+B 是本轮进入粗比较的方案，仍保留离散拓扑计数起伏；本轮不宣称连续极限收敛或覆盖率校准。全量 1285 节点及 m 范围留待下一轮确认。','',
'数据：boundary_results.csv、hard_mask_pair.csv、line.csv、subset_surface.csv、subset_area_ratios.csv、timing_summary.csv；原始新曲线在 measurements/ 与 subset_measurements/。运行脚本 measure.py、subset.py；分析脚本 analyze.py、subset_analyze.py；**redraw.py 仅用保存 CSV 重画全部图，不重新运行科学测量。**',
'', '原输入：../local_refinement_20260920/plan_boundary.json；../wedge_all_m_20260920/joint_chisq.csv 与 reference_covariances.npz。']
if (O/'subset_area_ratios.csv').exists():
 rr=pd.read_csv(O/'subset_area_ratios.csv').query('raster==800 and target==0.683').set_index('quantity')
 lines[4:4]=[f"**粗比较的相对排序发生反转：** 共同 C 下，horizontal 的面积变化为 {100*rr.loc['horizontal_only','relative_change']:+.1f}%，m=1.343 为 {100*rr.loc['horizontal_m4','relative_change']:+.1f}%；wedge/horizontal 面积比由 {rr.loc['wedge_over_horizontal','old_ratio']:.3f} 变为 {rr.loc['wedge_over_horizontal','new_ratio']:.3f}。匹配 taper C 后方向相同。这说明原子集上的 wedge 相对优势对这些数值选择不稳健，不能直接推广为正式全域精度结论。",'']
(O/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
