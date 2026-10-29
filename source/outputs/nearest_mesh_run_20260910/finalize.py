from pathlib import Path
import json,zipfile
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;DEST=ROOT/'paper_figures/fig__intrinsic_subtracted_contours/nearest_mesh'
a=json.loads((DEST/'continuity_comparison.json').read_text());state=json.loads((OUT/'status.json').read_text());qrows=[]
import csv
for r in csv.DictReader((DEST/'before_after_chisq.csv').open()):qrows.append(r)
lines=['# 最近4格距完整盒子对照结果','',f"已完成37个参数点（25点局部网格+12个极小轴向点）、148次建场、592条曲线。四并行测量耗时{state['elapsed_seconds']:.1f}秒。全部对象/HI质量保留；四红移fiducial共16条曲线及阈值与旧版逐元素相同。",'', '## 最小正向偏移处，旧裁切与新完整盒子的联合Q','','|参数轴|情景|旧Q|新Q|','|---|---|---:|---:|']
for v in a:
 if v['sign']==1:lines.append(f"|{v['axis']}|{v['branch']}|{v['old_Q_smallest']:.5g}|{v['new_Q_smallest']:.5g}|")
lines+=['','## 解释边界','','本轮保持intrinsic基线与协方差不变，只改变完整盒子建网格和各轴格距的处理。新的局部曲线不与旧裁切的trial数据拼接。原正确点附近裁去整层网格的问题，在此方案中不再发生。剩余离散阈值、整数N变化、CIC去卷积、有限边界及协方差物理校准问题不能仅凭本轮宣告全部解决。','','## 轮廓','','同时给出扣除/不扣除intrinsic的四情景×四红移组合，各含68.3/95.4/99.7% HPD轮廓。积分仅归一于当前新流程已测局部范围Omega_m=[.3091,.3131]、w=[−1.01,−.99]；不是原大范围上的最终限制，边界不能当作物理约束。若需要全局限制，必须用新流程补测更大范围，不能接入旧裁切trial结果。','','## 文件','','[连续性比较图](../../paper_figures/fig__intrinsic_subtracted_contours/nearest_mesh/continuity_old_vs_new.png)；所有新数据、Q、积分结果和图件位于paper_figures/fig__intrinsic_subtracted_contours/nearest_mesh。全部绘图文本数据保留，旧结果不覆盖。“错位相减”仅记录，没有执行。']
(OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');(DEST/'README.md').write_text('\n'.join(lines)+'\n\nMethod: see outputs/nearest_mesh_run_20260910/METHOD.md.\n')
with zipfile.ZipFile(DEST/'nearest_mesh_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in ['README.md','before_after_chisq.csv','before_after_summary.json','curves.csv','geometry.json','continuity.csv','continuity_comparison.json','verification.json','integrated_hpd_levels.csv','integrated_hpd_summary.json','continuity_old_vs_new.png','continuity_old_vs_new.pdf','exact_hpd.py','replot_integrated.py']:z.write(DEST/n,n)
 for p in (DEST/'integrated_probability').rglob('*.pdf'):z.write(p,p.relative_to(DEST))
 for n in ['run.py','test_geometry.py','analyze.py','diagnose.py','METHOD.md','REPORT_ZH.md','status.json']:z.write(OUT/n,'production/'+n)
 z.write(ROOT/'full_hi_pipeline/scan_sim_nearest_mesh.py','production/scan_sim_nearest_mesh.py')
print('Packaged new results')
