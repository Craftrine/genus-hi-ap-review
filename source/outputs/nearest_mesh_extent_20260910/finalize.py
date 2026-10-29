from pathlib import Path
import json,zipfile,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;P=ROOT/'paper_figures/fig__intrinsic_subtracted_contours';D=P/'extent_mesh';arc=P/'archive_before_extent';arc.mkdir(exist_ok=True)
for n in ['fig__intrinsic_subtracted_contours.png','fig__intrinsic_subtracted_contours.pdf','active_chisq_grid.csv','replot.py']:
 if (P/n).exists():shutil.copy2(P/n,arc/n)
for ext in ['png','pdf']:shutil.copy2(D/f'overview_intrinsic_subtracted.{ext}',P/f'fig__intrinsic_subtracted_contours.{ext}')
shutil.copy2(D/'before_after_chisq.csv',P/'active_chisq_grid.csv')
s=(OUT/'plot.py').read_text().replace('parents[2]','parents[3]');(D/'replot.py').write_text(s)
(P/'replot.py').write_text('''from pathlib import Path\nimport subprocess,sys,shutil\np=Path(__file__).resolve().parent\nsubprocess.run([sys.executable,str(p/'extent_mesh/replot.py')],check=True)\nfor ext in ['png','pdf']:\n shutil.copy2(p/f'extent_mesh/overview_intrinsic_subtracted.{ext}',p/f'fig__intrinsic_subtracted_contours.{ext}')\n''')
checks=json.loads((D/'boundary_audit.json').read_text());lines=['# Extended new-contract contour coverage','','The central149-point set is retained.18 pilot points and66 additional adaptive points extend the sampled degeneracy band, yielding233 cosmologies. All use the same full-box nearest-spacing contract, intrinsic template and fixed covariance. No cropped-trial data are merged.','','The support is the measured convex hull, extending Omega_m to0.20 and0.42 and following the tilted band in w. Gray plot regions are unevaluated; there is no interpolation there. HPD probabilities integrate exp(−Q/2) under a uniform prior inside this support. A closed contour is not by itself independent coverage calibration. Outer sampling is deliberately coarser than the central grid; this extension establishes coverage better than precision of detailed outer-boundary shapes.','','## Joint boundary audit','']
for v in checks:
 if v['variant']=='intrinsic_subtracted' and v['mode']=='joint':
  lines.append('- '+v['branch']+': '+', '.join(f"{100*r['probability']:g}% {'touches support' if r['touches_sampled_boundary'] else 'closed within support'}" for r in v['regions']))
lines+=['','Single-pair boundary checks are in boundary_audit.json. Wider individual-pair regions can still meet the support even if the joint regions close. The two formulations (with/without intrinsic subtraction) and all32 individual plots are saved; current main image shows intrinsic-subtracted regions.','','Text files:curves.csv,before_after_chisq.csv,geometry.json,integrated_hpd_levels.csv,boundary_audit.json. Replay:replot.py for overview and replot_integrated.py for individual panels. Previous dense and narrow-local versions remain archived. draft.pdf remains the earlier figure-compilation snapshot until explicitly refreshed.']
(D/'README.md').write_text('\n'.join(lines)+'\n')
with zipfile.ZipFile(D/'extended_contours_and_data.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in ['README.md','before_after_chisq.csv','before_after_summary.json','curves.csv','geometry.json','integrated_hpd_levels.csv','integrated_hpd_summary.json','boundary_audit.json','verification.json','overview_intrinsic_subtracted.png','overview_intrinsic_subtracted.pdf','overview_intrinsic_not_subtracted.png','overview_intrinsic_not_subtracted.pdf','replot.py','exact_hpd.py','replot_integrated.py']:z.write(D/n,n)
 for f in (D/'integrated_probability').rglob('*.pdf'):z.write(f,f.relative_to(D))
print('\n'.join(lines))
f=ROOT/'CURRENT_ANALYSIS.md';s=f.read_text().replace('# 当前最新分析结果\n','# 当前最新分析结果\n\n新流程退化方向扩展完成：保留149中心点，增加84自适应点，共233宇宙学位置。当前主contour指向extent_mesh；概率在实测凸包内积分，未采样角落灰色显示。闭合性见boundary_audit.json，未将外域插值或裁切旧结果接入。\n',1);f.write_text(s)
f=ROOT/'paper_figures/FIGURE_REGISTER.md';s=f.read_text().replace('新流程149点较密局部HPD，扣/不扣intrinsic；含同域网格敏感性','新流程233点扩展HPD，保留中心密网格，扣/不扣intrinsic；含边界审计');f.write_text(s)
(P/'README.md').write_text('# Current contour:233-point extended new-contract mesh\n\nCurrent results are in extent_mesh. See its README and boundary_audit.json; older local-domain statements below are historical.\n\n'+(P/'README.md').read_text())
