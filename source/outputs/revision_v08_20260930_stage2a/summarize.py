from pathlib import Path
import json,re
import pandas as pd
O=Path(__file__).resolve().parent;T=O/'tables';P=O.parents[1]
assert json.loads((O/'qa/analysis_validation.json').read_text())['completed']
w=pd.read_csv(T/'eleven_wedge_areas.csv');r=pd.read_csv(T/'eleven_wedge_raster_areas.csv')
s=pd.read_csv(T/'covariance_swap_areas.csv');f=pd.read_csv(T/'four_cases_regions.csv')
rows=[]
tex=(P/'paper/chinese_closure_v0.8_20260930/sections/05_results.tex').read_text()
tables=re.findall(r'\\begin\{table\}.*?\\end\{table\}',tex,re.S)
for table in tables:
    if 'tab:ninem' in table:
        for line in table.splitlines():
            if not re.match(r'^\d',line):continue
            vals=[x.strip().rstrip('\\').strip() for x in line.split('&')]
            mm=float(vals[0])
            for target,col in [(.683,1),(.954,2)]:
                a=w[(w.method=='linear_P')&(w.target==target)&(abs(w.m-mm)<.00051)].iloc[0]
                v=float(a.area_over_horizontal);old=vals[col];new=f'{v:.3f}'
                rows.append(dict(table='Table 3',row=vals[0],metric=f'ratio_{target}',printed_old=old,printed_new=new,new_unrounded=v,display_matches=old==new))
    if 'tab:smoothing' in table:
        for line in table.splitlines():
            if not (re.match(r'^\d',line) or line.startswith('noFG &') or line.startswith('Horizontal cut &')):continue
            vals=[x.strip().rstrip('\\').strip() for x in line.split('&')]
            label=vals[0]
            selected=r[(r.grid==800)&(r.target==.683)]
            if label in ['noFG','Horizontal cut']:
                selected=selected[selected.branch=={'noFG':'nofg','Horizontal cut':'horizontal_only'}[label]]
            else:selected=selected[abs(selected.m-float(label))<.00051]
            raw=selected[selected.variant=='no_gaussian'].iloc[0];sm=selected[selected.variant=='gaussian'].iloc[0]
            for i,metric,value,digits in [(1,'1e4_unsmoothed_area',raw.area*1e4,3),(2,'1e4_smoothed_area',sm.area*1e4,3),(3,'smoothed_over_unsmoothed',sm.smoothed_over_unsmoothed,2)]:
                old=vals[i];new=f'{value:.{digits}f}'
                rows.append(dict(table='Table 4',row=label,metric=metric,printed_old=old,printed_new=new,new_unrounded=value,display_matches=old==new))
check=pd.DataFrame(rows);assert len(check)==51 and check.display_matches.all()
check.to_csv(T/'manuscript_printed_tables_comparison.csv',index=False,float_format='%.17g')
primary=w[(w.method=='linear_P')&(w.target==.683)]
small=primary[primary.branch.str.startswith('m')].sort_values('m')
table='| m | 68.3% 面积 / 水平切面积 |\n|---:|---:|\n'
table+='\n'.join(f'| {x.m:.6f} | {x.area_over_horizontal:.6f} |' for x in small.itertuples())
minimum=small.loc[small.area.idxmin()]
swap=s[(s.method=='linear_P')&(s.target==.683)]
def area(res,cov):return swap[(swap.residual_case==res)&(swap.covariance_case==cov)].area.iloc[0]
qref=json.loads((O/'qa/input_gate.json').read_text())
val=json.loads((O/'qa/analysis_validation.json').read_text())
text=f'''# 阶段 2a 一页摘要（2026-09-30）

只使用已保存的 1285 点数据，完成数值后处理；未新增 genus 测量，未将新结果写入正文，未启动 2b 或阶段 3。以下均是公共凸包内的条件区域。

**数据与旧结果对照通过。** 四情景共 5140 个 joint Q 的最大差值为 {qref['max_abs_difference']:.3e}。水平切 68.3% 面积复现为：linear-P 三角积分 **2.43319079697e-4**；800×800 栅格 **2.43531573415e-4**；linear-Q **9.67932672608e-5**。现稿 Table 3、4 的 **51 个印刷数值全部匹配**；完整精度源表最大相对差为 {val['max_manuscript_relative_difference']:.3e}。

**十一档仍非单调。** 未平滑 linear-P 主结果如下；最小的已测 68.3% 区域仍在 m≈{minimum.m:.6f}，不是对连续 m 的最优值声明。

{table}

**边界。** noFG 的三个单红移对在三档概率水平下均触边；边界带概率分别为 7.893%、7.566%、9.054%。noFG joint 的 68.3%、95.4% 区域闭合，99.7% 触边，边界带概率为 0.634%。三个前景切割情景的 joint 三档区域均闭合。带宽定义为坐标分别除以 (0.00275, 0.01) 后距凸包边界一个单位。

**交换协方差。** 固定 noFG 残差，换用水平切协方差时，68.3% linear-P 面积从 {area('nofg','nofg'):.6g} 变为 {area('nofg','horizontal_only'):.6g}（触边）；固定水平切残差，换用 noFG 协方差时，从 {area('horizontal_only','horizontal_only'):.6g} 变为 {area('horizontal_only','nofg'):.6g}。这个对照中，水平切的权重没有单独造成收紧；不作可加贡献分解。加权余弦已逐点保存，只描述 d 与 I 的方向，不能证明噪声抵消。

**平滑与积分。** 十一档在 800 栅格上，额外平滑使 68.3% 面积扩大 **6.515–12.934 倍**。800→1600 的最大面积相对变化为 **0.5291%**（m≈1.517391，未平滑，99.7%）；仅 68.3% 时最大为 **0.3868%**。这验证固定概率面上的积分稳定性，不等于参数采样或三维建场收敛。

数值表、闭合/触边的逐轮廓记录、逐点余弦及四张诊断图见 [交付索引](README_ZH.md)。等待本轮数值验收，正文由作者起草。
'''
(O/'SUMMARY_ZH.md').write_text(text)
print('51 printed values match; one-page summary saved.')
