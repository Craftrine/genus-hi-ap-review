from pathlib import Path
import json,csv,zipfile
import numpy as np
OUT=Path(__file__).resolve().parent
def read(name):
    rows=list(csv.DictReader((OUT/name).open()))
    for r in rows:
        for k,v in r.items():
            try:r[k]=float(v)
            except ValueError:pass
    return rows
amp=read('amplitudes.csv');sel=read('sample_selection.csv');response=read('ap_response.csv');cov=read('covariance_summary.csv')
boundary=json.loads((OUT/'boundary_audit.json').read_text())
branches=['nofg','horizontal_only','horizontal_m1','horizontal_m5']
comparison=[]
for b in branches:
    record=dict(branch=b)
    for sample in ['top1m','all_halo']:
        a=next(v for v in amp if v['sample']==sample and v['branch']==b and v['snapshot']==118)
        z=next(v for v in amp if v['sample']==sample and v['branch']==b and v['snapshot']==94)
        record[sample+'_AR3_lowz']=a['AR3'];record[sample+'_AR3_highz']=z['AR3'];record[sample+'_evolution_percent']=100*(z['AR3']/a['AR3']-1)
    record['all_vs_top_highz_percent']=100*(record['all_halo_AR3_highz']/record['top1m_AR3_highz']-1)
    comparison.append(record)
with (OUT/'comparison_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
lines=['# All-halo HI加权与top-1M HI加权对照（2026-09-11）','','## 已执行范围','',
'全目录重算4个红移、37个参数位置、4种前景情景，共592条genus曲线；配对复用并验证top-1M的592条曲线。参数点为原25个局部规则节点加12个极小轴向节点，范围Ωm=[0.3091,0.3131]、w=[−1.01,−0.99]。R=20 h⁻¹ Mpc、目标格距4、各方向完整AP盒最近格距、原CIC/去卷积、160格padding、原Fourier切割和41个体积分数阈值均保持一致。',
'','all halo具体指所提供compact目录内位置/速度/HI质量有限、HI质量正的所有记录，以HImass加权；没有额外top-N门槛。目录本身的质量分辨率和选择仍然存在，不能把它解释为所有未解析HI或无观测选择的真实宇宙。',
'','## 样本规模与HI质量覆盖','','|snapshot|z|all有效对象数|top-1M占HI质量|','|---|---:|---:|---:|']
for r in sel:
    if r['sample']=='all_halo':
        t=next(v for v in sel if v['sample']=='top1m' and v['snapshot']==r['snapshot'])
        lines.append(f"|{int(r['snapshot'])}|{r['z']:.5f}|{int(r['objects']):,}|{100*t['HI_fraction']:.3f}%|")
lines+=['','源目录逐字节SHA256与既有完整选样审计一致，4个目录全部有效对象计数一致，全部网格HI质量守恒误差小于3×10⁻¹⁵。','',
'## 基准genus及红移演化','','|情景|top-1M高z AR³|all高z AR³|all相对top高z变化|top从低z到高z变化|all从低z到高z变化|','|---|---:|---:|---:|---:|---:|']
for r in comparison:lines.append(f"|{r['branch']}|{r['top1m_AR3_highz']:.7f}|{r['all_halo_AR3_highz']:.7f}|{r['all_vs_top_highz_percent']:+.3f}%|{r['top1m_evolution_percent']:+.3f}%|{r['all_halo_evolution_percent']:+.3f}%|")
lines+=['','AR³是同一41点Gaussian模板拟合的带符号振幅；不能把它替代所有曲线形状信息。全曲线和阈值保留在curves.csv。图中逐点误差来自与旧图相同的64块加性jackknife；all−top差值使用匹配块的差量，不把两个高度相关样本当独立样本相加误差。',
'','## AP响应与统计处理','','每个样本均扣除自己的fiducial本征演化，且重新计算自己的64块匹配红移差量协方差和OAS收缩矩阵。top-1M协方差逐元素重现现有参考矩阵，all样本的矩阵不借用top-1M。所有Q用Cholesky白化独立核对。',
'','同盒相减使两个样本的中心Q=0，因此中心恢复不是独立检验。64个空间块不是64个独立宇宙实现，123维原始协方差秩至多63；本次使用匹配的收缩规则保证可比性，不声称完成覆盖率校准。',
'','另保存把all残差放进top-1M协方差的Q，专门区分残差变化与误差权重变化；此曲线明确标为诊断，不能当作all样本的误差或参数限制。半倍/双倍收缩的joint Q也保存在covariance_sensitivity.csv。',
'','## 共同局部域的轮廓边界','','|样本|情景|68.3%触边|95.4%触边|99.7%触边|','|---|---|---|---|---|']
for sample in ['top1m','all_halo']:
    for b in branches:
        rr=[r for r in boundary if r['sample']==sample and r['branch']==b and r['variant']=='intrinsic_subtracted']
        lines.append('|'+sample+'|'+b+'|'+'|'.join('是' if r['touches_boundary'] else '否' for r in rr)+'|')
lines+=['','概率在两个样本完全相同的局部矩形内，以相同线性Q三角剖分和exp(−Q/2)积分方法归一。68.3/95.4/99.7%为该有限域内HPD，不是全局误差；触边时不能给闭合区间或面积精度比。边界检查纳入Delaunay凸包所有边界子线段的端点（包括矩形边上共线节点），并与矩形边全部实测节点交叉验证，不能只检查四角。没有把旧233点扩展域的top椭圆直接叠在37点all域上。若要比较完整全局轮廓，下一轮需按all样本的响应范围扩展共同支持域。',
'','## 加速与数值验证','','同一整数网格形状下，AP后的对象坐标与格距同比缩放，p′/Δ′=pN/L。因此可将每个完整目录按各所需N形状沉积一次，再逐参数点按真实AP格距计算Fourier波数、CIC去卷积、平滑和切割。没有用同一物理平滑场替代不同AP点，没有引入旧裁切。',
'','标量CIC与原CIC在含边界的测试中float64逐元素一致；四红移涉及的全部12个“红移/形状”组合、每个4分支，优化路径的负Euler计数与已保存直接AP top-1M测量逐元素一致，阈值满足绝对误差2×10⁻⁷。见replay_gate.json。该验证有限，不宣称任意浮点输入下路径必然位级相同。',
'','## 文件','','- fig__all_halo_vs_top1m__genus：最高红移四情景曲线及配对差量误差。','- fig__all_halo_vs_top1m__all_redshifts：四红移全部16面板。','- fig__all_halo_vs_top1m__amplitude_evolution：四红移拟合振幅和误差。','- fig__all_halo_vs_top1m__ap_response：轴向Q，独立样本协方差及共同协方差诊断。','- fig__all_halo_vs_top1m__contours / _raw：扣除/不扣除本征演化的同域联合HPD对照。',
'- curves.csv、fiducial_comparison.csv、amplitudes.csv、sample_selection.csv、chisq.csv、ap_response.csv、covariance.csv、covariance_sensitivity.csv：绘图与科学数据。',
'- stream.py：4进程有界全目录流式计算；analyze.py：两样本各自基线/协方差与Q；hpd_*/exact_hpd.py：同域HPD积分；replot.py：从文本数据重画；finalize.py：数据摘要及打包。',
'- snap*/density_summary.json：源SHA256、有效对象数、HI质量、格网守恒、缓存哈希；snap*/fiducial_blocks.npz：64块本征统计。',
'- 各参数summary.json的geometry来自匹配top-1M几何记录，其中旧object/weight字段仅为来源元数据，all对象/权重以density_summary.json为准；科学体积和格距与两个样本完全匹配。',
'','## 本轮解释','','在Ωm=0.3121、w=−1处，horizontal_only的joint Q：top-1M/自身协方差0.6637，all/自身协方差2.0819，all/top-1M协方差0.6685。因此该点的Q增强主要与样本专属协方差有关，不是残差幅度增加很多；不能把Q约三倍直接换算为全局参数精度提高约三倍。',
'','本轮支持all-HI样本也保留对AP几何的条件响应，并显示本征红移演化和误差结构依赖选样。noFG的top-1M拟合振幅从低z到高z约+0.11%，all约+2.91%；这不能仅由此四个点称作独立的演化显著性检测，也不能将“本征近乎不变”无条件推广给全样本。',
'','本次作为样本选择敏感性补充，不替换当前八张正式图，不更新旧draft或覆盖既有科学结果。']
direct=OUT/'full_catalog_direct_AP_verification.json'
if direct.exists():
    dd=json.loads(direct.read_text());assert dd['status']=='PASS'
    lines+=['','## 全目录直接AP独立复核','',f"另对snap94、Ωm=0.3121、w=−1的{dd['all_valid_objects']:,}个有效对象逐一执行真实AP变换和物理坐标CIC，独立于归一坐标复用。四分支41点负Euler计数与优化路径全部逐元素一致；密度float32格点最大差{dd['density_max_difference']:.6g}，差异格点比例{dd['density_fraction_different']:.6g}。详见full_catalog_direct_AP_verification.json。"]
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
with zipfile.ZipFile(OUT/'all_halo_vs_top1m_results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ['.md','.csv','.json','.py','.png','.pdf']:z.write(p,p.name)
    for p in OUT.glob('hpd_*/*'):
        if p.suffix in ['.py','.json','.csv']:z.write(p,p.relative_to(OUT))
print(json.dumps(comparison,indent=2));print('Packaged data and figures.')
