# 稿件数值表的紧凑副本

本目录复制了9份原始CSV，共44,206字节。所有文件与项目来源逐字节相同；未重算测量、没有修改精度或单位。绝对来源和SHA-256见 `SOURCE_MANIFEST.json`。目录只打包稿件相关的小表，完整曲线、矩阵、场、扫描缓存仍保留在原项目。

|稿件内容|本目录CSV|口径|
|---|---|---|
|样本振幅端点表|`sample_amplitudes.csv`|各红移投影振幅及其已保存误差；正文端点变化为高/低红移振幅之比减1|
|九档相对面积表|`nine_m_areas_exact.csv`|完整凸包内精确三角积分；包含linear-Q、linear-P和三个质量层级|
|平滑前后面积表|`smoothing_comparison_800.csv`|800×800栅格、linear-P、68.3%及95.4%；面积原始单位，正文显示乘10⁴|
|求积分辨率及noFG平滑对照|`smoothing_areas_all.csv`|含800/1600两档、全部11分支、三个层级；不可与精确三角表逐位混用|
|A/B网格边界表|`mesh_boundary_jumps.csv`|`nofg_3`为探针对A；`horizontal_only_1`为B；ΔQ=右−左|
|参数局部加密|`local_parameter_refinement_areas.csv`|四分支1285→1325、同凸包，68.3%面积变化|
|空间格距同点比较|`spatial_resolution_point_Q.csv`|各格距匹配sim模板；分别保存固定C4及匹配协方差Q|
|空间格距边界比较|`spatial_resolution_paired_jumps.csv`|A/B原位置及A2/B2新边界的ΔQ与配对残差范数|
|空间格距残差变化|`spatial_resolution_between.csv`|同点跨格距的残差变化范数，以C4为度量|

后三份空间分辨率CSV中的 `branch=0` 为noFG，`branch=1` 为仅水平切割；残差范数不是观测显著性的sigma。各分辨率采用相应模板，因此“固定C4”只固定度量，不表示模板也沿用格距4。

11幅稿件PDF的原始资产、完整绘图数据所在目录、仅重画入口和SHA-256见 `../notes/FIGURE_SOURCES.csv` 与 `../notes/figure_pdf_checks.json`。曲线/误差、切片、协方差及轮廓顶点均在所列来源中留存。重新执行绘图脚本不应运行测量用的 `run.py`。

本包保留有限域条件面积、非单调趋势与未收敛证据。数据副本的完整性不等于独立统计覆盖率或最终宇宙学精度验证。
