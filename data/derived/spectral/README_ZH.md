# 水平切割 genus 振幅比的 Fourier 复算

日期：2026-10-29。使用已有数据与矩阵，不重跑建场、FFT或拓扑测量，不修改冻结稿件。

条件：snap118，z=0.10235，scan=sim，R=20 h^-1 Mpc，水平切除 |k_parallel|<0.1 h/Mpc。两支直接振幅与当前v0.20稿件data/sample_amplitudes.csv逐值一致。由保存的3×3矩阵独立计算行列式，复现谱矩振幅到相对1e-12以内。

|方法|noFG AR³|水平切割 AR³|切割/noFG|
|---|---:|---:|---:|
|直接拓扑Gaussian拟合|0.005164041315|0.023645219003|4.578820649|
|补零域各向异性Fourier谱矩|0.005503678760|0.024626609764|4.474572525|
|保留区域谱导数梯度矩诊断|0.005359380670|0.024383196828|4.549629580|

主Fourier比值比直接拟合低2.27675%。保留区域梯度诊断低0.63752%，但不是严格边界修正，也不是纯补零域功率谱积分。

计算公式：M_ij = sum_k[k_i k_j |delta_filtered(k)|²] / sum_k[|delta_filtered(k)|²]，A_G=sqrt(det M)/(4π²)，振幅比=sqrt(det M_cut/det M_noFG)。三维求和保留方向信息；实际rFFT有半谱重数及Nyquist处理。已滤波谱包含平滑、CIC去卷积和mask，不能重复施加窗口。相同R下AR³之比等于A之比。

切割降低方差但改变归一化梯度协方差，因而可以提高振幅。不能对强各向异性场套用trace各向同性公式：本组数据会得到12.0362倍。

这支持“高斯二阶谱矩预测能够近似解释约4.6倍增强”，不支持精确重现、显著性判定或补零收敛结论。谱矩丢失相位信息；非高斯性、有限窗口和离散拓扑均可能贡献残差，本次未分解这些贡献。无需仅因残差大小选择诊断口径。

## 数据与复算

- [原始比较CSV](../../outputs/scan_sim_fixed_cell_20260909_v1/spectral_topology_comparison/comparison.csv)
- [旧计算报告及Parseval验证](../../outputs/scan_sim_fixed_cell_20260909_v1/spectral_topology_comparison/REPORT_ZH.md)
- [复算脚本](recalculate.py)，只用Python标准库读取既有CSV/NPZ。
- [比值CSV](ratios.csv)；[矩阵、数值核验与源CSV哈希](verification.json)。

## 既有文献依据

沿用[谱矩公式与文献登记](../../GENUS_SPECTRAL_AMPLITUDE_NOTE_20260909.md)，本轮未开展新的外部文献核查。登记包含Takahiko Matsubara (1996), Statistics of Isodensity Contours in Redshift Space, ApJ 457, 13–17, DOI 10.1086/176708, arXiv astro-ph/9501055；既有全文定位为§3.1、式(3.2)–(3.5)。各向同性谱矩公式与线性RSD见原文；一般det(M)式为项目Gaussian坐标变换推导，不能宣称原文验证了本项目硬mask及有限窗口。
