# 专家复核入口

本仓库用于查验和修改 H I genus–AP 论文 **v0.21（2026-10-29）**。建议先阅读论文，再按问题定位代码与保存结果。

| 内容 | 入口 |
| --- | --- |
| 论文 PDF | [main.pdf](../manuscript/main.pdf) |
| 可编辑论文 | [main.tex](../manuscript/main.tex)；分节正文在 [sections](../manuscript/sections/) |
| 方法与代码对应 | [来源说明](PROVENANCE.md)、[代码清单](../CODE_MAP.csv)、[source](../source/) |
| 图表与派生结果 | [data/derived](../data/derived/)、[数据清单](../DATA_MAP.csv) |
| 可复现范围及未完成事项 | [REPRODUCIBILITY_LIMITS.md](REPRODUCIBILITY_LIMITS.md) |

## 如何反馈和修改

只需提出意见时，请在 [Issues](https://github.com/Craftrine/genus-hi-ap-review/issues) 新建“Scientific review / 科学复核”，写明章节、公式、图表或代码位置，以及判断依据。一个 Issue 尽量讨论一个具体问题，便于逐项回应。

希望直接修改时，可 **Fork 本仓库 → 在自己的分支编辑 → 提交 Pull Request**。仓库所有者审核后合并；无需先获得本仓库的直接写入权限。论文措辞修改可直接编辑 `manuscript/sections/*.tex`。详细操作和验证范围见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

## 可运行的复核

按 [README](../README.md#installation-and-entry-points) 配置 Python 3.12 环境和依赖后，在仓库根目录运行：

```bash
python scripts/reproduce_saved_results.py --output-dir review_outputs/saved_checks
python scripts/redraw_figures.py --output-dir review_outputs/figures
```

第一条命令核对保存文件、Table 2、各向异性高斯谱矩及 jackknife/OAS 计算；第二条从保存输入重画图表。生成文件写入 `review_outputs/`。论文主文件为 `manuscript/main.tex`，使用 **XeLaTeX** 和 BibTeX；补充材料以 `manuscript/supplement.tex` 单独编译。

本仓库不含原始九天模拟目录和全部三维场，因此这些命令验证的是保存结果及后处理，不能替代从原始数据重新测量。部分 `source/` 文件保留历史生产入口及机器路径结构，运行前请先查代码清单，勿逐个直接执行。

## 本轮重点

建议重点核对 jackknife 的 `63/64` 归一化、OAS 标准化与方差恢复、约 4.6 倍振幅的各向异性谱矩解释，以及网格、硬 Fourier 边界与参数面平滑对区域面积的影响。

v0.21 保留了已有 TODO 和未完成的编辑项，包括具体 H I 目录来源、Data availability、Fig. 9 的末端点及若干方法补述。详见[未完成事项](REPRODUCIBILITY_LIMITS.md#unresolved-manuscript-edits-preserved-in-v021)。软件检查通过不代表这些问题已解决，也不代表数值收敛或统计覆盖率已获独立验证。
