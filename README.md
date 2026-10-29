# H I genus–AP analysis: review package

**专家入口：[论文 PDF](manuscript/main.pdf) · [中文导读](docs/EXPERT_REVIEW_ZH.md) · [反馈问题](https://github.com/Craftrine/genus-hi-ap-review/issues) · [提交修改说明](CONTRIBUTING.md)**

这是 v0.21 的公开查验稿。专家可以直接阅读 PDF、核对保存数据和代码，在 Issues 中提出问题，或通过 Fork 和 Pull Request 提交修改；最终修改由仓库所有者审核合并。当前稿件尚未完成的事项保留在[复现范围与限制](docs/REPRODUCIBILITY_LIMITS.md)中。

This repository supports a public cross-check of manuscript **v0.21 (29 October 2026)**, *Redshift evolution of the H I-weighted genus under Alcock–Paczyński distortions and foreground filtering*. It combines the analysis source, saved derived measurements, figure inputs, verification utilities, a readable PDF and the editable manuscript. Inclusion in this repository does not constitute independent scientific validation.

The available checks reproduce operations on saved numerical results. The original simulation catalogues and full three-dimensional fields are not included, so this is **not a self-contained rerun from raw Jiutian data**. The manuscript's remaining factual TODOs and numerical limitations are preserved.

## Contents

| Path | Purpose |
| --- | --- |
| `source/` | Copies of the measurement and analysis code, retaining their original relative paths. Some are historical or import-time execution scripts; use the code map before running them. |
| `tests/` | Selected existing numerical correctness tests. |
| `scripts/` | Portable entry points for checking saved results and redrawing figures. |
| `data/derived/` | Selected published-figure inputs, saved block statistics and verification records; no raw simulation catalogue. |
| `manuscript/` | v0.21 LaTeX inputs and original figure assets. |
| `CODE_MAP.csv` | Source provenance, roles and checksums. |
| `DATA_MAP.csv` | Data and manuscript provenance and checksums. |
| [Reproducibility limits](docs/REPRODUCIBILITY_LIMITS.md) | What each verification does and does not establish. |
| [Provenance](docs/PROVENANCE.md) | Measurement conventions, manuscript status and third-party attribution. |

## Installation and entry points

Use Python **3.12** in a separate environment. `requirements.txt` records the resolved review environment; it is not a reconstructed historical production environment. From the repository root:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=source NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests
PYTHONPATH=source NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python source/outputs/nearest_mesh_run_20260910/test_geometry.py
python scripts/reproduce_saved_results.py --output-dir review_outputs/saved_checks
python scripts/redraw_figures.py --output-dir review_outputs/figures
```

The saved-results check verifies input hashes, the complete redshift table, the anisotropic Gaussian spectral ratio and the archived block-covariance calculation. Use `--skip-jackknife` for the smaller hash/table/spectral check. Figure replay supports `--only basic`, `compact`, `smoothed` or `foreground`; its default runs all groups. The wrappers reconstruct required paths in temporary directories and write new results under the specified output directory.

The release's small-array check passed 15 tests and the original nearest-mesh geometry smoke check in the recorded review environment; see [the test record](tests/VERIFICATION.json). The saved-results replay verified 204 input entries, all 32 Table 2 rows, the spectral ratio and all 17 saved covariance cases. Figure replay generated 35 files; see [the release validation](data/derived/release_validation_20261029.json). These checks do not perform a production scan.

Do not run every file under `source/`: several original entry points create directories, write results or start production calculations when imported. Machine-specific path structure is retained for auditability, with personal home-path prefixes anonymized; it needs configuration before a full production run. The figure replay input ZIP contains numerical CSV inputs, JSON metadata and original plotting scripts, not raw simulation catalogues. The numerical data are preserved; seven JSON metadata files have local path prefixes anonymized. The maps record both original and release hashes for changed files.

For the manuscript, open `manuscript/main.tex` with **XeLaTeX** and BibTeX. `manuscript/supplement.tex` is a separate compilation target. The existing preamble contains `\XeTeXtracingfonts=0`, so an unchanged pdfLaTeX build is not supported. The MNRAS class and bibliography style are included with their original attribution. A local or Overleaf TeX environment must supply the other standard packages, including `newtxtext` and `newtxmath`.

The packaged main manuscript compiled successfully with the desktop editor's compiler on 29 October 2026. This verifies the local package, not the settings or uploaded files in a remote Overleaf project.

## Results to cross-check

- Table 2 uses all four redshifts, with changes measured relative to `z = 0.10235`.
- The 64-block jackknife covariance uses `63/64` times the centred outer-product sum. OAS regularization restores this variance scale; multiplying the result by 63 again would be wrong.
- The four main OAS coefficients correspond to **foreground treatments**, not four independent redshift fits. Each is fitted to a 123-component vector containing three redshift differences at 41 thresholds.
- At `z = 0.10235`, `R = 20 h^-1 Mpc`, the measured horizontal-cut/noFG genus-amplitude ratio is `4.578820649`; the anisotropic Gaussian spectral-moment estimate is `4.474572525`. Their agreement is approximate, not exact.
- Parameter-region areas depend on the mesh, Fourier filter, interpolation, smoothing and finite scan domain. The current curves are not independently calibrated cosmological confidence regions.

## Review status and reuse

This is a public scientific-review repository for the manuscript, code and selected derived products. No project-wide open-source licence has been selected, and inclusion does not grant new rights to the underlying simulation catalogues or third-party components. Preserve existing attributions. Licensing and the final data-availability wording remain separate author decisions.

The source and saved scientific arrays were not changed to remove manuscript TODOs or to make checks appear successful. See [the limits record](docs/REPRODUCIBILITY_LIMITS.md) before interpreting a passing software check as scientific validation.
