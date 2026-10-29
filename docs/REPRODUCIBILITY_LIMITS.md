# Reproducibility limits

Prepared on 29 October 2026 for the v0.21 review package. This records the scope of the local evidence; it is not a publication or statistical-calibration certificate.

## Three different levels of reproduction

| Level | Supported scope | Boundary |
| --- | --- | --- |
| Saved-data checks | Check covariance algebra and saved block contributions; verify spectral-moment arithmetic; compare table values and plotted coordinates. | Reuses already measured arrays and does not independently generate the underlying fields. |
| Figure replay | Redraw the included figures from saved CSV/JSON and contour vertices. | Does not repeat the parameter scan or infer a new probability surface. PDF bytes can differ with fonts, Matplotlib or metadata even when numerical coordinates agree. |
| Full production rerun | Source explains selection, AP/RSD, density deposition, FFT filtering, genus, block statistics and scan postprocessing. | Requires authorized raw catalogues, machine paths, stage caches and a production environment not supplied by this package. It has not been demonstrated from scratch in this review release. |

## Data and input availability

The manuscript identifies Jiutian 1G and the four snapshots `118`, `112`, `105` and `94`, at redshifts `0.10235`, `0.19095`, `0.31970` and `0.59339`. Its selected main sample has one million H I-weighted objects per snapshot. The publication cited for the simulation suite does not identify the exact local H I catalogue version or assignment model. Those remain unresolved in v0.21.

Raw halo/galaxy catalogues, the complete simulation volume, full three-dimensional fields and an independent mock ensemble are excluded. Two-dimensional slices and block-contribution arrays are derived products, not substitutes for those raw inputs. A historical reference to a path in an original script or JSON record does not mean that the path exists in this repository.

The historical `frozen_external_genus` backend referenced by `full_hi_pipeline/genus_grid.py` is not included. It is a legacy compatibility dependency, not the backend used by the current nearest-mesh negative-Euler measurement. Selected tests and saved-data wrappers must be assessed by their actual imports rather than treating all archived entry points as independently runnable.

## Jackknife and OAS

The production calculation keeps the field and volume-fraction thresholds fixed and assigns each cubical element to the block containing its lowest-index anchor. It constructs additive leave-one-block vectors, not fields remeasured after deleting catalogue objects. For `B = 64`, the centred covariance is

```text
C_JK = (63/64) X.T @ X
```

Each foreground branch has a joint OAS fit to three paired redshift differences, each with 41 thresholds. Standardization and restoration of the original variances give, up to floating-point roundoff,

```text
C = (1 - lambda) C_JK + lambda diag(C_JK).
```

The four familiar coefficients (`0.74521494`, `0.53461306`, `0.53461306`, `0.55253085`) label noFG, horizontal only, weak wedge and strong wedge. They must not be reported as four per-redshift coefficients. Single-redshift curve-error bars use the unshrunk jackknife diagonal in this chain.

The existing 29 October audit replays 1,968 curve errors, 1,968 paired-difference errors, 32 amplitude errors and 17 covariance branches with overlap. Those results establish arithmetic consistency of those saved inputs. They do not show that the covariance captures all physical or observational uncertainty. A 250-cell axis is split into blocks of 62 or 63 cells, so the equal-block coefficient is approximate despite the use of the actual retained-volume fraction in each leave-one-block denominator. The saved weighted alternative changes diagonal standard deviations by about -0.35% to +0.42%; it was not adopted as a calibrated replacement.

The baseline source audit used scikit-learn 1.3.0. Its OAS implementation omits the `2/p` term from the original paper's expression. Cross-version checks must record their actual dependency versions rather than assume exact numerical identity.

## The approximately 4.6-fold amplitude increase

For the filtered Fourier field, the directional Gaussian prediction uses

```text
M_ij = sum_k(k_i k_j |delta_filtered(k)|^2) / sum_k(|delta_filtered(k)|^2)
A_G = sqrt(det(M)) / (4 pi^2).
```

The horizontal/noFG ratio from the saved matrices is `4.474572525`, versus `4.578820649` from the Gaussian-template fit to the measured topology (a difference of about -2.27675%). A retained-region spectral-derivative diagnostic gives `4.549629580`, but is not a rigorous boundary correction. The isotropic trace formula gives `12.0362` for this strongly anisotropic field and is not the applicable prediction. Rechecking a determinant is not a new FFT, genus measurement, or test of padding convergence.

## Parameter-response limitations

The main scheme uses complete AP-rescaled boxes with each integer mesh count chosen near the target spacing. The old cropped geometry still exists among shared functions and must not be mistaken for the main scheme. Fixed-mesh and tapered-filter runs are diagnostic alternatives with a different measurement definition.

The manuscript describes observed jumps from integer mesh changes and modes crossing hard Fourier boundaries. Fixed meshes and continuous weights suppress the particular jumps examined. These checks do not establish spatial convergence, independent cosmology recovery, calibrated coverage, an optimal foreground cut, or instrument-level forecast precision. Intrinsic templates and covariance estimates reuse the same simulation; external modes and model uncertainty are not fully captured by internal resampling.

## Unresolved manuscript edits preserved in v0.21

- The specific H I catalogue provenance and assignment model remain a TODO.
- The Data availability section remains a TODO rather than an asserted public repository statement.
- Section 5.4 retains the title “Cosmological Constraints”, although its text limits the interpretation to a pipeline response.
- Figure 9 still excludes the `m = 1.692` endpoint; its saved result remains in the supplementary table and another figure.
- The requested AP cutoff relation `k_cut,scan = k_cut,ref / alpha_parallel` is not explicitly included; the new derivation concerns frequency–distance–wavenumber conversion.
- The OAS numerical coefficients have not been added to the manuscript text.
- The 29 October spectral-moment calculation has not been incorporated into the result paragraph.
- Appendix B does not specify the padding size in v0.21. Original source uses at least 160 cells per side before fast-length rounding; this needs careful transcription, not a change to stored measurements.

The complete Table 2 and explicit jackknife normalization are present. This release preserves that state rather than representing the requested editorial list as fully completed.
