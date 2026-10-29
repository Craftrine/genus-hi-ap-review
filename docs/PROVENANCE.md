# Provenance and attribution

This review snapshot is assembled from the local genus–AP project as read on 29 October 2026. The manuscript baseline is `paper/english_revision_v0.21_kcut_jackknife_20261029/`. `CODE_MAP.csv` and `DATA_MAP.csv` record the actual included source locations and original/release checksums. Copy integrity establishes what was included; it does not prove that every historical computation used the current on-disk source revision.

Nine source copies have their original personal home-path prefix replaced with `/Users/LOCAL_USER`; seven archived JSON metadata files have local root paths anonymized. The maps identify each transformation and retain the original checksum alongside the release checksum. No original project file, numerical CSV, scientific array, or measurement algorithm was changed by this packaging step. These substitutions do not make old absolute-path drivers portable.

## Measurement chain

The relevant main chain is valid-object filtering and H I mass selection; simulation-cosmology plane-parallel RSD; AP coordinate and box rescaling; complete-box nearest mesh geometry; mass-conserving open-boundary CIC; centred zero padding and CIC-window correction; Gaussian smoothing and scan-coordinate hard foreground masks; volume-fraction thresholds; negative open-boundary Euler characteristic per volume; fixed-field block contributions; intrinsic redshift differences; standardized OAS covariance; quadratic response; and saved-surface parameter-region postprocessing.

| Original project location | Role and distinction |
| --- | --- |
| `run_top_hi_weighted_all13_audit.py` | Valid selection and H I mass ordering; does not identify the upstream H I assignment model. |
| `full_hi_pipeline/scan_sim_nearest_mesh.py` | Complete AP box and actual per-axis cell spacing. |
| `full_hi_pipeline/scan_sim_fixed_cell.py` | Shared RSD/AP, CIC, masks and Euler functions. Its cropped geometry is historical and is not the current main geometry. |
| `full_hi_pipeline/fast_counting.py` | Threshold rank encoding and counting support. |
| `paper_figures/fig__cosmo_redshift_evolution/matched_stencil_source.py` | Matched local cubical-element block attribution used by the measurement chain. |
| `outputs/nearest_mesh_figure_refresh_20260910/measure.py` and `refresh.py` | Saved main curves, matched errors and reference covariance. The refresh script has file-writing operations at module scope. |
| `external_genus_workbuddy/dense_band_20260917/run_dense_fast.py` | Seven-branch stage; includes both main A and diagnostic B variants. |
| `outputs/m_turnover_pilot_20260923/` and `outputs/m_turnover_full_20260923/` | Additional wedge stages; do not equate a nine-slope stage with the final eleven-slope result. |
| `outputs/revision_v092_20260930_stage2b/source_scripts/` | Archived eleven-slope postprocessing evidence. |
| `paper/english_revision_v0.19_response_focus_20261020/figure_source/` | Source for unchanged v0.21 presentation assets. Parameter-surface preparation and pure figure replay are different operations. |
| `notes/kcut_jackknife_review_20261029/` | Saved-block arithmetic audit and the v0.21 method clarification. |
| `notes/hcut_amplitude_fourier_20261029/` | Determinant and amplitude-ratio check from saved spectral matrices. |

Some source paths name an external sibling project. In this package, `external_genus_workbuddy/` denotes archived source material, not a network dependency. Consult the code map for the exact files present.

## Figures, numerical evidence and references

Stable `fig__` asset names are retained, including the original `fig__noFG_hcut_msan_comp` spelling. The eight historical official figure identities are distinct from the manuscript's numbered main/supplementary figures and auxiliary renderings. The review package does not create new official figures or rename the registered assets.

The LaTeX references in `manuscript/references.bib` are inherited from v0.21. Where reference registers are included with derived evidence, their stated verification level and formula conventions apply. Their presence is not a new full-text literature verification. No external literature search was performed to create this provenance record.

The suite citation used in the manuscript is Han et al. (2025), *The Jiutian simulations for the CSST extra-galactic surveys*, DOI `10.1007/s11433-025-2712-1`, arXiv `2503.21368` (BibTeX key `Jiutian2025Overview`). The source register supports using this as a simulation-suite description; it does not settle the local H I product provenance. The original `references.bib` retains the bibliographic author list rather than substituting a project author.

## Third-party files and licence status

`manuscript/mnras.cls` and `manuscript/mnras.bst` retain their original copyright, author and licence headers. They state the LaTeX Project Public License, version 1.3 or later; the class identifies the Royal Astronomical Society as its current maintainer. These notices are not overwritten with Wells and are not relicensed by this package.

No top-level open-source licence was supplied for the project analysis code at the time of assembly. This package therefore does not select MIT, BSD, GPL, or another project-wide licence. Software dependencies retain their own licences; listing or importing a package does not change them. The simulation's access and redistribution conditions remain separate from the right to review derived results.

Existing manuscript author fields and frozen figure metadata are preserved. Newly written review documentation is prepared for Wells; this is not a determination of the eventual paper author list, ownership, or third-party authorship.

## Deliberate exclusions and disclosure boundary

The package is assembled by an explicit file list rather than by uploading the working directory or its old Git history. Unrelated exam material, browser/session credentials, local configuration, personal conversation exports, full third-party papers, raw simulation catalogues and large field caches are outside its intended scope. Source and evidence may retain anonymized local-path structures for provenance. Those paths are not portable execution instructions.

A local content check and successful package creation do not establish a GitHub upload or project visibility. The actual remote URL, commit and access level must be reported from the hosting service after an upload succeeds.
