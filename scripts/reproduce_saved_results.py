"""Verify saved manuscript numbers and replay the archived covariance audit.

Author: Wells. No density fields, FFTs, genus measurements or parameter scans run.
All generated files are written below --output-dir, never over archived inputs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/derived"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def determinant(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def verify_manifest():
    rows = read_csv(ROOT / "DATA_MAP.csv")
    archives = {}
    try:
        for row in rows:
            path = ROOT / row["release_path"]
            member = row["archive_member"]
            if member:
                if path not in archives:
                    archives[path] = zipfile.ZipFile(path)
                contents = archives[path].read(member)
            else:
                contents = path.read_bytes()
            assert len(contents) == int(row.get("release_bytes", row["bytes"])), row
            assert hashlib.sha256(contents).hexdigest() == row.get("release_sha256", row["sha256"]), row
    finally:
        for archive in archives.values():
            archive.close()
    return len(rows)


def table_two():
    amplitudes = read_csv(DATA / "manuscript_tables/sample_amplitudes.csv")
    table = read_csv(DATA / "manuscript_tables/table2_all_redshifts.csv")
    lookup = {(r["sample"], r["branch"], float(r["z"])): float(r["AR3"])
              for r in amplitudes}
    errors = []
    for row in table:
        a = lookup[(row["sample"], row["branch"], float(row["z"]))]
        ref = lookup[(row["sample"], row["branch"], float(row["reference_z"]))]
        assert a == float(row["AR3"]) and ref == float(row["reference_AR3"])
        expected = 100 * (a / ref - 1)
        errors.append(abs(expected - float(row["change_percent"])))
    assert len(table) == 32 and max(errors) < 1e-11
    return {"rows": len(table), "redshifts": sorted({float(r["z"]) for r in table}),
            "maximum_percentage_point_error": max(errors)}


def spectral_amplitudes():
    import numpy as np
    saved = json.loads((DATA / "spectral/verification.json").read_text())
    original = {r["branch"]: r for r in read_csv(DATA / "spectral/comparison.csv")
                if int(r["snapshot"]) == 118 and float(r["R"]) == 20
                and r["branch"] in ("nofg", "horizontal_only")}
    values = {}
    with np.load(DATA / "spectral/snap118_arrays.npz") as arr:
        for branch in ("nofg", "horizontal_only"):
            matrix = arr[f"R20_{branch}_M_padded"]
            np.testing.assert_array_equal(matrix, np.array(saved["matrices"][branch]))
            a = math.sqrt(determinant(matrix)) * 20**3 / (4 * math.pi**2)
            assert math.isclose(a, float(original[branch]["spectral_padded_AR3"]), rel_tol=1e-12)
            values[branch] = a
    ratio = values["horizontal_only"] / values["nofg"]
    measured = (float(original["horizontal_only"]["topology_gaussian_AR3"])
                / float(original["nofg"]["topology_gaussian_AR3"]))
    return {"snapshot": 118, "z": .10235, "R": 20,
            "anisotropic_gaussian_spectral_ratio": ratio,
            "measured_template_ratio": measured,
            "relative_difference_percent": 100 * (ratio / measured - 1),
            "interpretation": "Approximate explanation, not an exact match or boundary correction."}


def covariance_replay(destination):
    """Run the unchanged historical audit in a disposable reconstructed tree."""
    jdir = DATA / "jackknife"
    relative_script = Path("notes/kcut_jackknife_review_20261029/audit_saved_statistics.py")
    source_files = [relative_script, Path("full_hi_pipeline/fast_counting.py"),
                    Path("full_hi_pipeline/scan_sim_block_statistics.py"),
                    Path("paper_figures/fig__cosmo_redshift_evolution/matched_stencil_source.py")]
    with tempfile.TemporaryDirectory(prefix="genus_saved_audit_") as td:
        stage = Path(td)
        shutil.copytree(jdir / "inputs", stage, dirs_exist_ok=True)
        for rel in source_files:
            target = stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "source" / rel, target)
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, str(stage / relative_script)],
                                cwd=stage, env=env, capture_output=True, text=True)
        (destination / "jackknife_replay.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("Saved covariance audit failed; see jackknife_replay.log")
        report = stage / relative_script.parent / "AUDIT_RESULTS.json"
        fresh = json.loads(report.read_text())
        baseline = json.loads((jdir / "AUDIT_RESULTS.json").read_text())
        for key in ("small_counting_tests", "curve_error_entries", "difference_error_entries",
                    "amplitude_errors_checked", "reference_cases_replayed", "unique_wedge_slopes"):
            assert fresh[key] == baseline[key], key
        for new, old in zip(fresh["covariances"], baseline["covariances"]):
            assert math.isclose(new["rho"], old["rho"], rel_tol=1e-10, abs_tol=1e-12)
        shutil.copy2(report, destination / "jackknife_replay.json")
        return {k: v for k, v in fresh.items() if k not in ("fractions", "covariances")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "review_outputs/saved_checks")
    parser.add_argument("--skip-jackknife", action="store_true",
                        help="Only verify input hashes, Table 2 and spectral amplitude ratios.")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    for protected in (ROOT / "data", ROOT / "source", ROOT / "manuscript"):
        if out == protected or protected in out.parents:
            raise ValueError("Output must not be within archived source, data or manuscript.")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"author": "Wells", "new_field_measurements": 0,
               "verified_manifest_entries": verify_manifest(),
               "table_2": table_two(), "spectral_amplitudes": spectral_amplitudes()}
    if not args.skip_jackknife:
        summary["jackknife"] = covariance_replay(out)
    summary["scope"] = "Arithmetic and saved-data consistency only; not independent statistical calibration."
    (out / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
