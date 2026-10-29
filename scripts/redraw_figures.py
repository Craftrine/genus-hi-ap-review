"""Replay archived figure scripts in a temporary directory. Author: Wells.

Uses saved CSV curves, errors, image pixels, contour vertices and area values.
Does not rerun measurements or integrate new parameter-region surfaces.
"""
from pathlib import Path
import argparse
import json
import runpy
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "review_outputs/figures")
    parser.add_argument("--only", choices=["all", "basic", "compact", "smoothed", "foreground"], default="all")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    for protected in (ROOT / "data", ROOT / "source", ROOT / "manuscript"):
        if out == protected or protected in out.parents:
            raise ValueError("Output must not overwrite archived inputs.")
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import matplotlib.backends.backend_pdf as backend_pdf
    original_save = Figure.savefig

    def save_with_author(fig, filename, *a, **kw):
        if str(filename).lower().endswith((".pdf", ".png")):
            metadata = dict(kw.pop("metadata", {}) or {})
            metadata["Author"] = "Wells"
            kw["metadata"] = metadata
        return original_save(fig, filename, *a, **kw)

    Figure.savefig = save_with_author
    original_pages = backend_pdf.PdfPages

    class AuthoredPages(original_pages):
        def __init__(self, filename, *a, **kw):
            metadata = dict(kw.pop("metadata", {}) or {})
            metadata["Author"] = "Wells"
            super().__init__(filename, *a, metadata=metadata, **kw)

    backend_pdf.PdfPages = AuthoredPages
    outputs = []
    with tempfile.TemporaryDirectory(prefix="genus_figure_replay_") as td:
        stage = Path(td).resolve()
        with zipfile.ZipFile(ROOT / "data/derived/figure_replay_inputs.zip") as archive:
            for info in archive.infolist():
                destination = (stage / info.filename).resolve()
                if stage not in destination.parents:
                    raise ValueError("Unsafe archive path")
            archive.extractall(stage)
        old = stage / "outputs/revision_v093_20260930_stage2b_fixes"
        current = stage / "paper/english_revision_v0.19_response_focus_20261020"
        (old / "qa").mkdir(exist_ok=True)
        (current / "figures").mkdir(exist_ok=True)
        jobs = [
            ("basic", old / "replot.py", ["--only", "foreground,redshifts,cosmologies,slices,covariance,evolution"], old / "figures"),
            ("smoothed", current / "figure_source/smoothed_results/redraw.py", [], current / "figure_source/smoothed_results/figures"),
            ("compact", current / "figure_source/compact_revision/redraw.py", [], current / "figure_source/compact_revision/figures"),
            ("foreground", current / "figure_source/fig__foreground_kspace/redraw.py", [], current / "figure_source/fig__foreground_kspace"),
        ]
        for name, script, argv, generated in jobs:
            if args.only not in ("all", name):
                continue
            sys.argv = [str(script), *argv]
            runpy.run_path(str(script), run_name="__main__")
            destination = out / name
            destination.mkdir(exist_ok=True)
            for source in sorted(generated.iterdir()):
                if source.suffix in (".pdf", ".png"):
                    shutil.copy2(source, destination / source.name)
                    outputs.append(str((destination / source.name).relative_to(out)))
            print("Replayed", name, flush=True)
    (out / "replay_manifest.json").write_text(json.dumps({
        "author": "Wells", "new_field_measurements": 0, "outputs": outputs,
        "note": "Saved v0.21 plot content is preserved, including the omitted m=1.692 point in the trend panel."
    }, indent=2) + "\n")
    print("Files generated:", len(outputs))


if __name__ == "__main__":
    main()
