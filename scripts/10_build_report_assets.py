"""Phase 10 - Collect every phase summary, build the dashboard and LaTeX tables.

Produces:
    reports/PIPELINE_STATUS.md    dashboard with the claim roll-up and the
                                  test-set audit trail
    paper/tables/*.tex            LaTeX tables generated from result JSON, so
                                  no number is ever retyped by hand
    paper/figs/                   every figure copied in for the LaTeX build

Run:  python scripts/11_build_report_assets.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from _bootstrap import setup

from src.utils.io import load_json, ensure_dir
from src.utils.summary import PhaseSummary, collect_summaries, render_pipeline_status
from src.utils.timing import Stopwatch

PHASE = "11_build_report_assets"


def escape_latex(text: str) -> str:
    """Escape the characters LaTeX would otherwise interpret."""
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = str(text).replace(old, new)
    return text


def latex_table(rows, columns, caption, label, column_names=None,
                column_format=None, notes=None) -> str:
    """Render a list of dicts as a booktabs LaTeX table."""
    if not rows:
        return f"% No data available for {label}\n"

    column_names = column_names or [c.replace("_", " ").title() for c in columns]
    column_format = column_format or ("l" + "r" * (len(columns) - 1))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(escape_latex(n) for n in column_names) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "--")
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            elif isinstance(value, bool):
                cells.append(r"\checkmark" if value else "--")
            elif value is None:
                cells.append("--")
            else:
                cells.append(escape_latex(value))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if notes:
        lines.append(rf"\begin{{flushleft}}\footnotesize {escape_latex(notes)}\end{{flushleft}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg, logger, args = setup(PHASE, "Build report assets and LaTeX tables")
    summary = PhaseSummary(PHASE, cfg, next_phase=None)
    watch = Stopwatch()

    reports_dir = Path(cfg["_abs_paths"]["reports_dir"])
    results_dir = Path(cfg["_abs_paths"]["results_dir"])
    figures_dir = Path(cfg["_abs_paths"]["figures_dir"])
    paper_dir = Path(cfg["_abs_paths"]["paper_dir"])
    tables_dir = ensure_dir(paper_dir / "tables")
    paper_figs = ensure_dir(paper_dir / "figs")

    # ---- dashboard ---------------------------------------------------------
    with watch.section("collect_summaries"):
        summaries = collect_summaries(reports_dir)
    logger.info(f"Collected {len(summaries)} phase summaries")

    status = render_pipeline_status(summaries)
    status_path = reports_dir / "PIPELINE_STATUS.md"
    status_path.write_text(status, encoding="utf-8")
    summary.add_artifact(status_path)
    logger.info(f"Dashboard written to {status_path}")

    # Re-run the test-set audit here so it appears in this phase too.
    test_readers = [s["phase"] for s in summaries
                    if "test" in (s.get("splits_touched") or [])]
    logger.info(f"Phases that read the test split: {test_readers}")
    summary.add_finding("phases_reading_test_split", test_readers)
    summary.add_assertion(
        "test_split_not_used_for_selection",
        all(not p.startswith(("01_", "02_", "03_", "04_", "05_")) or p.startswith("01_")
            for p in test_readers),
        "Phases 03-05 (clustering, classical training, CNN training) must not "
        f"appear in this list. Actual readers: {test_readers}",
    )

    # ---- LaTeX tables ------------------------------------------------------
    n_tables = 0

    test_results_path = results_dir / "evaluation" / "test_results.json"
    if test_results_path.exists():
        payload = load_json(test_results_path)

        rows = []
        for name, result in payload["models"].items():
            recording = result["recording"]
            ci = (result.get("bootstrap") or {}).get("macc", {})
            rows.append({
                "model": name.replace("_", " "),
                "accuracy": recording["accuracy"],
                "sensitivity": recording["sensitivity"],
                "specificity": recording["specificity"],
                "macc": recording["macc"],
                "macc_ci": (f"[{ci['lower']:.3f}, {ci['upper']:.3f}]"
                            if ci else "--"),
                "f1": recording["f1"],
                "auc": recording["roc_auc"],
            })
        rows.sort(key=lambda r: -r["macc"])
        (tables_dir / "table_results.tex").write_text(
            latex_table(
                rows,
                ["model", "accuracy", "sensitivity", "specificity", "macc", "macc_ci", "f1", "auc"],
                caption="Recording-level performance on the held-out test set. "
                        "MAcc = (Se + Sp)/2 is the primary metric; 95\\% bootstrap "
                        "confidence intervals are resampled over recordings.",
                label="tab:results",
                column_names=["Model", "Acc.", "Se.", "Sp.", "MAcc", "95\\% CI", "F1", "AUC"],
                column_format="lrrrrcrr",
            ), encoding="utf-8")
        n_tables += 1

        # Baselines
        baseline_rows = [
            {"baseline": name.replace("_", " "),
             "accuracy": m["accuracy"], "sensitivity": m["sensitivity"],
             "specificity": m["specificity"], "macc": m["macc"]}
            for name, m in payload.get("baselines", {}).items()
        ]
        (tables_dir / "table_baselines.tex").write_text(
            latex_table(
                baseline_rows, ["baseline", "accuracy", "sensitivity", "specificity", "macc"],
                caption="Trivial baselines on the test set. The always-Normal "
                        "predictor illustrates why accuracy is not an appropriate "
                        "headline metric for this dataset.",
                label="tab:baselines",
                column_names=["Baseline", "Acc.", "Se.", "Sp.", "MAcc"],
            ), encoding="utf-8")
        n_tables += 1

        # Pairwise significance
        mcnemar_rows = [
            {"comparison": key.replace("_", " "),
             "discordant": f"{r['a_only_correct']}/{r['b_only_correct']}",
             "p_value": r["p_value"],
             "significant": r.get("significant_at_0.05", False)}
            for key, r in payload.get("mcnemar", {}).items()
        ]
        (tables_dir / "table_significance.tex").write_text(
            latex_table(
                mcnemar_rows, ["comparison", "discordant", "p_value", "significant"],
                caption="Pairwise McNemar tests on recording-level predictions. "
                        "Discordant counts are (A correct / B correct).",
                label="tab:significance",
                column_names=["Comparison", "Discordant", "$p$", "Sig."],
                column_format="lccc",
                notes="Significance is assessed after Holm-Bonferroni correction "
                      "across the family of comparisons.",
            ), encoding="utf-8")
        n_tables += 1

        # Literature
        literature_rows = [
            {"system": r["system"], "macc": r.get("macc"),
             "sensitivity": r.get("sensitivity"), "specificity": r.get("specificity"),
             "comparable": r.get("comparable", False)}
            for r in payload.get("literature_comparison", [])
        ]
        (tables_dir / "table_literature.tex").write_text(
            latex_table(
                literature_rows, ["system", "macc", "sensitivity", "specificity", "comparable"],
                caption="Comparison with published PhysioNet/CinC 2016 entries.",
                label="tab:literature",
                column_names=["System", "MAcc", "Se.", "Sp.", "Comparable"],
                column_format="lrrrc",
                notes="Published entries were scored on the official hidden test set, "
                      "which was never released. Our numbers come from a held-out "
                      "split of the public training data, so the comparison is "
                      "contextual rather than like-for-like.",
            ), encoding="utf-8")
        n_tables += 1

        # Per-site
        site_rows = [r for r in payload.get("per_subdatabase", []) if "macc" in r]
        if site_rows:
            (tables_dir / "table_per_site.tex").write_text(
                latex_table(
                    site_rows, ["group", "n", "prevalence", "macc", "sensitivity", "specificity"],
                    caption="Best model's performance broken down by acquisition "
                            "sub-database. Uniform performance argues against the "
                            "model exploiting site-specific cues.",
                    label="tab:persite",
                    column_names=["Sub-database", "$n$", "Prev.", "MAcc", "Se.", "Sp."],
                ), encoding="utf-8")
            n_tables += 1

    # Clustering
    clustering_rows = []
    for domain in ["mfcc", "logmel", "pwp"]:
        path = results_dir / "clustering" / f"{domain}_clustering.json"
        if not path.exists():
            continue
        payload = load_json(path)
        k2 = next((r for r in payload["sweep"] if r["k"] == 2), None)
        if k2:
            clustering_rows.append({
                "domain": domain.upper(),
                "silhouette": k2.get("silhouette"),
                "davies_bouldin": k2.get("davies_bouldin"),
                "ari_class": k2.get("ari"),
                "ari_site": k2.get("ari_vs_site"),
                "pca95": payload["pca"]["n_components_for_95pct"],
            })
    if clustering_rows:
        (tables_dir / "table_clustering.tex").write_text(
            latex_table(
                clustering_rows,
                ["domain", "silhouette", "davies_bouldin", "ari_class", "ari_site", "pca95"],
                caption="Unsupervised structure of each feature space ($k=2$). "
                        "ARI against the diagnosis is compared with ARI against the "
                        "recording site to identify what the clusters actually track.",
                label="tab:clustering",
                column_names=["Features", "Silh.", "DB", "ARI (class)", "ARI (site)",
                              "PCA dims (95\\%)"],
            ), encoding="utf-8")
        n_tables += 1

    # XAI alignment
    alignment_path = results_dir / "xai" / "alignment.json"
    if alignment_path.exists():
        payload = load_json(alignment_path)
        rows = payload.get("table", [])
        if rows:
            columns = ["model"] + [c for c in rows[0] if c.startswith(("mass_", "E_"))]
            (tables_dir / "table_alignment.tex").write_text(
                latex_table(
                    rows, columns,
                    caption="Cardiac-cycle alignment of model attributions. "
                            "$E_s$ is the enrichment (attribution mass divided by "
                            "time fraction); $E_s = 1$ corresponds to uniform "
                            "temporal attention.",
                    label="tab:alignment",
                    column_names=[c.replace("_", " ") for c in columns],
                    notes=f"Computed on the "
                          f"{100 * rows[0].get('inclusion_rate', 0):.0f}\\% of test "
                          "windows the segmenter labelled with sufficient confidence.",
                ), encoding="utf-8")
            n_tables += 1

    # ---- copy figures ------------------------------------------------------
    n_figures = 0
    for pattern in ["*.pdf", "*.png"]:
        for source in figures_dir.glob(pattern):
            shutil.copy2(source, paper_figs / source.name)
            n_figures += 1
    supplementary = figures_dir / "supplementary"
    if supplementary.exists():
        target = ensure_dir(paper_figs / "supplementary")
        for source in supplementary.glob("*.pdf"):
            shutil.copy2(source, target / source.name)
            n_figures += 1
    logger.info(f"Copied {n_figures} figure files into {paper_figs}")
    summary.add_finding("n_figures_copied", n_figures)

    # ---- claim roll-up -----------------------------------------------------
    claims = {}
    for phase_summary in summaries:
        claims.update(phase_summary.get("claim_verdicts") or {})
    logger.info("")
    logger.info("Claim roll-up:")
    for claim_id, verdict in sorted(claims.items()):
        logger.info(f"  {claim_id}: {verdict['verdict']:14s} — {verdict['statement']}")
    summary.add_finding("claim_verdicts",
                        {k: v["verdict"] for k, v in sorted(claims.items())})

    unsupported = [k for k, v in claims.items()
                   if v["verdict"] in {"contradicted", "weak"}]
    if unsupported:
        summary.add_note(
            f"Claims {unsupported} are weak or contradicted. These must be stated "
            "as such in the paper. A contradicted claim reported honestly is a "
            "result; a contradicted claim quietly dropped is misconduct."
        )

    summary.set_timings(watch.as_dict())
    paths = summary.write(cfg["_abs_paths"]["reports_dir"])
    logger.info(f"Summary written to {paths['markdown']}")
    logger.info(f"Dashboard: {status_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
