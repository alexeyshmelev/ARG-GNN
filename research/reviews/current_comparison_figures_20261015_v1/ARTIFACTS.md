# Frozen figure packet

- present182_confusion_mean_sd.png / .svg / .pdf: three true-row-normalized confusion means with sample SD, class order ANA/CHG/WHG/EHG.
- present182_per_class_mean_sd.png / .svg / .pdf: per-parent precision, recall and F1 points plus equal-parent means and sample SD.
- population_accuracy_mean_sd.png / .svg / .pdf: seven cohorts, native ARGMix, derived C4 AP, H045 foreign_episode and separate H047 original/balanced arms.
- cohort_score_table.md: concise all-cohort accuracy/SD table and complete common NLL, macro F1, boundary F1, Brier and ECE table.
- cohort_scores.json: exact common metric means, sample SD, support, source states and hashes.
- measured_evidence.json: exact original per-parent confusion counts, normalized matrices, means, sample SD, per-class metrics and support.
- CONCLUSIONS.md: supported descriptive error patterns and limits.
- evidence_spec.json and plot_comparisons.py: pinned inputs and reproducible source.
- verification.json, reproducibility_check.json and pdf_render_check.json: arithmetic/provenance checks, exact repeated exports and visual QA.

Reproduce only on anogena, from /workspace-SR003.nfs2/ARG-GNN:

    . scripts/project_env.sh
    "$ARGGNN_PYTHON" -B research/reviews/current_comparison_figures_20261015_v1/plot_comparisons.py

331 arithmetic/provenance checks passed. All13 figure/evidence/table outputs reproduced with identical SHA256 hashes in a repeat of the final script. Each of the three single-page PDFs was rendered with Poppler and visually inspected; current PDF hashes match those inspected renders. No clipping, overlap or missing glyphs was observed. PNG and vector outputs are publication-ready source assets, with SVG text preserved.

Only six pinned completed metric JSON inputs are opened by the plotting source. It does not open truth, genotype, prediction arrays, models or checkpoint payloads. No GPU, guard/admission action, global process inventory, Git action or README edit was performed. Root owns publication and README embedding.
