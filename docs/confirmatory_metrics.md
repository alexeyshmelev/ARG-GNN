# Frozen-analysis metrics implementation

This pure CPU module prepares analysis tools for the proposed independent cohort. No final ancestry parent, prediction or label has been generated or opened by this work. These calculations alone do not establish a confirmatory experiment.

Primary NLL is calculated from float64 stable log-softmax of original uncalibrated logits. A separate historical 1e-12-floor score remains visible for compatibility. Every fixed class slot is preserved. Missing class denominators are null with support zero, while the historical all-class macro-F1 convention uses zero. Explicit nonoverlapping haplotype pairs define diploid dosage and preserve phase-swap invariance. Fifteen fixed bins preserve all observations, including certainty one.

paired_parent_report receives already aligned N-parent by S-checkpoint loss matrices. It averages fixed training-seed differences within parent, then treats only parents as independent units. It supplies paired t inference, a deterministic 10,000-whole-parent bootstrap, all per-parent/per-seed losses, achieved precision and the 0.010-nat practical comparison. It conditions on frozen training data and checkpoints. It does not treat seed cells as independent parents or replace individual predictions with a probability ensemble. Holm adjustment retains contrast order.

Physical boundary matching uses maximum-cardinality earliest-feasible one-to-one matching on sorted coordinates. Missed and extra boundaries remain counted. Localization distances describe those deterministic pairs and are not a minimum-distance assignment. No true boundaries gives undefined recall/F1 with an explicit false-boundary count, including when predictions are present.

Dataset admission, model/checkpoint/analysis locks, sealed predictions before label joining, dense-grid extraction, tract lengths, tile-edge grouping and provenance binding remain runner work. This module is not an executable final-evaluation freeze. Tests use synthetic arrays only.
