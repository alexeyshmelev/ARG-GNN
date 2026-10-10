# Biological development comparison

These are completed development results on the same two independent Relate-inferred validation parents. Each has 64 present-day query haplotypes scored on the exact original 100 bp-center clipped-bin physical support. Parents were split before tiling. The final holdout is unopened. Different optimizer seeds are repeated fits, not additional independent biological replicates.

| Method | Optimizer seeds | Accuracy | NLL |
| --- | --- | ---: | ---: |
| Current author ARGMix | 50 | 85.5537% | 0.435323 |
| Candidate age D2 | 101 | 84.8906% | 0.427252 |
| Candidate plain D2 | 101 | 84.8410% | 0.427359 |
| Matched GRU D2 | 101 | 84.1769% | 0.451988 |
| Released AncestralPaths Dense | 101, 202, 303 | 82.0603% | 0.819053 (stable logits) |
| GST-style width 32 | 101 | 80.0330% | 0.568897 |
| GST-style width 64 | 101 | 79.8554% | 0.569015 |

The age candidate gains 2.8302 percentage points over released AncestralPaths Dense and 4.8576 points over the reconstructed GST width-32 control in this pilot. It trails current author ARGMix by 0.6632 points. Candidate NLL is slightly lower than ARGMix, whereas its Brier score is slightly higher. Age and plain D2 differ by only 0.0495 accuracy points. These descriptive development differences do not establish statistical superiority or an age-geometry mechanism.

ARGMix uses the unchanged current author model, objective, native checkpoint selection and overlap/refinement post-processing. The April preprint code archive is unavailable. This is an explicit transfer of the current release to the biological development panel, not reproduction of the April reported numbers.

AncestralPaths uses the released TensorFlow Dense architecture and native cohort-shell features. Its sampler is explicitly capped by available source-path training rows. Six path probabilities are summed to four ancestry sources before argmax. The supplied source implements Dense, whereas the supplementary CNN description lacks an executable architecture. The finite NLL shown above is an auxiliary stable-logit readout with unchanged weights and argmax. Native probability underflow and infinite-NLL cells remain preserved.

GST is an independently reconstructed architectural control. Physical-span token weighting, gate initialization, full-tree feature access and per-query ARG readout are disclosed. Width 64 is a capacity sensitivity with the same selection rule. This result describes those controls on this task and does not reproduce the paper's author benchmark or demonstrate a universal advantage over GST.

Native ARGformer fitting and reference-corpus retrieval completed for the Relate condition, but its representation adequacy failed: the export has no reused connected internal ancestor IDs across trees. ARGformer's published inference family is tsinfer plus tsdate, which can share inferred ancestors. A separately versioned documented historical-sample input pipeline is undergoing independent qualification. Different upstream inference pipelines are end-to-end method comparisons. Model attribution requires identical qualified observed trees and separately retrained checkpoints. No invented cross-tree identity or reinterpretation of the existing checkpoint is allowed.

Sample haplotype IDs provide genuine within-panel correspondence across trees. Internal nodes represent inferred ancestors and can appear only on portions of the genome. Clade equality, tree-local integers and chronological order alone do not establish that two internal nodes are the same ancestor. Historical reference dates older than the census remain reference evidence, not target lineages at that census.

Two further candidate optimizer seeds and seven prespecified mechanism controls remain running across physical GPUs 2 and 3. The two GST fits have completed. The shared pool permits concurrent workers per device, records memory/storage admission and exact process titles, and naturally releases each device after its assigned jobs drain.

The generic diffusion/covariance kernel uses inherited mathematics. A source-only census-cut and clade-turnover proposal is separately documented and unimplemented. Neither kernel novelty nor strongest-baseline superiority is established. Further architecture changes require new source/gradient qualification and matched falsification controls.

Compact scores and exact source hashes: [comparison receipt](../experiments/results/biological_comparison_status_v1/summary.json).
Individual result receipts: [candidate](../experiments/results/biological_candidate_pilot_v1_seed101_summary/summary.json), [ARGMix](../experiments/results/official_argmix_biological_metadata_repair_v3_summary/summary.json), [AncestralPaths stable readout](../experiments/results/ancestralpaths_stable_readout_v1/summary.json), [GST controls](../experiments/results/biological_gst_control_v1_seed101_summary/summary.json).
