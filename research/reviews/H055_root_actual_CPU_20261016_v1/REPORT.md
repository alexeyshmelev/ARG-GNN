# H055 fixed Danish genotype corroboration

Completed on 2026-10-16T01:23:18.416040+00:00. Genuine CPU caller wait0 and owned process/group absence are recorded. The three fixed GRCh37 chr20/21/22 [20,30)Mb windows completed.

Both prespecified positive-direction hypotheses failed on this panel. All 14,432 metadata-eligible SNPs had the common 13-person GP support. No sample, direction, filter, window or proxy was changed after observing the outcome.

| Support | SNPs | f4(later,earlier;Sidelkino,Bar8) | f4(later,earlier;KK1,Bar8) |
|---|---:|---:|---:|
| 20 | 3,270 | -0.00019291 | -0.01166537 |
| 21 | 7,066 | -0.00432352 | -0.00381045 |
| 22 | 4,096 | -0.00025896 | -0.00035553 |
| Pooled | 14,432 | -0.00223403 | -0.00460966 |

The descriptive SD across the three chromosome means is 0.00236597, 0.00579581. These are chromosome-level descriptive spreads, not independent training-seed SDs or significance estimates.

All three leave-one-chromosome pooled values remain negative for both contrasts. Deleting NEO92 changes the pooled Sidelkino contrast to +0.00062950. The other nine subject deletions retain its negative sign. All ten subject deletions retain a negative KK1 contrast. Thus the Sidelkino sign is sensitive to an individual even though its three chromosome means are negative.

Independent recomputation from the 14,432 stored GP-dosage records reproduced the pooled means, per-chromosome means and all ten subject deletions. See [recomputation receipt](independent_stored_dosage_recomputation.actual_v1.json). No raw genotype rerequest or QC change was used.

These are genotype-affinity observations. They do not measure local ancestry accuracy, mixture fractions, causal historical change or a new biological discovery. The data are imputed and do not provide independent ancestry truth. The initial positive expectation was a prior hypothesis, not a known property of this exact cohort and support. The post-outcome archaeological/context audit is recorded separately.

[Descriptive plot](genotype_f4_fixed_windows.svg) · [Actual summary](../../../experiments/results/H055_Danish_genotype_corroboration_CPU_v1/summary.json) · [Actual caller terminal](actual_supervisor_tool_terminal.root_v1.json) · [Registered scientific protocol](../Danish_genotype_corroboration_source_20261016_v2/protocol.source_frozen.inactive.v2.json)

The unchanged producer retained about 2.4 MB of scientific products and transferred 1.197 GB of bounded compressed input to RAM. Runtime 462.4 seconds. CPU only. No neural model ran.

H056 separately freezes the existing H045 model for the same ten query people and nominal windows. It was registered before these H055 estimates were opened, after H055 computation had finished. It has no phase/model data or result yet. Existing H045 application covers seven of these ten people on chr22 only.

The post-outcome [biological context audit](../H055_actual_biological_interpretation_audit_20261016_v1/) confirms that the earlier cohort consists of Early Neolithic people after the initial Danish farmer arrival. The later contexts are Middle Neolithic B or Late Neolithic. This comparison addresses an earlier-to-later Neolithic contrast, not the initial arrival of farming. These dates alone do not establish the expected affinity direction.
