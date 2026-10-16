# Paper and benchmark progress

Updated UTC: 2026-10-16T11:36:59.228434+00:00.

The eleven-page AAAI draft, equations, preliminary schematic and Overleaf bundle exist. The scientific evidence is incomplete. No overall top-1, established architecture novelty or submission-ready claim is supported.

| European shared development pipeline | Present182 span accuracy, mean ± simulation SD (%) |
|---|---:|
| Our stable-leaf memory | 97.0778 ± 0.5674 |
| Our matched same-budget continued training | 97.0000 ± 0.4978 |
| Current author ARGMix release | 94.4461 ± 1.1491 |
| Released AncestralPaths, derived four-source endpoint | 93.7830 ± 0.8385 |

These scores use eight reused 2 Mb VAL simulations, one training seed and 952 haplotypes / 686 queries. Methods share scored support, but their SNP inputs, training and checkpoint selection differ. This is a pipeline comparison, without architecture-only attribution or an exact published-table reproduction.

Stable-leaf memory adds 0.0778 percentage points over continued training, while mean boundary F1 falls from 19.0488% to 18.3807%. The higher-order triple-support candidate fails its registered utility rules. The large difference from ARGMix does not prove the added memory mechanism is useful.

| Required evidence | Status |
|---|---|
| European development and population-level metrics | Completed, with confusion maps, NLL, boundaries and calibration in the scoreboard |
| American ARGformer / FLARE / GNN common-data comparison | Incomplete. Native FLARE on one VAL parent: 98.0100% focal accuracy, 97.7151% macro F1 over 10,000 fixed focal records. GNN and native ARGformer ancestry scores remain unavailable |
| American GNN training | Running on physical GPU0, at least 224/600 completed updates and durable checkpoint200 |
| American held-out preparation | VAL002 native paths and its full30Mb GNN block plan completed. VAL003 and VAL004 full30Mb simulations completed, with inference and native paths still needed |
| Larger ARGMix condition | Numerical generation completed for1,611 diploids over35,178,458bp. Parent monitoring failed after child wait0. Saved products have passed a source/metadata audit, separate data acceptance pending. No training or comparison score |
| Independent European test and final uncertainty | Eight reserved outer simulations remain ungenerated. Final selected-model seed uncertainty is unfinished |
| Useful distinct architecture | Unproved. Current added-mechanism gain is small and boundary utility worsens |
| Cross-domain result | TGB completed, negative. Persistence exceeds all learned variants. Full GST comparison is unfinished |
| Real-data validation | Danish application and genotype analysis completed descriptively. No ancestry truth. Independent genotype contrasts did not support the registered directions |
| Final manuscript | Working11-page AAAI draft compiled. Final benchmark tables, final architecture figure and fresh manuscript review remain |

The next scientific milestones are complete American head-to-head predictions, accepted larger-panel data and a faithful ARGMix comparison, then an independently confirmed useful mechanism. More prose cannot substitute for that evidence.

[Current score details](../../current_scoreboard.md) · [Actual runtime and generation snapshot](snapshot.root_v1.json) · [Working paper](../../../paper/current_full_panel_working.pdf) · [Overleaf bundle](../../../paper/current_full_panel_overleaf_working.zip)
