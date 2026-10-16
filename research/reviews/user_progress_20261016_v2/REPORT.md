# Paper and benchmark progress

The compiled eleven-page AAAI draft and Overleaf working bundle are available. The paper is not submission-ready. The required comparisons are incomplete, and top-1 performance is not established.

| European development pipeline | Present182 span accuracy, mean ± simulation SD (%) | Boundary F1 at 1 kb (%) |
|---|---:|---:|
| Our H050 stable-leaf memory | 97.0778 ± 0.5674 | 18.3807 |
| Same-budget continued training | 97.0000 ± 0.4978 | 19.0488 |
| Current author ARGMix release | 94.4461 ± 1.1491 | 11.7372 |
| Released AncestralPaths, derived C4 endpoint | 93.7830 ± 0.8385 | 7.0281 |

These are eight reused European development simulations, one optimizer seed. Inputs, objectives and training schedules differ. The 2.6317 pp development pipeline gap to ARGMix is not an architecture-only gain or a reproduction of published tables. The memory variant adds only 0.0778 pp over continued training and worsens boundaries. The higher-order candidate failed its registered utility rules.

Native FLARE scored 98.0100% focal accuracy and 97.7151% macro F1 on one full-30 Mb American VAL parent, with the 10,000 fixed common focal records. There is no between-parent SD or completed GNN/native ARGformer head-to-head score yet.

C3 is training on physical GPU 0 after all 20 TRAIN observation banks completed. At least 197/600 optimizer updates are verified, including 97 after retained checkpoint 100. Latest progress receipts do not assert a new durable optimizer checkpoint. No American GNN ancestry score.

Full VAL002 native paths and VAL003 simulation completed. VAL003 inference/native paths and VAL004 preparation are pending. Native ARGformer still has no completed American ancestry score.

First full prospective 1,611-diploid TRAIN generation failed after 120.54 seconds because the storage boundary check detected scope stat/membership changes. Actual outer exit1, child wait-15 and owned drain recorded. No completed simulation, fit or ancestry score. Failed artifacts and full reservation retained, no automatic retry.

Remaining requirements: the full American common-data comparison, the larger ARGMix panel, untouched European outer tests, a useful controlled mechanism gain with defensible novelty, stronger biological corroboration, and final independent review. The completed TGB experiment did not show higher-order benefit. The Danish application is descriptive and its independent genotype contrasts did not support the prespecified directions.
