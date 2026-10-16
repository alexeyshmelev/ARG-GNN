# Existing European development confusion diagnostic

Rendered from saved selected metrics. No model fitting, prediction generation, target joining or checkpoint selection was performed.

Present182, eight reused validation simulations, 2 Mb. Normalize each true-class row within each simulation, then equally average eight simulations. All methods have identical per-parent scored support and true-class denominators.

[Confusion matrices](present_day_confusion.png) · [Paired recall differences](paired_class_recall.png) · [Values and source hashes](metrics.json)

| True ancestry | H045 recall % | ARGMix recall % | AP C4 recall % | Paired gain, pp ± SD |
|---|---:|---:|---:|---:|
| ANA | 97.198 | 94.087 | 95.830 | +3.111 ± 0.933 |
| CHG | 95.930 | 93.499 | 90.708 | +2.431 ± 1.966 |
| WHG | 96.046 | 93.406 | 94.742 | +2.640 ± 1.644 |
| EHG | 97.607 | 96.403 | 93.721 | +1.203 ± 0.644 |

SD describes validation simulations, not training seeds. This does not reproduce original-paper tables or isolate architecture. ARGMix inputs, training budgets and model selection differ. AncestralPaths uses the released Dense model with six-path probabilities collapsed to four classes before argmax. H045 is the selected foreign-episode seed101/pass30 development checkpoint.

This diagnostic does not establish independent generalization, useful cross-tree novelty or performance against ARGformer/FLARE.
