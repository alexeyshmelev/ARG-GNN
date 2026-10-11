# Research success criteria
Updated 2026-10-12 after the user's efficiency and benchmark steering.

The objective is a useful, distinct whole-tree GNN for local ancestry inference. Top-1 accuracy and architectural novelty are goals, not established results.

## Evidence required for a performance claim
- Compare native ARGMix, native ARGformer and released AncestralPaths with our method on identical independent biological parents, observed panels, genealogy inputs where compatible, query identities, physical coverage and declared ancestry ontology.
- Preserve author algorithms, training objectives, selection and postprocessing. Record input adapters and publication versus release differences. Report native endpoints alongside the derived common C4 endpoint.
- Keep original-paper data or simulation-recipe reproductions distinct from the shared full-panel 2 Mb development experiment. Missing original assets remain explicit fidelity gaps.
- Keep the eight outer parents out of tuning. Freeze the candidate and competitor protocol before evaluating them. Report all registered parents and coverage, with no replacement of difficult cases.
- GST concerns general graph collections. Evaluate its native tasks separately. It cannot supply a biological local ancestry score merely by appearing in a shared ranking.
- Introduce an additional benchmark only for a documented scientific question or evaluation flaw. Preserve the existing evaluation and disclose the change before results.

## Efficient learning
Use PyTorch and PyTorch Geometric for our experiments. Discovery uses one initialization seed and FP32 model arithmetic. Prepare observed whole-parent banks once, reuse them across updates and preserve full genomic history. Avoid repeating predictions over a scored grid during TRAIN when exact span aggregation provides the same loss.

Measure preparation, training, scheduled validation, inference and total wall time, peak CPU/GPU memory, completed updates and accuracy at fixed checkpoints. Compare time and memory to the achieved quality. High GPU utilization alone is not evidence of efficient learning.

H038 has 32 independent TRAIN parents, eight VAL parents and eight reserved outer parents. Each TRAIN pass covers all 32 parents. Maximum training is 30 passes and 960 whole-parent updates per arm. Validation occurs every five passes. Selection and stopping follow the frozen registration. The initial actual backward counts as training.

## Novelty
For each candidate, identify the closest published operator, its actual difference and the biological reason it could help. Require a meaningful ablation and a gain under fair conditions before claiming a useful new mechanism. Established message passing, cavity sweeps and GRU recurrence alone are not a novel architecture.

## Current factual position
The standalone PyG pilot reached 83.90% on two reused VAL simulations. The historical native ARGMix pipeline reached 85.55% with different genealogy inputs and training budgets. This does not establish superiority. The broader full-panel experiment has no measured model accuracy yet.
