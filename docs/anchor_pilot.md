# Fixed-anchor ancestry pilot and source receipts

The fixed-anchor prototype is compared on the existing paired inferred 64-query train/validation cohort. It has 16 independent training parents and four validation parents, eight exact tiles per parent, 64 query haplotypes, 24 observed reference haplotypes and all 2,560 original fixed coordinates retained. This is exploratory architecture selection. No final test was generated or scored.

The frozen plain study is configs/anchor_inferred_cohort64_plain_pilot_v1.json. It uses 32 fixed equal-width physical midpoint anchors, hidden32, sourceGNNdepth2, latent8, three repeated plain branches with initial coupling rates [5,20,80] and damping1. Full, scalar, raw diagonal and gain-matched diagonal modes are compared at depth1 and depth2. A depth2 log/softplus rate comparison has identical initial outputs and schemas. Rate/mixture parameters receive zero weight decay for all models.

Scalar propagation is an ordinary temporal-kernel comparison and changes diagonal decay/gain as well as coupling. Raw diagonal preserves each full generator diagonal and changes its equilibrium gain. Gain-matched diagonal transforms both signal and ones anchor impulses by D K_full^-1, then propagates independently with D. This matches homogeneous continuous-forcing DC gain and infinite-domain integrated impulse gain. It does not match arbitrary finite-anchor messages or a changing-operator global equilibrium. Shared source encoding and instantaneous source mixing remain in these controls.

Age interfaces are separate sidecars from the unchanged observed boundary bank. age_dataset.py ties every cache to the verified plain tile ID/hash, exact observed source and preparation, code dependencies, safe envelope and artifact hash. No supervision is stored in the operator envelope. The label-counterfactual test flips every source ancestry label and verifies that all age operators are unchanged. Age caches use common plain normalization and matched scalar-age attenuation. Cache preparation is CPU-only.

train_anchor.py reuses exact parent-coordinate scattering from train_boundary.py. It verifies the source/boundary manifest and independent parent counts. It transfers prepared banks once, records the numerical, source-loader, metric, seed and resource dependencies, and rejects source drift during training. It retains external labels, per-parent probabilities, parameter counts, checkpoint hashes, selected validation NLL, runtime/memory, final effective rates/mixtures/raw parameters, and per-epoch preclip gradient norms and relative rate/damping movement.

Root schedules every GPU job through scripts/launch_gpu.py. Current prototype code is frozen for active training. All runtime artifacts stay inside the repository. A fresh output directory is required.

Example plain invocation after checking an eligible free device:

```bash
source scripts/project_env.sh
.runtime/env/bin/python scripts/launch_gpu.py --gpu 3 \
  --job anchor-inferred-cohort64-plain-v1-seed101 arggnn.train_anchor \
  --config configs/anchor_inferred_cohort64_plain_pilot_v1.json \
  --data data/boundary_inferred_cohort64_pilot_v1 \
  --output experiments/results/anchor_inferred_cohort64_plain_pilot_v1_seed101 \
  --seed 101
```

Age training additionally requires a sealed sidecar manifest and frozen configuration hashes, then passes --age-data. The age factorial config is created only after sidecar sealing. Observed-only qualification measured five adjacent graphs from each source, with measurable structural departures from scalar plain geometry and sparse strict350 reference grounding. That small correlated sample is a geometry check, not a panel estimate or ancestry result.

Verification before the final diagonal extension: the full integrated suite passed536 tests and16 subtests. The corrected anchor/runner/sidecar suite passed94 tests. A fresh-context independent audit then passed109 targeted tests and inspected actual inferred Q64 float32 forward/backward, readout independence and both new controls, with no blocking finding. Logs and audit report remain in .runtime/logs and research/reviews/anchor_integration_audit_v1.md. These checks establish implementation contracts rather than novelty or predictive advantage.
