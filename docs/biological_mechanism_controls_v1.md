# Biological mechanism controls v1

Status: source-only extension of H020, not executed on biological data. Root owns experiment registration, independent review, scheduling and integration. All pinned H020 v1 source/configuration/prepared outputs remain untouched. No biological bank/array, truth target, live fitted score or checkpoint was opened during implementation. No GPU, simulation, installation or Git operation ran.

## Frozen observation and execution contract

This module imports the unchanged biological_candidate definitions for observation joins, fixed lossless bank loading, external truth capabilities, exact probability NLL, prediction sealing, optimizer grouping, storage admission and actual GPU qualification. It supplies a separate seven-condition factory and fit coordinator. It does not monkeypatch the imported module, prepare another bank or invoke precompute/packing. There is no preparation CLI mode.

The reference is configs/biological_candidate_pilot_v1.json at SHA2564151e7e156db3f188512bcc738b43597407d7707bc77c420509165c2e9bfc2f1. Reused prepared manifests must bind that original config and its original source dictionary, rather than the extension's new dictionary. Root separately supplies factual extension config, base config, observed manifest, prepared manifest and complete storage snapshot hashes. Source drift is checked before inputs and at epoch boundaries/completion. Prediction seal records and selected checkpoints carry extension/base/observed/prepared/source identities. Checkpoint identities are verified again before reloading.

Every control consumes the same original fixed AnchorKernelBank, with lossless original age/cache fields and precomputed eigenpairs. The unchanged loader verifies all280 leaves/F47, dynamicsQ184, R96, exact inferred IDs, branch-excluded columns, original coordinates and mixed64 query flags. All120 actual query outputs are selected through the original output_indexes. Mixed64 remain unlabeled dynamic states and never contribute to training, VAL selection or metrics.

All original marginal-tree nodes/edges and the same engineered reference information remain available. Four pure-source class inputs are observed reference labels. No extra cohort, query truth, truth-tree geometry, reference summary, target-informed eligibility or target selection enters the extension.

Coordinates are original physical bp/2Mb in float64. Eight250kb tiles,32 equal-cell midpoint anchors, physical cell-width masses, coverage clipping and100bp retained-center score support remain unchanged. A full tile has7,812.5bp anchor cells. Age350/1400, strength3 and common plain full-boundary mean-diagonal normalization remain fixed. No local tile rescaling or unit conversion is introduced. Since coupling rates also condition reference injection, a coordinate/rate-only rescaling would change the model and is not a supported symmetry.

The plain BoundaryResponseGNN source encoder is independently trained in each condition with identical dimensions and source fields. References remain independently encoded observed graph-local sources. All conditions use width32/source depth2/latent8/S3 and initial coupling5/20/80, initial damping1 and log-positive raw rates. Plain branches in split/source controls retain the same three rate branches.

TRAIN mass integrates exact external C4 class span over bins and all120 query outputs. VAL truth admission follows an exclusively written and hashed prediction file. Raw probabilities retain zeros, no floor and no posthoc renormalization. Selection is minimum equal-parent mean present64 VAL raw-probability NLL, epoch cap30, patience8 and earliest strict minimum. Ancient56 remain secondary. AdamW lr0.001, other weight decay1e-4, zero coupling/damping/mixture decay and gradient clip1 are unchanged. Seeds101/202/303 and seeded per-epoch tile order are fixed.

## Conditions and identifying contrasts

Existing candidate_age_d2, candidate_plain_d2 and matched_gru_d2 are references in the frozen H020 study. This extension does not rerun them. All its conditions are new biological cells whose repeat reasons must be registered against H004/H005/H006/H020 before execution.

| Condition | Implementation | Parameters | Identifying comparison and limit |
| --- | --- | ---: | --- |
| bio_mc_v1_scalar_age_d2 | AnchorKernelModel matched_scalar_age/full/D2 |10,420| Versus age D2 tests relative age geometry beyond mean-diagonal attenuation. Both QQ and QR change, so this does not locate an effect in propagation. |
| bio_mc_v1_age_propagation_only_d2 | SplitAgeAnchorKernelModel propagation_only/full/D2 |10,420| Age QQ/eigenpairs with plain QR. Versus plain D2 and full age D2 localizes selected block effects conditionally. |
| bio_mc_v1_age_source_only_d2 | SplitAgeAnchorKernelModel source_only/full/D2 |10,420| Plain QQ/eigenpairs with age QR. Completes the plain/plain, age/plain, plain/age, age/age four-cell comparison with the H020 references. |
| bio_mc_v1_age_gainmatched_d2 | AnchorKernelModel age/gain_matched_diagonal/D2 |10,420| Versus age/full D2 removes transfer of prior query state and retains instantaneous D K^-1 source mixing. It matches homogeneous continuous-forcing DC gain only. |
| bio_mc_v1_noncontext_age_d2 | NoncontextualAnchorKernelModel age/full/D2 |10,420| Projects original independent query anchor embeddings at every layer. Decoder residual context and two ordered kernels remain. Tests contextual source feedback rather than all memory. |
| bio_mc_v1_age_d1 | AnchorKernelModel age/full/D1 |8,188| One-kernel biological depth reference. D1/D2 alone confounds a second kernel with2,232 added parameters and another nonlinear update. |
| bio_mc_v1_age_d1_plus_pointwise | OneKernelPointwiseModel |10,420| One ordered age QQ action plus one active query-local residual block. Versus age D2 tests a second ordered QQ action against this specific additional local-depth alternative. |

Equal counts/shapes and active gradients do not establish equal function class, optimization difficulty, initialization semantics or damping meaning. No contrast alone rules out every capacity explanation. The two split QQ/QR pairs need not be one physical Kron response, so their interpretation is operator-block attribution. At nonlinear depth, changing either block changes later query sources. No isolated biological pathway or physical mixed-tree response is claimed.

## Matching source and ones systems

The existing kernel controls use S=q-lambda*CQR*r and m=1-lambda*CQR*1. Every QR intervention selects exactly the same block for numerator source and matching ones mass. The selected QQ eigenpairs supply K=lambda*CQQ+d*I.

The gain-matched control sends the bundled numerator AND ones sources through D*K_full^-1 at each anchor, with D=diag(K_full), before independent diagonal propagation. Source mixing is retained. It does not match finite anchor transients, normalized messages, spectra, finite-tile boundaries or changing-operator equilibria. In general exp(-Dt)D*K^-1*S differs from exp(-Kt)S. No diagonal model is renamed spatially independent in this documentation. A raw diagonal biological arm and finite-transient/source-held diagnostic remain outside this small block and are needed for stronger coupling attribution.

The noncontextual control retains fixed reference sources, branch states, chronological actions, gates and repeated target updates. Its D1 form is weight-copied identical to the ordinary D1 candidate and is deliberately omitted.

## One-kernel plus active pointwise residual

OneKernelPointwiseModel inherits the exact D2 learned tensor schema. It calls layer0.messages once to obtain the same first kernel as the age D1 model. It uses layer1 only as a coordinate-local reference-conditioned update, with no QQ multiplication, spectral propagation, impulse, scan, global source-bank feedback or aggregation between coordinates.

At an anchor or readout x in observed graph g, let U(x) be the first-kernel updated query embedding and R_g the original independently encoded reference embeddings. The second block computes, for each branch/channel:

~~~text
q = tanh(Wq U + bq)
r = tanh(Wr Rg + br)
S_s = q - lambda_s CQR_g,s r
m_s = 1 - lambda_s CQR_g,s 1
a_s = d_s / (1 + d_s)
M = sum_s pi_s a_s (S_s / m_s)
U_next = LayerNorm(U + sigmoid(G[U,M]) Wm M)
~~~

Signal and ones mass use the same unmodified age QR. The positive d values are bounded pointwise gains here. The denominator's1 is the fixed reference rate in the same canonical bp/2Mb rate unit as d. They are not temporal damping, inactive tensors or padding. Per-branch normalization occurs before mixture in this second block, unlike the first kernel's mixture of propagated numerator/mass. The layer also adds direct reference injection from a readout's containing graph. This uses only the common observed bank but changes how local reference evidence is used. All these differences are disclosed.

Each added layer has q projection264, reference projection264, message projection256, gate1,312, LayerNorm64 and three24-element coupling/damping/mixture groups, totaling2,232 parameters. Together with the age D1 count8,188, the comparator has10,420, exactly matching age D2 schemas and tensor shapes. All added tensors and groups have finite nonzero gradients on the nondegenerate handmade test. That verifies participation, not global identifiability or universal nonzero gradients.

The pointwise function requires finite positive rates, finite features, finite nonpositive QR and positive finite mass. It rejects nonfinite computations explicitly with no floor or clipping. The QR sign rejection is strict and may reject a cache containing accepted positive roundoff that the frozen preparation tolerances admitted. No real bank has been opened to assess that case. References are guaranteed nonempty by the biological R96 contract. The pointwise normalized branch is a convex combination of bounded q/r signals, and d/(1+d) plus the nonnegative mixture keeps the second message bounded by1 in exact arithmetic. Its residual embedding/head is not thereby bounded or a PSD kernel.

This comparator was implemented directly and semantically qualified. No parameter deletion, reassignment, surgery or attached unused padding was used. It remains an explicit architectural control whose second action differs from the candidate.

## Fresh outputs, storage and GPU admission

Only a fresh direct child of results named biological_mechanism_controls_v1_* is admitted. Existing directories, nested paths, v1 candidate outputs and data/preparation namespaces are rejected. Predictions use exclusive file creation. Failed/partial extension outputs are preserved with runner_failure.json and a failed receipt when available.

Storage reuses the exact AccountedStorage guard. Root supplies one hashed successful complete project-root du snapshot with timezone-aware completion. The extension reserves96MB per VAL parent/epoch/condition plus12GB fixed margin. For seven conditions/two VAL parents/30 epochs, the conservative reservation is52.32GB, plus50GB concurrent-write reserve and current root usage below the strict450GB admission ceiling. Direct written bytes are accounted with cheap free-space checks and no recursive per-epoch scan. This is an admission estimate, not measured live storage.

Both script and module CUDA entry apply the inherited real root lease and exact title AI Auto Research Pipeline Test before Torch/model import. The unchanged configure_fit applies deterministic policy before CUDA, verifies one visible device, current driver-context UUID/own process title, lease/property consistency and actual NVTOP own compute PID. Lease/title are rechecked through fitting/completion. At most two project jobs on allowed physical2-5 remain root-owned. This handoff performs no CUDA access or mock GPU admission.

Example root commands after independent review and registration, from the dedicated repository:

~~~sh
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" scripts/train_biological_mechanism_controls_v1.py --check-source-only \
  --expected-config-sha256 EXTENSION_CFG_SHA \
  --expected-base-config-sha256 4151e7e156db3f188512bcc738b43597407d7707bc77c420509165c2e9bfc2f1
~~~

~~~sh
.runtime/env/bin/python scripts/launch_gpu.py --gpu 3 --job biological-mechanism-controls-v1-seed101 \
  arggnn.biological_mechanism_controls_v1 --train --seed 101 --device cuda \
  --expected-config-sha256 EXTENSION_CFG_SHA \
  --expected-base-config-sha256 4151e7e156db3f188512bcc738b43597407d7707bc77c420509165c2e9bfc2f1 \
  --expected-observed-manifest-sha256 RELATE_MANIFEST_SHA \
  --expected-prepared-manifest-sha256 PREPARED_MANIFEST_SHA \
  --storage-snapshot ROOT_STORAGE_SNAPSHOT \
  --expected-storage-snapshot-sha256 ROOT_STORAGE_SNAPSHOT_SHA
~~~

Root supplies factual hashes and qualifies storage at scheduling time. Optional explicit --device cpu requires empty CUDA visibility. There is no unbound/default actual-data fit or new extraction mode.

## Source qualification and remaining scientific work

The dedicated tests use small handmade observed graph/interface fixtures from the frozen mathematical suites, an independent SciPy dense action oracle, explicit scalar-loop pointwise equations, fake metadata and temporary source/config fixtures. They do not use biological arrays or actual truth. Coverage includes every selected numerator/ones/QR/QQ action, gain transform, split aliases/nonmutation, noncontextual feedback, parameter schemas/counts, all active gradients and one handmade optimizer step, both subdivision contracts, serialization/readout changes, exact first-kernel identity, no second kernel call, pointwise query/coordinate locality, source/config/output/prepared guards and pre-Torch lease ordering.

Independent dense actions and feedback comparisons use3e-11 float64 tolerance, with pointwise scalar equations at3e-12. Matching raw ones sources and noncontextual source identities are exact. The initial qualification log is preserved:45 passed and four tests failed because second-layer sources were compared bitwise after independent dense feedback, differing by at most1.11e-16. Only the test's source comparison tolerance changed to the existing3e-11 network tolerance. Production code did not change for those failures.

Actual final test counts, hashes, source-only CLI output and frozen-source reconciliation are recorded in research/reviews/biological_mechanism_controls_v1_source_audit/review_receipt.json. This is the author's source qualification, not an independent review, real-cache admission, biological fit or accuracy claim.

The four H020 parents have only two VAL parents. Outcomes from the eventual block remain descriptive development results. Independent parents, frozen contrasts and parent-level uncertainty are needed for confirmation. Seeds/sites/windows/trees are not independent biological replicates. Whole-tree versus nearest-reference restriction, anchor/grid/tile sensitivity, strict unordered-set controls, fair SSM/covariance/MP-SSM comparisons and reconstruction/demographic/reference perturbations remain separate planned work. The source-first block establishes no novelty or mechanism outcome.

## Handoff hashes

| Artifact | SHA256 |
| --- | --- |
| src/arggnn/biological_mechanism_controls_v1.py | 28f3e7d3f28a48eeee0944b2e4ad55e3ea6ebb314d8b0342bd785facb08dcbeb |
| scripts/train_biological_mechanism_controls_v1.py | f19bdead17a135a42ad70513f1522cef2a9fd58dc159d04f189adbe08de67dec |
| configs/biological_mechanism_controls_v1.json | 67d7aba454d3cc57b11faa32151dd6af397210fa89d9b56b5fcd0014f662f1b4 |
| tests/test_biological_mechanism_controls_v1.py | 07b990efa63230bb93affb4856d86e59e910d3278c9379f8ba8c90e0061294ee |

Final dedicated qualification: **49 passed in8.10 seconds**. Script and module source-only checks both passed with no observed/truth/bank access and no GPU.
