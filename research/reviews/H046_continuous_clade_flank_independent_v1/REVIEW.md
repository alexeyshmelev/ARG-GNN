# H046 independent scientific and API source review

No concrete scientific or API blocker was found in the frozen model/worker sources below. This is a source review, not GPU admission, efficacy evidence, or a worst-case runtime-fit certificate.

Frozen model: src/arggnn/continuous_clade_flank_gnn_v1.py, SHA 2c81b74c59a532b46ed75f7fd656ea526ec06fd9bf62af35855ab291f5451e20.
Frozen worker: src/arggnn/train_continuous_clade_flank_v1.py, SHA eda198ef04192d978433c64f921b865aa89a9304adcc76893fa2852a14d72ae4.

## Scientific implementation

The two segmented affine scans integrate the four-channel field and a unit-mass channel with stable exp/expm1 weights. Their shifted exclusive states exclude the entire original focal native hold. The final distance factors preserve attenuation from each focal endpoint to the scored point. No division by foreign mass or subtraction of nearly equal full/focal pools is used. Segmentation prevents context across an absent clade; singleton episodes have zero foreign support.

Context comes from first-sweep proper-clade states before genomic recurrence. Exact sampled-descendant episode matching and deepest unary representatives reuse the H042/H045 primitives. One root-to-leaf map selects the deepest clade with at least two consecutive source holds. The complete original H042 learned flag adapter, two PyG sweeps, SNP encoder, genomic BiGRU, original trees and vertices are retained.

Both arms use the same four projected channels, foreign scan, masses, local observations and head capacity. The own arm replaces foreign values with mass times the focal state while retaining a zero-gradient foreign link. The new output projection starts at exact zero, the 377 added parameters yield 31,165 total, complete initial tensors match, and optimizers/body parameters are independent and trainable.

The original observation binding fixes focal intervals and readout membership. The actual proof is fixed-field/fixed-original-support/episode/position integration-mesh invariance. It does not extend to redefining source holds, recalculating the SNP field, adding BiGRU steps, ancestor identity, or chromosome reversal. The shared swapped-view head has only its stated supplied-view symmetry.

Local evidence uses original observed allele-cardinality/log-evidence primitives, reference-only frequencies, called query SNPs and a callable reference in every class. Centering is across four classes. Missing sides make the gate zero. The bounded correction and eligibility mask are explicit. FP32 evidence prefixes rebase every1024 unique integer SNPs; a <=1000bp half-open side has <=1000 sites and therefore spans at most two such blocks.

## Target, barrier and API lifecycle

TRAIN seals all32 observation recipes before its truth gate. Each parent constructs one complete original100bp-grid exact-span target mass through unchanged H044 original_mass_v1, shares it across the two sequential arm updates, and releases it before the next parent. Each arm performs one whole-parent backward and actual optimizer step. Actual callbacks count 960 steps per arm and preserve the fixed six-checkpoint schedule.

The H046 phase-reader and complete96-product RAM barrier classes match frozen H044 exactly after namespace substitution. All two arms x six checkpoints x eight original grids are sealed and verified before VAL truth. Selection and selected full diagnostics retain the inherited support/cohorts/boundary/short-tract machinery. No new optimization or outer access is present in evaluation.

The source materializes complete [P,686,4] logits and their differentiable graph. Query/center checkpoints bound head intermediates, but this is one-parent target streaming, not blockwise loss/backward streaming. CPU observations/caches, full logits/targets and the96-grid probability matrix remain distinct memory costs. Named tensor bounds and CPU synthetic checks do not prove a production GPU peak. Root must qualify actual scan/autograd/residency inside authorized continuous training and keep the actual resource/source closure.

## Reporting findings resolved before freeze

The first worker lacked the proposal's promised local-callability strata. The frozen worker now reports fixed min(n_left,n_right) strata 0,1–2,3–5,>=6 for selected Present182 models, using only original observed usable SNP counts for membership and the existing primary exact-span mass cache for support/accuracy/NLL. These diagnostics run after the complete96-product truth barrier. Empty support is explicit. Zero probability with positive mass retains infinite NLL, without a probability floor. Equal-parent summaries expose support counts and parent variability. No stratum-specific boundary-matching rule was invented.

Root clarified the prospective practical criterion before admission: equal-parent mean Present182 accuracy gain >=.001 and equal-parent mean boundary-F1 loss <=.01 across all eight VAL parents. The frozen worker requires and reports this aggregation and explicitly disclaims a worst-parent bound. Undefined parent boundary F1 cannot silently satisfy the rule. These are development criteria, not significance evidence.

## Validation and limits

The accompanying independent stdlib/source checks cover the inherited barriers, original bindings, strict endpoint/exclusion interpretation, local evidence/gate, matched control/head, original-grid lifecycle, reporting updates and literal scalar integral versus endpoint recurrence. A fixed-support neighboring-cell split preserves the scalar operator and its tau derivative in the independent mathematical oracle. This oracle is not a Torch execution or FP32 precision claim.

The author's inspected CPU receipts report34 actual FP32 model checks and28 worker smoke checks, including scan/state/tau gradients, finite differences, checkpoint equivalence, fixed-support refinement, actual complete952 two-sweep/SNP/BiGRU synthetic backward, same-grid initialization identity, missingness/multiallelic prefixes, one synthetic optimizer update per arm, barriers, reporting edge cases and noncontiguous selected-query/chunk integration. Those receipts match the frozen source/checker hashes. This independent reviewer did not rerun them.

Only source/metadata reads and stdlib AST/pure scalar checks were performed here. No real data, VAL/GT, model/checkpoint, GPU, native binary, process, guard, accounting, installation or Git action was performed. The preliminary proposal's implementation_authorized:false predates root's later registration, which explicitly authorizes source implementation only. Runtime admission remains root-owned.
