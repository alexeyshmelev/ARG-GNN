# Exact changed-clade flag control: source only

Implemented separately in src/arggnn/anonymous_clade_flag_control_v1.py. This is a prospective substantive control, not a trained model or a result. Root owns any new registration, source closure, resource admission, checkpoint binding, execution and evaluation. The current frozen H040 worker is not changed and does not accept this new flags= interface.

The scientific question is whether simple exact observed topology-change information explains an apparent benefit without learned endpoint correspondence. No publication-priority conclusion depends on empirical outcomes.

## Exact structural feature

For each native tree, compute exact descendant sampled-leaf sets with ephemeral Python integer bitsets. Compare only proper clades with 2 through H-1 leaves. Collapse identical clades along unsampled unary chains and place a flag at the deepest original node in each chain. Retain every original node in the neural network. Empty, singleton and full-panel-root clades receive zero flags.

Each original node has two uint8 channels in fixed order: gained_from_left, lost_to_right. The unavailable outer neighbor is zero. A proper clade absent from the preceding/following tree gets the corresponding flag. No internal-node identity is shared between trees, no endpoint pair is emitted, and no other tree's latent vector is gathered.

The builder processes adjacent trees sequentially. Exact masks exist only for previous/current trees and are not returned, serialized or moved to a model device. It returns an N-by-2 uint8 tensor (2N bytes) and compact metadata/hashes. Two full-forest int32 child-link arrays use 8N temporary bytes; tree masks/worklists and Python-container overhead are additional and unbenchmarked. No dense bitset cache or cross-tree all-pairs matrix exists.

The actual-bank wrapper requires the original CPU MutationRoutedWholeTreeInputsV1 with all 952 sampled haplotypes and 686 query columns. Flag construction reads only parent/sample_index, plus the query shape check. Stable sampled-haplotype columns are certified by root's observation reader, not inferred from numeric node indices. Forest topology, one distinct native root per hold, the complete leaf panel, acyclicity/coverage and unsupported sampled-interior cases are checked. Reference classes, ages, branch times, mutations, targets and outcomes do not construct flags.

These flags detect every change in rooted leaf-labelled topology after unary suppression. They do not detect changes consisting only of branch ages/lengths, mutations or hidden history, and cannot establish ancestral identity, homology, recombination or population ancestry transitions. Tree inference errors can create flags. The original model still uses its unchanged observed age/mutation/reference features.

## Model and initialization

The model reuses frozen H040 prepare_tree_base_v1, sweep_tree_state_v1 and leaf_context_v1 operations. It performs a full first PyG upward/downward sweep, injects a local node-state adapter, performs a fresh full second sweep and applies the unchanged ordered bidirectional genomic GRU/head. It preserves all original readout mappings and geometry.

With first-sweep node state h, the local adapter is

    z = gained_from_left * (U h) * (V h)
        - lost_to_right * (U h) * (W h)
    injection = O z

Products are componentwise. U,V,W are bias-free 32-to-8 maps, and O is a bias-free 8-to-32 map. Added parameters: 3*32*8 + 8*32 = 1,024. Original body: 29,764. Total: 30,788. This is parameter-budget matching, not equality of hypothesis classes.

O starts at zero. The same four state keys are used as H040: clade_coupling.{U,V,W,O}.weight. All original state keys and parameters remain unchanged. load_h038_flag_warmstart_v1 accepts the same payload schema: arm=aligned, seed=101, teacher_used=false, model_state=the selected original H038 state. It requires exactly those four missing projection keys. Root must bind the actual selected checkpoint later. This source task did not open it.

## Integration API

- build_clade_flag_arrays_v1(parent_array, sample_index_array) returns (N-by-2 uint8 ndarray, metadata). This pure structural helper permits a small handmade panel for verification.
- build_clade_flags_v1(original_cpu_bank) returns (AnonymousCladeFlagsV1, metadata) for the frozen 952/686 actual-bank contract. Keep the metadata/content hashes with the observation lineage. There is no disk loader or writer.
- flags_cpu.to(root_qualified_device, non_blocking=...) transfers only the compact flags. flags.tensor_bytes() reports the exact runtime payload bytes.
- AnonymousCladeFlagControlGNNV1(**original_model_kwargs) constructs the body plus adapter. Seed CPU initialization with 101 before construction.
- load_h038_flag_warmstart_v1(model, actual_selected_payload) checks the original warm-start schema. Reset AdamW moments/step and match exposure, parent order and checkpoint opportunities in a separately registered future worker.
- model.forward_holds(bank_device, flags=flags) returns [T,952,4].
- model.forward_train_holds(bank_device, flags=flags) returns [T,686,4].
- model(bank_device, flags=flags) returns [R,686,4] using original readout_hold.

Only control=aligned is accepted through the reused observation preparation. There is no boundary= event payload, tensor/eventMLP interaction switch, launcher, target reader, VAL scorer or admission bypass. Current H040 execution sources remain frozen.

## One authorized handmade CPU check

handmade_CPU_fixture.py ran once in the existing project environment with CUDA_VISIBLE_DEVICES empty, PyTorch FP32 and one CPU thread. source_CPU_check.json records success on seven leaves, two trees and 26 native nodes. It checks the exact K4-blind nonlandmark rearrangement, node/sample-column covariance, reversal channel swapping, unary representative/equivalence, identical topology, local-adapter covariance, zero-O output and first-step gradient blocking, followed by nonzero gradients after a deterministic assignment to O. There were zero optimizer updates.

Using a synthetic random backbone state only, it verifies the original warm-start schema, exact 29,764+1,024 parameter count, and identical initial state keys/values with H040 under the same seed. No original observation, real checkpoint, target, VAL data, full neural-bank forward, training or GPU context was used. This is a source property check, not evidence of biological usefulness or speed.

## ARGMix methods follow-up

ARGMix_method_evidence.json extracts only Methods 4.1 and 4.3 from the already verified cached primary XML, DOI 10.64898/2026.04.05.714076, PMC13082007. These describe TMRCA relative-positional self-attention on a 29-node subgraph (25 reference haplotypes, three inferred ancestors and the query). No adjacent-tree changed-LCA incidence or genomic recurrent stage is described in those methods. Tree skipping is sampling policy. The cached supplement is supplementary figures rather than an additional methods specification.

The initial priority audit has been updated to close the full-methods access gap for this primary version. The narrow candidate remains H040's specific incidence/routing rule. This is not an exhaustive priority finding, a superiority claim or an empirical review.
