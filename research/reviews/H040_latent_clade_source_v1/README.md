# H040 source payload

`src/arggnn/anonymous_clade_witness_v1.py` builds and verifies observation-only event sidecars. `src/arggnn/anonymous_clade_gnn_v1.py` supplies the rank8 coupling, actual shared PyG sweeps, H038 warm-start loader and matched family factory. `inactive_protocol.json` is a proposal, with no execution or storage admission.

Use the original frozen full-panel cache reader and bank. `prepare_parent_sidecar_v1(parent_id, bank, entry_ref, reader, guarded_output, cache_parent_ref=...)` returns the sealed record ref, runtime events and metadata. It joins the original observed roster, selects four distinct haplotype landmarks by seed101 hash priorities over stable sample keys and parent namespace, and enumerates all5,692 triples containing at least two landmarks. It carries only the prior tree's own structural winner state/receiver into its adjacent-boundary comparison. Unsupported sampled interiors/ties are unresolved; every query remains in the backbone and readout. No true ancestral correspondence is inferred.

`prepare_all_sidecars_v1` decodes the original index once and each of the40parents once, seals a complete observation-only manifest and reads no targets. It discards each bank after writing. For a combined prep/fit process, call the parent API and retain the already-built bank/events so strict read-time LCA verification need not be repeated. `read_parent_sidecar_v1(..., bank=bank)` separately verifies source/cache/roster and local bank hashes, the complete canonical landmark/triple recipe, every within-tree winning LCA/shared sample, and the endpoint degree weights, in bounded CPU blocks.

After root has verified and loaded the actual selected aligned H038 checkpoint payload, the minimal model entry is:

```python
family, optimizers, proof = build_matched_family_v1(
    payload, original_h038_model_kwargs, device=device, execution_guard=root_guard)
events_on_device = events.to(device)  # shared read-only events for both arms
for arm, model in family.items():
    model.train()
    logits = model.forward_train_holds(
        bank_on_device, boundary=events_on_device, interaction=arm)
    # Original external TRAIN exact-span mass/loss, AdamW and clip1.
# Evaluation: model.eval(); model(bank, boundary=events, interaction=arm).softmax(-1)
```

The main tensor and event_mlp arms have the same1024 additional bias-free learned weights and identical initial body/U/V/W/O states. They use the same events, eta, signed scatter, two complete-tree sweeps and unchanged full-native-hold GRU/head. The event MLP is GELU(Ua+Vb+Wc)-GELU(Ua+Vc+Wb); it can model three-way interactions. The optional pairwise API uses all four matrices too, but its raw core consists only of pair interactions. Enable it only prospectively. `O=0` yields the original tree representation initially; gradients reach O first, then U/V/W after O moves. All sampled contexts and old/new clade states stay attached to autograd. Event blocks are checkpointed without tile resets or detached histories.

Native TRAIN logits retain[T,686,4]; evaluation retains the original bank.readout_hold grid and query_column order. The proposed discovery protocol is seed101/FP32, same32TRAIN/eightVAL, same selected H038 warm start, ten additional passes/320updates per registered arm, eval5/10, no early stopping, equal fresh AdamW state and original Present182 selection. ModelAPI does not open any targets or predictions. Root must build its own prospective phase/fit registration and admitted driver.

The single handmade CPU fixture passed: two five-leaf trees produced exactly one changed triplet with old/newLCA5/12 and shared sample0;1024extra/30788total weights; zeroO shared-sweep identity bitwise; empty/equal-clade messageszero; reversal and sample/node covariance exact for tensor/eventMLP; pairwise reversal error5.96e-8; full shared-PyG sweep node covariance exact; finite connected old/new/sample/four-projection gradients afterO moves. It used no real arrays, GPU or optimizer. Production sidecar preprocessing, full-data warm-start compatibility, GPU memory/time and accuracy are untested.

Raw archival arrays cost26bytes/event plus68,320bytes static audit data per parent. With T native holds, E<=5692*(T-1); `sidecar_byte_bound_v1(T)` returns this bound. At illustrativeT=1000 for each40parents, the bound is5,916,493,120raw array bytes before NPZ/JSON/filesystem overhead. Runtime event tensors cost20bytes/event (<=113,726,160bytes/parent atT1000). No recombination sparsity or compression saving is assumed. Root must reserve from actual metadata bounds and measure preparation/I/O. One-parent preparation, hold-block16 and bounded read verification limit scratch; original bank memory is additional. Projected node/sample states and rank8 scatter costO(Mdr+Er), plus one extra full-tree sweep; active autograd memory and throughput remain unmeasured.

The antisymmetric trilinear product is established by the imaginary-relation term of ComplEx (Trouillon etal., arXiv:1606.06357). Higher-order/orientation-aware message passing and anonymous temporal motifs also have strong prior art (MPSNs andCAW-N). The candidate contribution is the particular sample-anchored two-tree reassociation incidence, coupling actual old/new anonymous LCA states before signed injection and whole-tree rerouting. A temporal factor-graph MPNN given these witnesses can represent the same mechanism. No kernel or architecture priority, general expressivity improvement, full-GRU chromosome-reversal invariance, or empirical superiority is claimed.

No H039, TRAIN worker, guard, scheduler, source pin, reservation, live job or Git state was changed. Dependencies are the existing full-panel/PyG/body and original observation cache/reader APIs; no H039 continuation/worker is imported and no broad legacy source closure was added.
