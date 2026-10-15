# H042 arbitrary-panel state compatibility and the required H018 loader

The learned-state argument is established at source level: the original H042 layers have42 state tensors and30,788 learned values, independent of the number of haplotypes or queries. The new factory retains those exact keys/shapes and requires the genuinely selected completed H042 state in full. No checkpoint/state tensor, model, VCF, native tree or ancestry target was read or instantiated here. Strict runtime loading and numerical parity remain root qualifications.

## What is preserved

The original width32 backbone has38 learned tensors/29,764 values. Site input width is19; mean/max pooling plus logcount/empty/span has width67; leaf input has32+class4+unknown1+age1=38; node input has width3; edge gates have geometry4+support5=9. The full native-hold bidirectional GRU has input35, hidden32 and two directions. Its64 history channels join32 context channels, so the first classifier layer remains32x96, followed by the unchanged four-class output. The rank8 flag adapter adds exactly U/V/W:8x32 and O:32x8, all bias-free, for1,024 more values and four state keys. No H/Q dimension occurs in a learned tensor.

The source-derived state_schema.json contains every key/shape. The source checker independently derives the same schema from original Linear/LayerNorm/Sequential/GRU declarations without importing Torch. The new constructor calls the same original full-panel PyG base initializer with the explicit actual H count, hidden32, site chunk256 and checkpoint_sites=True, then constructs the same original RankEightCladeFlagAdapterV1 in the same order. Expected query count is a plain attribute. It does not initialize fake952 rows, add/pad inputs, or call the H038 partial warmstart routine. All42 completed-H042 tensors, including the trained adapter O, load strictly. Initial random values are overwritten and exact loaded values/digests are rechecked; CPU RNG state is restored around construction. The returned model is CPU FP32, eval-only and requires_grad=False.

The subclass inherits the original H042 encode_trees, forward_train_holds and forward methods. Therefore the two complete original PyG sweeps, original cavity recipient exclusion, local rank8 flag arithmetic, site encoder and readout indexing remain the original methods. Its forward_holds body changes only h!=952 to h!=self.expected_haplotypes and query count686 to self.expected_queries. Restoring those two guard expressions gives exact original AST equality, including every FP32/device check, geometry operation, full-hold GRU call and classifier call. forward_count_guard_only.diff makes this reviewable.

Flag preparation calls the original build_clade_flag_arrays_v1 and AnonymousCladeFlagsV1.from_verified_arrays directly after a new explicit panel-identity validation. No exact-clade math is copied or changed. It still forms sampled-column bitsets, compares adjacent complete native holds, excludes singleton/full-panel clades, collapses equal unary clades to the deepest representative, and emits gained_from_left/lost_to_right uint8 flags. Structural bitsets remain local to previous/current trees. Different H changes the actual observed panel represented by the same definition; it does not invent missing leaves or reference donors.

## Fixed features, with no real-domain rescaling

The donor feature function remains observed_allele_frequency_arrays_v1: alpha0.5, younger ages<350 generations and older ages>=350, channel order younger ANA/CHG/WHG/EHG then older ANA/CHG/WHG/EHG. Common donor callability within each cohort, missing-call masking, raw allele frequencies and site entropy are unchanged. Query input classes remain-1. Proxy labels are observed assignments; the source does not establish biological pure ancestry.

All learned log scales stay frozen: node/sample-age geometry uses the original log10001; support/count features use original log100001; span and neighboring-span features use log(2000001), including inherited encode_sites and the full native-hold GRU geometry. H018's extension endpoint normalizes physical stability integration over10,000,000 bp. That endpoint metadata does not change the learned model divisor to log(10000001). Real ages, sparse R16 donors and the longer native sequence remain distribution-transfer questions, not shape fixes or calibration authority.

## Existing production assumptions that cannot serve as the H40 loader

| Frozen source | Actual restriction |
| --- | --- |
| anonymous_clade_flag_control_v1.py:254 | H042 constructor explicitly requires hidden32 and H952 |
| same source:273 | forward_holds explicitly requires H952/Q686 |
| same source:198 | bank-to-flag integration wrapper explicitly requires H952/Q686; its underlying array routine at88 is already panel-parametric |
| mutation_routed_full_panel_observations_v1.py:24 | Original48 simulation parents,32TRAIN/eightVAL/eightouter, original independent seeds |
| same source:52 and66 | Fixed H952/Q686/pureR266 and class counts50/28/102/86 in production joins/preparation |
| same source:78 and94 | Original476-diploid roster and source/native sample/date joins |
| same source:86 | Genotype width952 and VCF485 fields=9+476 subjects |
| same source:106 | Full-bank952/686 and original reference-class counts |
| same source:114 | Fixed entry/source/inference schemas and sample counts; original2Mb coordinate/coverage/normalization contract |
| mutation_routed_tree_observations_v1.py:429 | Base production preparation still requires H280/Q120/pureR96 unless fixture; fixture=True is not a production bridge |
| mutation_routed_whole_tree_gnn_v1.py:163 | Its original CPU structural validator additionally retains the special H280 query/reference constraint |

Every listed frozen source remains unchanged. The generic model factory supports explicit H>=3 and0<Q<H at learned-state/forward level. The provided CPU validator intentionally reuses the original validator, including its H280 special case; it is not a universal real-data loader. The first reviewed panel contract here is H018 H40/Q24, which does not hit that H280 clause. Other panels need their own honest identity contracts and review of any relevant production assumptions.

## New explicit H018 loader/identity requirements

No loader is implemented in this folder. The root-qualified bridge must establish the following with actual evidence, then supply the in-memory original MutationRoutedWholeTreeInputsV1 dataclass. Shape flags and caller booleans alone do not prove these facts.

1. Bind the exact H018_R16_20_30Mb_extension_v1 source condition, its fixed4,148 QC-site identities, actual observed nucleotide calls/missingness and original baseline or fixed coherent HS draw policy. Do not read AP/true ancestry. Native retained sites/coverage remain unknown until qualified. Preserve all intended sites or report an honest failure/support change; do not silently intersect/reorder masks or invent bases/phases.
2. Bind twenty genuine distinct diploids, each original haplotype0/1 in order, with R16 then Q24. Proxy classes are0x4,1x4,2x4,3x4 followed by24 unknown-1 queries; query columns16..39. Preserve exact subject/haplotype keys and the source-to-native leaf mapping across every hold. No padding, duplication, substitute reference or query-as-proxy label is allowed. Stable keys are not contained in the tensor bank, so the external native/source join is essential; a shape check cannot reconstruct it.
3. Preserve original calibrated BP1950 mean dates divided by28, no2026 offset, integer rounding, clamp or invented modern date. Both haplotypes of each diploid share that unchanged source date. Native/source chronology and raw branch durations must be checked in CPU float64 before the original FP32 feature cast. The intended eight donor channels each contain two actual reference haplotypes; the H40 gate checks the full-precision source-age cohort assignment and exact bank age vector. It does not make those proxies biologically pure.
4. Bind the qualified map/native inference recipe, absolute GRCh37 chr22[20,30)Mb versus local coordinate transforms, all original native holds and unary branches, node/sample dates, every retained site/readout interval and genuine clipped support. Preserve physical spans as bp and the exact original geometry/concordance/donor feature functions. No terminal extrapolation, collapsed holds, restarted/chunked GRU history or synthetic ancestry labels are introduced to make runtime fit.
5. Supply an actual bridge completion and explicit identity contract, exact ordered keys/classes/full-precision ages/query columns, all bank tensor hashes and their combined bank_fingerprint_sha256. The CPU gate reuses original structural validation and compares the actual bank tensors to the qualified vectors/content fingerprint. It cannot independently establish the retained source/native joins named by that external evidence.

H018_H40_observation_identity.inactive.json is deliberately qualified:false with missing actual refs, keys, ages, bank fingerprint and donor evidence. It cannot pass. Its shape/order constants are prospective requirements, not a materialized bank. Proxy pure ancestry remains explicitly unverified and no real truth is available.

## Actual selected best H042 state and next root evidence

selected_state_identity.inactive.json is also unqualified and has no selected pass/checkpoint/evaluation ref or tensor hashes. Root must use the genuine completed original H042 selection protocol and checkpoint identity, not merely assert a convenient pass or provide same-shaped H038/H040 weights. The factory checks the supplied root identity and all exact tensor hashes/values but does not open checkpoint paths or determine which checkpoint won. The caller's digest map only becomes authentic when the separately guarded original checkpoint/selection reader binds it to the actual completed selection.

Root first needs a separately bounded original952/686 parity qualification: authenticate the selected full H042 CPU state, load original and new classes strictly, compare every state tensor and flags from the same actual original bank, then compare complete-hold/query/readout outputs under a frozen numerical/runtime acceptance contract. No tolerance is invented from observed disagreement. H40 then requires the new loader/identity qualification plus finite actual output shapes[T,40,4], [T,24,4] and [R,24,4] and chronology/support round trips under existing root guards. The full native-hold sequence must remain intact.

No real-bank, original-panel tensor parity, resource/runtime or domain-transfer claim is made here. This is software generalization of a frozen learned state. Any later baseline-versus-eight-HS stability result retains the existing natural-log JS/common-support endpoint and is not accuracy, reference purity, novelty or retraining evidence.
