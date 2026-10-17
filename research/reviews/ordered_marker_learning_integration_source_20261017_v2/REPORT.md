# Ordered-marker learning source v2

17 October 2026. This new namespace implements the complete scientific TRAIN/VAL caller for the two placement arms. It is source-only, unexecuted and awaiting independent review. It creates no operational wrapper, resource guard, admission or job. The original H058 and inactive marker v1 sources are preserved.

## Exact review delta

The independent v1 review is `research/reviews/ordered_marker_learning_independent_20261017_v1/review.v1.json`, SHA256 `8e8988e095ec4b999fad43f6f1daad73dc54ad7f12df780c8a185bf1d7258d84`. Its three blockers drive this revision.

**B1: inner backward lifetime.** The old positive backend directly returned dense block features into a full-hold feature graph. V2 instead exposes one bounded `marker_block_v2`. The model's forward and reverse block consumers both construct those features and consume them through the directional GRU *inside* nonreentrant checkpoints. Every original block contains at most16 markers. Dense seed/up/outside/message arrays and sparse learned fields are therefore inside the block recomputation boundary. The mandatory whole-hold checkpoint and inherited backbone/site checkpoints remain.

No full `[S_hold,Q,24]` feature or mask sequence is assembled. Forward recurrence processes blocks in original order; reverse recurrence processes the same blocks in reverse order and reverses rows within each block. Hidden states cross block boundaries without detach and reset only at native holds. Every observed marker, including a missing position in a partially observed hold, remains in the sequence. The sparse/dense decision remains the same observation-only block rule in both directions.

Two public unidirectional width8 GRUs implement the two directions of the previous bidirectional GRU. Their parameter names differ from inactive v1, so no v1 new-module checkpoint is claimed compatible. The function class and864 recurrent parameters remain; both new placement arms share identical initialization. The new module still has1440 parameters analytically, giving33966 per arm with the unchanged H045/H058 body.

**B2: trainability/FP32.** `validate_trainable_fp32_v2` requires every allocated parameter to be trainable FP32 and the full33966 allocation. The initializer checks it before and after loading the exact supplied H045 state. The family builder checks both arms; the scheduled step checks it again. First scheduled gradients are checked for every named parameter, with finite FP32 values and actual fresh AdamW counters. Zero output heads can initially give deeper new weights zero objective gradients; zero is allowed, while an absent gradient fails the scheduled qualification.

**B3: immutable registration provenance.** `SOURCE_REGISTRATION_SNAPSHOT.v2.json` freezes the exact candidate protocol, protocol source hash, old candidate refs and sealed independent review. The stale ledger ref is retained only as explicitly inactive historical drift evidence. It is absent from active source refs. This snapshot records source registration for review and grants no experiment or execution authority. A future admitted caller must supply the actual immutable experiment registration referring to this snapshot. No shared ledger or experiment decision was changed here.

These are implemented source remedies, not a new independent SOURCE_PASS.

## Actual scientific caller

`producer_v2.run_context_v2(ctx,input_ref)` now calls the full scientific path. The observation bridge uses the original two-worker full-parent reader, stable observed joins, original H045 episode builder and original F1 binder. All32 TRAIN parents are loaded; every original node, native hold, SNP,952 sampled identities,686 queries and266 references remains. One aligned packed route and positive plan per parent is supplied identically to both new arms. Both placement arms use aligned reference routing; the frozen H058 alignment-null experiment remains unchanged.

All phase observations, episodes, routes and positive plans are prepared and sealed before TRAIN targets. Supplied-state validation authenticates the same completed H045 own-state pass30 metadata. Seed101 initializes identical new weights; the clone has independent parameter storage. Both models retain every31812 inherited weight,714 same-marker weights, both complete tree sweeps, clade/episode coupling, original chromosome-wide BiGRU, classifier and native-hold output support.

The two honest arm names remain `ordered_marker_tree_feedback` and `ordered_marker_sample_readout`, under the distinct candidate id `ordered_marker_placement_v2`. The common H058 residual is identical. The ordered residual enters query nodes before sweep2 in the first arm and the same query sampled contexts immediately after sweep2 in the second arm. This tests placement of ordered evidence; both arms receive order.

The producer invokes the actual original `h45.train_one`, preserving exact native-hold span-mass loss, FP32 backward, norm1 clipping and fresh independent AdamW (lr0.001, weight decay0.0001, betas0.9/0.999, epsilon1e-8). The original `Random(101+pass)` order gives30 complete passes over32 parents,960 updates per arm/1920 total. The first real scheduled update is retained and qualified inline. Checkpoints5/10/15/20/25/30 carry new schemas, names and source/initializer lineage. There is no early stop, disposable update, silent continuation or automatic retry.

The evaluation caller loads the exact new12 checkpoint products, validates30-pass histories/counters and predicts every8 VAL parent at every checkpoint in both arms. Its RAM transport requires all96 products and verifies frozen probabilities before opening either arm's VAL truth. Selection uses the original equal-parent Present182 accuracy, lower NLL, earlier-pass rule. Original100bp lookup, all7 cohorts and inherited selected-checkpoint tract diagnostics remain. No old H058 or H050 result is relabeled as this candidate.

## Lifetime, cost and accounting

The block boundary removes full-hold feature-sequence retention; it does not establish constant memory or measured fit. During backward, checkpoint boundary hidden states and common sum/count/max outputs can scale with the number of blocks and Q. Each active block still has its complete dense `N*B*8*4` graph or sparse path graph, recurrent work and indexing temporaries. Whole-forest states, residual/scatter rows, original stages and original site encoder remain. Evidence is recomputed for both recurrent directions and again for backward. No learned gate, field, state or activation is cached.

Persistent positive plans add256 count bytes per site plus parent/query tuples, covers, stamps and Python objects. Preparation remains straightforward site/reference access and tree/cover construction; this has no demonstrated speed advantage. The bridge counts owned Python objects and the union of CPU tensor/NumPy storage intervals, uses the existing bank/episode/route caps and preserves the32GiB aggregate preparation and128GiB RSS condition. A separately supplied positive-plan subcap is mandatory. It checks the retained bridge plus original target allowance; temporary native/allocator/workspace peaks still require actual ROOT RSS enforcement and scheduled full-parent fit evidence. No actual footprint, fallback fraction or timing was measured.

## Exact remaining external interfaces

The scientific trainer/caller is implemented. Future ROOT-only execution integration remains absent: a source-pinned admitted context/factory, exact phase reader/target bindings, bounded writer, existing guard and episode-preflight callbacks, configured CPU32/interop1 runtime, complete transitive operational closure, actual experiment registration, source-bound runtime declaration, source counts/layout proof and reviewed caps including `CPU_positive_plan_cap_bytes`. The caller requires these inputs and cannot silently accept the original H058 configuration/schema. Evaluation additionally requires this candidate's genuine completed TRAIN ref.

Independent complete source review, actual initialization/gradient equivalence and full-parent forward/backward fit remain unperformed. No toy result is promoted to this new model. The standard-library checker parsed/compiled8 new Python files and passed15 source assertions, including unchanged accepted feature algebra, checkpoint call structure, full schedules,96-product barrier and25 direct immutable refs. It did not import candidate modules or instantiate models.

No scientific payload/truth, model/numerical execution, guard state, live resource/process/job, install or Git operation occurred. Native-hold predictions still cannot localize a switch within a hold. One seed and reused8 VAL parents remain development evidence. No speed, efficacy or novelty result is claimed.
