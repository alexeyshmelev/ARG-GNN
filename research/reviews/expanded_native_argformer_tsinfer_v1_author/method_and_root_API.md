# H038 native inference ARGformer condition

The frozen full-Relate source expanded_native_argformer_v1.py (fcd2c80e...) remains intact as a disclosed transfer diagnostic. Its marginal-tree internal IDs do not provide the cross-tree coalescent identity described by ARGformer. No clade hashing, rank matching, rounded-time identity, branch-association substitution or checkpoint reuse repairs that condition.

The NEW condition H038_native_ARGformer_current_tsinfer_tsdate_all952_actual_ages_v1 uses the identical completed H038 observed VCF, genotype sample order, sample roster, actual dates, site-coordinate map, scoring grid and observed donor labels. It retains952 haplotypes/476 diploids,686 queries (Present182, Bronze162, BAA14, Neo302, Yam26) and266 pure source proxies (Ana50, CHG28, WHG102, EHG86), mixed0. The34 older-than350 donors remain observed source-deme proxies; no fabricated census descendants or query labels enter inference.

Primary fitting consists of8 independently prepared, transductive VAL ARGs. Each receives its own vocabulary and fresh MLM/contrastive fit using only that ARG's paths and genuine donor labels. The32 TRAIN parents remain in the frozen registry and can be CPU-inferred under an explicit future-protocol scope; their checkpoints do not train or improve the8 VAL runs. These transductive VAL runs must be reported separately from an inductive cross-parent learner.

## Native inference recipe and evidence

The ARGformer article (DOI10.1093/bioinformatics/btag438; cached paper body/fulltext and pinned author commit f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596) specifies tsinfer+tsdate and paths exploiting shared coalescent events. Its original benchmark is AmericanAdmixture4B18:1600 samples,30Mb, three sources, no genotype/phasing error. The article/release does not provide exact tsinfer/tsdate versions or a complete versioned inference script. That paper-original reproduction remains a separate benchmark; this current pinned heterochronous transfer does not claim its numerical reproduction.

The existing qualified native recipe is scripts/infer_biological_paths_tsinfer_v1.py (b1f1c9a5...), configs/biological_paths_tsinfer_v1.json (85d8dc47...), and runtime receipt04876226... . The new condition extracts only its pinned scientific functions in an isolated namespace. Old reduced-panel parents, old storage ledger/caps and old observations are excluded. Source-only bytecode checks confirm the extracted scientific functions unchanged. No full952 native inference has been run by this authoring task.

Qualified observation runtime: .runtime/envs/biological_tsinfer_v1, Python3.11.17, NumPy2.4.6, tskit1.0.3, tsinfer0.5.1, tsdate0.2.7. Qualified model/path preparation runtime: .runtime/conda/envs/argformer_native_v1, Python3.11.9, NumPy1.26.4, tskit0.6.4, torch2.4.0. The sanitized tree's decoding across these pinned tskit runtimes must be established by the full952 canary; source checks do not prove binary/runtime capability.

The two-step historical-sample recipe is the native tsdate historical-samples recipe, whose pinned documentation warns that historical variational_gamma dating on tsinfer topologies has not been extensively tested. This is a concrete capability caveat, not permission to zero or adjust dates.

1. Build SampleData on the2Mb original bp0 domain, inserting actual diploid ages and every original phased VCF site in its original order.
2. Subset actual age=0 diploids. Infer, preprocess and date their genealogy, then estimate site ages.
3. Apply tsdate.add_sampledata_times to all samples. Its native np.maximum applies the oldest observed derived-carrier sample-age bound. Native NaN ages for nonvariable modern sites stay NaN; official tsinfer excludes those from ancestor inference and overlays their observed genotypes. No custom allele/time fallback.
4. Generate ancestors, insert native historical proxy ancestors, match ancestors, then match all samples with force_sample_times=True and overlay_non_inference_sites=True.
5. Preprocess/redate the full genealogy, verify all952 actual flagged sample dates/order and exact original phased allele strings at all original sites, and measure connected shared internal-ID reuse over the prospective comparison interval.
6. Clip/sanitize without simplify or invented node identity. Verify native node count, IDs, sample IDs, node times and order unchanged during sanitation; retain the full original native dated tree as a separate artifact.

Exact frozen settings: mutation_rate1.25e-8; synchronous num_threads0 with one CPU thread; recombination_rate=None and mismatch_ratio=None (native defaults, not the separately validated VCF map); path_compression/post_process=True. Preprocess delete_intervals=[], split_disjoint=True, filter_populations/individuals/sites=False. Dating variational_gamma, max_iterations25, rescaling_intervals1000, rescaling_iterations5, match_segregating_sites=False, singletons_phased=True, constr_iterations=None, min_branch_length1e-8, allow_unary=False, set_metadata=True, time_units=generations. Site ages use unconstrained=True, node_selection=child, min_time1. Historical proxies allow_mutation=True, epsilon=None. Selected APIs expose no random_seed parameter: deterministic original ordering and pinned synchronous native settings are used; no invented inference seed.

Population/source labels are excluded from SampleData and the inference functions. Dates are calibration constraints in native inference/dating, and are not inserted as ARGformer time tokens.

## Coordinates, sites and actual node mapping

VCF POS is1based and maps to the pinned original tree_position_0based=POS-1. REF is ancestral only because the pinned tskit1.0.3 VCF producer places the ancestral allele first; both the exact observed bytes and writer header are bound. Inference retains multiallelic/monomorphic nucleotide sites through native overlay and verifies allele strings; it does not apply Relate's variant filter.

The original fixed grid stays50+100*i on the2Mb domain. The frozen Relate receipt supplies the observations-only common support: first retained Relate SNP through last retained SNP+1, with its exact retained grid indexes and source site join. The tsinfer topology must connect all samples across that entire interval and demonstrate at least one native internal-ID reuse occurrence; gaps, multiple roots or absent reuse fail without identity repair, tail extension or selective scoring. The count is a representation adequacy check, not proof that inferred biological events are correct.

The new output records native full-site joins and explicit common comparison SNP joins separately. It retains every native site inside the common support, including sites rejected by Relate, while model comparison SNP outputs use the identical frozen retained Relate original-site coordinates. Each common row carries its actual tsinfer_site_id as well as the frozen comparison join; this prevents confusing inferred site IDs with original site IDs. Full original inferred flanks/sites remain in all_dated.trees; the clipped/scored output excludes tails.

Clipping shifts physical positions by coverage_left, so original=local+coverage_left. Node IDs are untouched. Genotype sample_index/genotype_slot, original source_observed_node_id and final actual flagged inferred_tree_node_id are explicit independent fields. The final952 actual IDs and actual dates are verified; node=genotype_slot is never assumed. The native dated genealogy may create/split/renumber nodes during its own supported inference/preprocessing; the new source claims preservation during subsequent sanitation, not identity between native pipeline stages.

## Model and comparison interpretation

The new adapter executes pinned fcd scientific functions in fresh isolated globals. The only adaptations are native observation reader, condition/schema identity, complete-corpus resource checks and VAL fitting scope. Source bytecode matches for factories, MLM/contrastive training, scientific backbone/initializer checks, GPU admission, pooling/retrieval and all numeric settings. Fit/retrieval bytecode differs only in output schema constants.

All952 full sample-to-root paths, sample-then-tree ordering and all occurrence joins remain. Deduplicate full native paths before leaf skip/truncation; tokenize real node IDs sorted within the parent:treeSHA namespace with PAD0/MASK1/CLS2/SEP3. Seed42 path90/10 split, length128. MLM uses all952 unlabeled paths; contrastive uses only266 pure observed donor leaves, preserving unequal classes and all donor full paths across internal train/val in the retrieval corpus. Query ancestry is never a fitting or selection input.

Fresh same-ARG MLM5ep seed17, full512 bf16, then contrastive1ep seed36, full4096 fp32. Required MLM train>=512, donor train>=4096 and nonempty internal validation fail without batch reduction. Native class-weight uniform fallback, temperature.3, unused3-logit classifier, special-token/80-10-10 masking behavior, author scheduler/optimizer/factories and same-parent MLM encoder initializer remain disclosed. No old vocabulary or checkpoint crosses ARGs.

Native retrieval calls ensure_l2_normalized and topk_indices_streaming independently for k1 primary and k5 sensitivity, q_batch1024/e_chunk100000. All unique donor full paths are included, including identical embeddings with different full-path IDs. Source CUDA tie behavior and raw hard-vote probabilities, zeros and potentially infinite NLL remain. No probability floor, calibration, balancing or coordinate restriction is added.

Native tsinfer ARGformer versus Relate AP/ARGMix/GNN is an end-to-end inference-plus-method comparison. It includes topology, dating, shared-node representation and inference cost. It cannot identify model-only superiority. Putting other methods on the same tsinfer genealogy is another separately frozen comparison cell.

## Root execution and provisional resources

The CPU supervisor's default is a source report. Its draft starts only the full952 ap_expanded_val_000 canary, with all40 registry/completion references retained as metadata. The exact permitted scopes are canary (VAL000), primary_VAL (all8VAL), and all40_future_protocol_CPU_only (all40). A larger CPU scope requires the completed same-source full952 canary summary and corpus eligibility. The primary fit/predict wrappers reject TRAIN parents.

The author has not reserved storage, measured runtime, started native workers or touched storage accounting. Source ceilings are provisional stop bounds: one worker/thread,48GiB RSS,4h per CPU stage, at most4GB observation+preparation per parent,160GB upper ceiling only for the optional40-parent CPU protocol;2M nodes,20k trees,20M edges,200k sites,10M full-path occurrences and1M unique paths. The occurrence ceiling also applies prospectively as952*num_trees. These are not measured needs or a request for a160GB reservation. The root input can bind lower positive per-parent limits within these ceilings, and its total allowance is bounded by its selected scope. There is no unmeasured400GB fit reservation.

Root first runs/inspects the complete canary and records real vocabulary/path counts, actual retained bytes, inference/preparation timings and memory. Root then selects realistic per-parent retained caps and schedules8 independent fits, with source batches unchanged. Fit storage and GPU memory require root corpus-specific admission; caps or unsupported native counts fail with preserved artifacts, never silent semantic reductions. Stage-boundary file checks and sampled/boundary CPU memory checks are not an OS filesystem quota.

Every native worker has owned TMP/cache directories. The supervisor waits for child exit before writing its bounded16MB launch log/exit receipt, avoiding concurrent filesystem guard writers. There is no implicit resume or cleanup of partial output.

Storage uses the pinned project_storage_terminal_retired_scopes_v1 helper, with all six accounting components plus original v9 stub in the source closure. Root must finish all H038/success-guard writers before activating the retired authority. The user's1TB project cap and950GB warning supersede the old500GB settings; the old biological recipe's storage fields are excluded. Source/docs do not retire/apply anything.

## Root API

Default source reports:

    .runtime/env/bin/python -B -m arggnn.prepare_expanded_native_argformer_tsinfer_v1
    .runtime/env/bin/python -B -m arggnn.prepare_expanded_native_argformer_tsinfer_v1 --admission-draft

The preparation runtime report is metadata only, called by root in the qualified native interpreter with CUDA_VISIBLE_DEVICES empty:

    .runtime/conda/envs/argformer_native_v1/bin/python -B -m arggnn.expanded_native_argformer_tsinfer_v1 --runtime-report prepare

Bind its JSON reference, exact source/config/selected completion pins, active expiry, per-parent resource_caps, selected execution_scope/parent IDs, fresh output and retired-scopes storage authority. Root binds only the selected parent completion refs as storage consumer bindings; all40 remain explicit registry metadata.

    .runtime/env/bin/python -B -m arggnn.prepare_expanded_native_argformer_tsinfer_v1 --supervise --input ACTIVE.json --expected-input-sha256 SHA

Canary output summary is prepared_scope.json, with observation/prepared manifests under parents/ap_expanded_val_000/. Larger CPU scopes additionally bind native_canary_ref to that completed summary, and use a fresh output. The script performs CPU inference+path preparation only. After actual canary qualification, root can separately call the adapter's guarded fit/infer APIs for the8VAL prepared manifests.

    observe_parent_v1(parent_id, completion_ref, output_directory, *, config, guard, resource_caps=None)
    prepare_parent_v1(parent_id, observation_ref, output_directory, *, config, guard, resource_caps=None)
    fit_parent_v1(prepared_ref, output_directory, *, config, guard, gpu_authorizer)
    infer_parent_v1(fit_ref, prepared_ref, output_directory, *, config, guard, gpu_authorizer)

Each fit remains fresh within its own parent:treeSHA namespace. Root's GPU callback retains eligible physical2/3/4/5, canonical UUID/held lease, two-device limit, actual driver-context identity and corpus-specific full512/full4096 memory admission.
