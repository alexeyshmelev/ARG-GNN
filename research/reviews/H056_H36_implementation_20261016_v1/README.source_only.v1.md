# H056 H36/R16/Q20 source implementation

This packet implements the approved plan `research/reviews/H056_H36_source_20261016_v1/PATCH_PLAN.md` (SHA256 `16692fa12edf2a2d5922054b9e1e08f6fe7605f85d8277419bcb875df08b4654`). It is source-only: no source module was imported or executed, no numerical fixture ran, and no genotype, tree, checkpoint or prediction payload was read or hashed. AST parsing and compilation without execution are the only code checks recorded here. Root operational binding and qualification remain separate.

The frozen protocol is `research/reviews/H056_frozen_H045_Danish_application_registration_20261016_v1/protocol.root_frozen_v1.json`, SHA256 `9544bdbd499bddafeb0774c67e4e86ef25535d41dd2409e710c29b679619cd6e`. Its positive EHG/CHG predictions remain unchanged by H055 outcomes.

## Fixed scientific and source boundaries

The contract fixes the ordered18 people, adjacent haplotypes0/1, first16 reference haplotypes, query columns16:36, four reference haplotypes/class, and class order ANA/CHG/WHG/EHG. The first five query people form the earlier group; the final five form the later group. Query labels are-1 and no query ancestry truth is supplied. Ages remain source mean calBP1950/28; original fractional ages and native half-even six-decimal ages are separate, with exact FP32 agreement required.

Each chromosome20/21/22 requires fresh phased GT and a fresh genealogy on GRCh37 bp0[20M,30M), including chromosome22. Original VCF REF/ALT GT codes supply every observed feature; native variants, mutations, populations and inferred ancestral alleles supply no model feature. Native-minus1 coordinates retain only genuine support, with all qualified constants, sites, sample identities, unary vertices and native holds retained. Strict native-GT export facts remain actual independent facts; graph-only readiness does not force that gate to pass.

The selected H045 foreign_episode seed101/pass30 checkpoint remains SHA256 `bfd8a87632015a900d9da6009d4b0ee037958460038d7949d0998e80b770796b`, with strict unchanged31,812 finite FP32 values. The learned encoder, adapters, full chromosome GRU, head and normalizations reuse frozen source. There is no optimizer, calibration, weight transformation, architecture change or reselection.

Native inference retains inherited pre-size N30000, mutation rate1.25e-8, 28years/generation, zero population-size iterations and seed414855340. The pre-size real-data versus European post-size training difference is disclosed. The inactive profile grants no execution authority.

## Pure contract and count seals

`contract_v1.py` supplies constants, frozen refs, the21 unchanged source pins and:

- `planned_roster_v1()`: deterministic metadata only; no file or authority is written.
- `validate_roster_identity_v1(rows)`: exact36-row subject/haplotype/role/class/date identity.
- `validate_counts_v1(counts, graph=False)`: exact keys `candidate_sites,sites,constants`; C>=S>=1000 and0<=K<=S.
- With `graph=True`, additionally require `native_trees,native_nodes,native_edges,vertex_occurrences`. These are actual T/global-node/global-edge/summed-forest-vertex counts, not historical or predicted dimensions.
- `variant_id_v1(chromosome,pos,ref,alt)`: `H056_{chromosome}_{absolute_POS_bp1}_{REF}_{ALT}`.
- `validate_source_manifest_v1(manifest)`: all21 frozen pins plus the six new reader/application runtime modules.

The root must seal independent actual counts, site identities, serialized phase/nucleotide/sample/date/map joins, native dimensions and coverage before downstream use. A chromosome below1000 qualified sites or a missing person/chromosome makes the registered application inconclusive. No replacement, thinning, padding or new window is implemented.

## Acquisition, bridge and native APIs

The separately sealed `acquisition_native_handoff.v1.json` and `acquisition_native_source_receipt.v1.json` describe these units:

- `extract_h36_v1.execute_v1(ctx)` uses the root's existing opener, request checks and append writer. Root owns compression, indexing, completion seals and actual serialized readback.
- `bridge_h36_v1.assemble_baseline_v1(vcf_gz_raw,qc_raw,roster,*,chromosome,counts,expected_vcf_sha256,expected_qc_sha256,max_output_bytes=25000000)` returns `(payloads,report)`. It preserves the original phased/nucleotide/QC semantics and emits41-field haps,27-field VCF,18-person sample and36-age records. Its audit flag is `monomorphic_among_panel`.
- `native_h36_core_v1` provides `expected_observations_v1`, `parse_anc_v1`, `parse_headers_v1`, `cli_stages_v1`, `verify_graph_only_v1` and `verify_native_v1`. Header parsing requires explicit root `max_chunks<=S`.
- `verify_native_v1` returns schema `H056_H36_actual_native_facts_v1`, with top-level protocol/chromosome/counts, actual `strict_GT_actual_status`, `graph_only_ready`, coverage and `native_sample_haplotype_date_mapping`.

No duplicate native root controller, admission system, launcher or scheduler is supplied. Root must bind the existing supervision/wait/RSS/timeout/drain/cleanup route to this reviewed source family.

## Panel, preparation and reader APIs

`panel_h36_v1` exposes `panel_metadata_v1`, `verify_native_tip_dates_v1` and `translated_interval_view_v1`. `preparation_h36_v1.prepare_arrays_v1(trees,intervals,positions,genotype,allele_count,roster)` preserves predecessor arithmetic; the pure array API permits small synthetic inputs, while the empirical reader enforces the1000-site minimum. `to_torch_bank_v1` uses the original tensor dataclass. The conservative resource formulas remain256*N+16*S*40+65536 and2058*N+128*40; they are bounds, not measured RSS.

`reader_h36_v1.parse_original_sites_v1(vcf_gzip,roster,frozen_identities,*,chromosome,counts,max_decompressed_bytes)` joins exact qualified identities to original phased GT. `prepare_from_root_context_v1(ctx)` returns only RAM `bank,flags,metadata`; it writes nothing.

Root reader context requires `root_authorizer="root_agent"`, `admitted=True`, `reader.root`, `reader.read(ref)`, `reader.decode_json(ref)`, `before_stage(stage,plan)`, and the exact `cfg/config_ref`. Config schema is `H056_H36_reader_config_v1`, status `root_admitted_observation_preparation`, `preparation_authorized=True`, with protocol/chromosome/full `expected_counts`, `genuine_regional_bp0`, feature semantics, readout/coordinate policy, no-truth/native-feature flags, source manifest and positive read/decompression/preparation/tensor/flag caps.

`observation_refs` names are `roster,frozen_sites,original_VCF,serialized_readback,bridge_haps,bridge_sample,bridge_ages,native_mapping,graph_contract,native_graph`. The reader source manifest schema is `H056_H36_reader_source_manifest_v1`, with identical protocol/observation refs and full `source_sha256` pins.

The graph contract schema is `H056_graph_structure_plus_original_GT_v1`, binding protocol/chromosome/counts, native graph/facts refs, `graph_only_ready=True`, `immutable_original_GT_features=True`, genuine coverage and actual strict status. Mapping `source_native_facts_ref` must match; mapping rows and strict status must equal the actual native facts. Serialized readback must record genuine root wait/group absence and actual serialized site/phase/order, sample/haplotype/date and qualified map/local-site joins. Actual graph projection checks dates, sample flags, strict chronology, nodes/edges/trees and summed vertex occurrences against the seal.

## Model and application APIs

`model_h36_v1.H36EpisodeInterfaceV1()` constructs36/20 directly from the original lower PyG body plus original flag/episode adapters. Its explicit40/24 and952/686 paths delegate the unchanged predecessor constructor for qualification only. `forward_holds` is a frozen method alias; H045 encoder and forward remain inherited. `episode_inputs_v1(bank,*,known_flags,max_payload_bytes)` uses original episode construction and requires identical aligned flags.

`apply_h36_v1.execute_v1(ctx)` uses the original injected `reader.read(ref),writer(name,bytes),before_stage(stage,plan)` boundary and exact root-authority/source/config/runtime/expiry bindings. Root supplies count-derived positive caps; no active application metadata is created here. The source manifest schema is `H056_H36_application_source_manifest_v1`, with protocol, full source pins and `required_metadata_refs`. The direct closure is at most64 .py/.sh/.json refs; payload and raw TSV bindings are transitive.

Application config schema is `H056_H36_CPU_application_config_v1`, with `execution_enabled=True`, exact protocol/checkpoint/descriptor refs, `CPU_threads=4`, `optimization_steps=0`, source manifest, and chromosomes20/21/22. Each chromosome item contains `counts,bank_ref,bank_completion_ref,preparation_wait_ref,roster_ref,genuine_regional_bp0`. Caps are `output_cap_bytes,max_serialized_artifact_payload_bytes,bank_read_cap_bytes,checkpoint_read_cap_bytes,episode_cap_bytes,max_host_RSS_bytes,wall_deadline_seconds`; serialized cap is below output cap and checkpoint cap is at most1MiB.

Each new bank completion must use schema `H056_empirical_CPU_preparation_completed_v1`, status `completed_observation_preparation`, exact protocol/chromosome/counts/bank/input refs and reader metadata, with no query truth/model/checkpoint/GPU execution. The genuine new wait must record `actual_exit_code=0,actual_direct_wait_observed=True,process_group_drained=True` and exact `completed_ref`; old H018 childexit1/recovery is not accepted. A saved bank uses schema `H056_empirical_CPU_observation_bank_v1`, exact protocol/chromosome/input refs, identical metadata, `bank_tensors` and `flag_node_flags`. Embedded metadata protocol/chromosome/count/roster and complete tensor hashes are checked.

The application validates(S,36),(T,36),(T,20,4), exact donor/age/query vectors and flags/episodes. It runs the full history separately for each chromosome, with no2Mb resets or cross-chromosome carry. It emits all native-hold probabilities and genuine intervals, summary, provenance and a completion receipt last. A partial failure has no completed receipt.

`summarize_probabilities_v1(chromosome_results)` preserves physical-span weighting per haplotype, then averages two haplotypes/person, pools chromosomes by actual covered bp, equally averages five people/era and reports later-minus-earlier EHG and CHG regardless of sign. It reports all four classes, chromosome/person values, uncovered tails, simplex/domain diagnostics and descriptive subject/site/chromosome deletions. It supplies no ancestry accuracy, mixture calibration or SNP-independent p-values.

## Deferred single qualification

Only after independent source review and root's separate bounded CPU qualification decision, run once from the project root with its pinned existing runtime:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .runtime/env/bin/python research/reviews/H056_H36_implementation_20261016_v1/deferred_tiny_FP32_H36_fixture_v1.py --run-once
```

This command was not run. It uses synthetic trees/four sites/constant-site retention, CPU FP32 seed560001, strict identical synthetic31,812-value states, finite H36(2,20,4), H40 original/shared logit comparison at atol5e-6/rtol5e-5, H952 state keys/shapes/values only, and a synthetic three-chromosome reducer check. It never reads the real checkpoint or empirical payloads. It does not qualify H952 forward execution, real checkpoint compatibility, full-parent RSS, native runtime or end-to-end application.

`source_handoff.v1.json` binds all final sources, focused predecessor diffs, this note and the two static receipts. The source review itself grants no execution authority.
