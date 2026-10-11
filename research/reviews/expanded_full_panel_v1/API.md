# Expanded author-full-panel producer v1

Source: src/arggnn/prepare_expanded_full_panel_v1.py. Source-only authoring.
Default invocation performs metadata/source qualification and prints the frozen
registry. It runs no simulation, native inference, model preparation or GPU job.

## Existing corpus findings

The original full-author family has four completed generated/Relate parents:
ap_full_train_000 and001 are downstream TRAIN; ap_full_train_002 is explicitly
reserved INTERNAL_VAL despite its original generation tag train; and
ap_full_discovery_holdout_000 is OUTER_DISCOVERY. The fifth registered outer001
has no output directory. Thus full-panel ready eligibility is2TRAIN+1VAL+1outer,
not3TRAIN or a ready32+8 corpus. Existing inner parents can be adapted to the
strict952 loader as historical comparison products with their frozen downstream
roles preserved. The outer parent remains outside model tuning/training.

Other ready inferred development data has64TRAIN+16VAL independent parents in
ancient_inferred_development64_v1. It has88haplotypes,64queries,24references,
1Mb length, different fixed proxy dates/128grid and inference/polarization
policy. It could support a separately declared auxiliary robustness condition
with its own strict adapter. It is not an author952/686query full-panel corpus.
A smaller common-input engineering family has16TRAIN+4VAL. The four280-haplotype
biological pilot parents remain the repeatedly used H036 development set.
All inventory evidence came from metadata, with no outer truth arrays opened.

## Frozen new registry and unchanged biology

build_config_v1() returns the proposed frozen config. The source-only proposed
file is research/reviews/expanded_full_panel_v1/proposed_config.json. It contains
32TRAIN,8INTERNAL_VAL and8UNTOUCHED_OUTER streams. New parent IDs are
ap_expanded_train_000..031, ap_expanded_val_000..007 and
ap_expanded_outer_000..007, under data/expanded_full_panel_v1.

The full author's476diploid/952haplotype roster and original within-cohort age
order stay unchanged. Queries are686: Present182, Bronze162,BAA14,Neo302,Yam26.
Pure proxies are266: Ana50,CHG28,WHG102,EHG86. There are no mixed-reference-only
leaves. All686 query input classes remain unknown, including BAA. References
keep actual observed dates. The inherited author's demography,2Mb length,
recombination1e-8,JC69 mutation1.25e-8, census350 external target mapping and
Relate1.2.4 initial plus five-size-iteration recipe remain unchanged.

New deterministic SHA256 seed domains include the expanded namespace, root101,
split, parent index and purpose. There are432 distinct ancestry/mutation/initial
Relate/size-base-and-five-iteration values. They do not collide with361 prior
registered seed values from the original biological pilots, original full-panel
registry and older development/common-input manifests. Final branch reestimate
reuses the fifth-size iteration seed as in the released original workflow.
All48 streams are sealed before generation. Collision or failed generation is
reported; no automatic seed replacement or favorable resampling is performed.

Independent simulations improve coverage of stochastic genealogical variation
within this fixed recipe. They do not vary the author's demographics, sampling
ages or other systematic assumptions and are not a general representativeness
claim. All new48 are independent of the earlier parents; none is reused.

## Root-only execution

The root calls execute_parent_v1(config,parent_id,stage,admission,guard=callback),
one parent at a time, with stage simulate,relate or all. The exact original
full-panel generate_parent/infer_parent functions are reused. Model/feature
preparation is a separate strict952 H038 adapter, not included here.

CLI example:

```sh
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m arggnn.prepare_expanded_full_panel_v1 \
  --config ROOT_FROZEN_CONFIG \
  --expected-config-sha256 CONFIG_FILE_SHA256 \
  --expected-source-sha256 SOURCE_FILE_SHA256 \
  --parent ap_expanded_train_000 --stage all \
  --admission ROOT_PARENT_ADMISSION --execute
```

Default without --execute prints source-only status and registry. Root execution
requires both expected hash flags, frozen config and an active admission.
Admission fields are schema=expanded_full_panel_v1_root_admission,
root_authorizer=root_agent,execution_authorized=true,status=active,parent_id,
allowed_stages,output_directory,source_ref,config_content_sha256,
parent_registry_sha256,runtime_ref,expires_at_utc,outer_evaluation_authorized=false,
storage and storage_adapter_ref. The config content digest uses the inherited
canonical digest helper; expected-config-sha256 pins actual config file bytes.

The root storage_adapter_ref can pin the compatible original full-reservation
consumer or the new project_storage_closed_scopes_v1.py. The CLI verifies that
exact source before importing its unchanged budget_check_v9/validate_storage_v9
API. There is no accounting mutation during source-only authoring. Callable API
execution lets the root supply its admitted write guard directly. Inherited
per-parent cap is4GB with1GB native-step headroom,8Relate threads,2GB native chunk
memory and7200s timeout per step. Root owns CPU concurrency and all admissions.
Existing outputs cannot be overwritten. simulate can be followed by separately
admitted relate; its earlier expanded registry seal must match exactly.

Generation computes and seals external targets separately from observed files.
Relate reads the original observed whitelist only. Outer streams remain producer
products with evaluation_authorized=false; this source never opens outer target
archives for training, selection or evaluation. The original observation JSON's
legacy config-kind string is retained because producer functions are unchanged;
new expansion seal/completion metadata pins the actual expanded config/registry.
The original native focal-sampler3sequence metadata is retained for provenance;
H038 whole-forest processing does not consume that focal sampler.

## Strict952 observed handoff

After inference, each parent emits full_panel_observed_entry.json with schema
mutation_routed_full_panel_observed_entry_v1,parent_id,split=train|val|outer,
observations_only=true,truth_or_oracle_read=false,coordinate_contract and refs:
source_roster,vcf,site_coordinates,scoring_coordinates,native_tree,derived_roster,
inference_receipt,observation_manifest. The entry explicitly disclaims empirical
GRCh37 coordinates. Original site JSON is synthetic and lacks that optional
flag; the agreed parser safely treats it as false without altering observations.

full_panel_observed_inference_receipt.json binds the original source roster,
inferred tree, actual sample_index_to_inferred_tree_node_id map, coordinate
contract and original native inference receipt. No original inference file is
modified. Native source IDs are join indices, not model embeddings. This handoff
matches the new strict952 loader owned by the H038 trainer author.

The receipt records18 focused registry/recipe checks and concrete metadata
inventory. No actual simulation, Relate process, outer truth-array read, GPU,
install, Git or budget operation was performed. Existing sources are unchanged.
