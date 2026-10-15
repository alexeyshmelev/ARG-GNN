# H018 selected H045 empirical CPU application source

Use inference_v2.py / config.active_template_v2.json / source_manifest_v2.json. The retained inference_v1.py and v1 config/manifest are unexecuted drafts tied to the earlier current-v3 proposal and are superseded. V2 scientific inference imports no guard and has no accounting-mode or policy selection.

Actual target: data/H018_empirical_observation_bank_root_v3/completed.json SHA fb33cd56fdca047148747521b03b53b6dce8adb6a0c3184346e47ac1510284c8. Its observation_bank.pt SHA is34eca72dddb0a9491aed307129a1ace5941ec794e262cf87f3415689053c2487. All355 native holds,2074 original observed SNPs including277 constants,40 haplotypes and24 original query haplotypes are retained. Source preparation uses originalVCF codes and graph node/edge-only Relate semantics. No bank/VCF/native preparation is rerun. H045 episode memberships are derived only in RAM from saved parent/sample/span tensors, using the original exact episode-array algorithm; their flags must equal the originally saved aligned flags.

Selected checkpoint: experiments/results/H045_GPU5_episode_train_seed101_v1/foreign_episode.pass_30.pt SHA bfd8a87632015a900d9da6009d4b0ee037958460038d7949d0998e80b770796b. checkpoint_descriptor.json pins the actual root selection summary, original paired TRAIN completion/config and learned scientific refs. The current guard requires metadata/source suffixes, so this descriptor is directly bound and the checkpoint ref is also explicit in the active config. Checkpoint bytes are SHA-verified by the injected root reader and again immediately before CPU torch.load. No checkpoint payload was opened during this source task.

There is no learned architecture blocker. Frozen H045/H042 constructors/convenience forward impose952/Q686 gates, while learned operations and the raw exact episode-array builder are count-generic. variable_panel_model_v1.py is a separate disclosed I/O adaptation for explicitly qualified40/Q24 and952/Q686 pairs. At40/Q24 it reuses the existing reviewed H018 VariablePanelFlagInterfaceV1 constructor/count/device gates and identical full-span geometry/GRU/head arithmetic, retaining the original inherited H045 encode_trees, query selection and readout methods. All31812 state parameters must load strictly with exact keys/shapes/finite FP32 values. No learned layer, width, state transform, padding, donor replacement, optimizer, fine tuning or training update is introduced. This is variable-panel application, not exact original fixed-panel execution or an empirical accuracy result.

15 tiny FP32 CPU source tests passed. With identical synthetic initialized states and nonzero flag/episode projections, original952/Q686 logits and all952 outputs are bitwise equal (maximum absolute difference0). The same complete state loads into40/Q24 and yields finite3x24x4 synthetic probabilities with unit row sums. These synthetic tests used599785472B measured peak RSS. They read neither actual bank nor checkpoint and establish source/interface behavior, not empirical accuracy or live application capacity. V2 additionally passed23 source/static checks; the prior pure PCA/SVG source checks remain identical.

Root invocation is inference_v2.execute_v1(ctx), after the root's actual admission and with an actual separately pinned IO context. See root_IO_contract_v2.json. Required ctx keys:

- root_authorizer='root_agent', admitted=True
- input_ref: actual root-authored input JSON using the provided inactive v2 template, with execution_authorized=true, GPU_authorized=false, optimization_authorized=false, root_host_resource_enforcement_active=true, actual producer/config/runtime/output/job/expiry refs
- source_ref, config_ref, runtime_ref and cfg: exact actual active records and decoded config
- consumer_bindings: the actual reservation's complete direct refs, no more than64
- io_context_ref: actual pinned root IO source; resource_binding_refs: actual configured guard/policy/authority/marker/helper/launcher refs, all directly bound
- reader: object with read(exact_ref)->bytes, performing actual root-authorized immutable SHA/size checks
- writer: callable(name,bytes)->exact project-relative path/SHA, using actual configured guard, fresh exclusive output scope and exclusive fsynced writes
- before_stage: callable(stage,plan), performing actual root storage/source/cap validation and resource checks

Root's IO can use the currently authorized or newly activated resource guard without editing scientific code. The entrypoint validates all21 scientific source refs,11 metadata refs and its actual producer/config/runtime/manifest. These give35 base direct refs before adding actual root IO/resource refs. Root must bind those actual additions through its chosen admission. Source v2 never installs or monkeypatches any guard. Root sets project cwd, project Python/src imports and CUDA_VISIBLE_DEVICES='' before import, with no CUDA API/context.

Keep the exact20,000,000B allocation, CPU4 threads,2,000,000,000B RSS plan and1800s wall deadline. Root owns the actual outer Popen monitor/direct wait, with minimal fsynced actual wait evidence before risky final guard/log writes. RSS polling is a soft monitoring bound, not a kernel quota. The application also checks its own peak RSS/deadline. before_stage runs before payload reads, selected checkpoint loading, scientific forward and each planned output write. Complete output growth is preflighted at6MB and serialized artifact payloads are capped at5MB, within20MB; no new bank/checkpoint copy is saved.

Useful outputs after real admitted execution:

- probabilities_all355_holds.npz: exact355x24x4 FP32 probabilities, class order ANA/CHG/WHG/EHG, every native-hold interval/readout and original query column
- query_bpweighted_proportions_and_PCA.tsv:24 original query haplotype IDs/dates, posterior ancestry means weighted by genuine physical spans, and PC1/PC2
- query_genotype_PCA.npz and query_genotype_PCA.svg: descriptive centered, unscaled original query per-site four-allele indicator PCA over every2074 site; missing calls use query allele means and are counted; deterministic loading sign
- bpweighted_ancestry_proportions.svg: descriptive posterior composition bars
- provenance.json and completed.json: complete refs, adaptation/resource statements and preparation failure/recovery lineage

Proportions cover only regional[427,9997139), absolute[20000427,29997139), exactly9996712bp. No missing edge fill, extrapolation, common100bp scoring grid or2Mb reset is introduced. PCA uses24 haplotypes from12 subjects, not24 independent subjects, and does not use predicted ancestry or truth as input. It is not ancestry/demographic validation. The empirical pre-size fixedN30000 graph, R16/Q24 donor sampling, ascertained SNPs/constants and355-hold near10Mb recurrent history differ from952-panel simulated2Mb training; original feature normalizations and absent-donor masks remain unchanged.

The original preparation child's direct exit1 and root completion-only metadata recovery remain explicit in output provenance. The native lossless genotype-export gate stays failed. No query ancestry truth, empirical accuracy, calibration, benchmark superiority or demographic interpretation is asserted.

No actual bank/checkpoint payload, empirical model inference, guard/admission, GPU/native execution or Git operation was performed by this source task. Root owns actual application and reporting.
