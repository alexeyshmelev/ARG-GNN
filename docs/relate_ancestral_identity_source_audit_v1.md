# Relate ancestral identity source audit v1

Source-only finding: the current biological trees have tree-local internal IDs because the official RelateFileFormats ConvertToTreeSequence exporter deliberately allocates new internal nodes for every marginal tree. This behavior matches official documentation and the pinned/current source. The producer's cleaning preserves those IDs. It is not a newly discovered ARGformer tokenizer bug.

Recommendation: create a separately versioned native_ARGformer_tsinfer_tsdate_biological_v1 condition from the same observed phased genotypes, 280-haplotype panel, observed dates and physical scoring coordinates, using the inference family in the published ARGformer benchmark. Do not merge existing internal IDs or repeat the current native fits on tree-local IDs as the remedy. Root must freeze and qualify that new observation producer before preparation/fitting.

No frozen source/config/data producer was changed. This task opened only source, documentation and root's observation-only identity receipt. It did not open the four tree-sequence arrays again, packed paths, truth/score/checkpoint arrays, or run fitting, conversion, inference, GPU, installations or Git.

## Connected identity evidence

Reuse root's corrected receipt, experiments/results/relate_observed_identity_audit_v1/receipt.json, SHA256 8b575be1728fc8d25a069f8d217272afd2ef2e34dc01aabc7baa76c0bdd6af23. Its method unions ancestors reached by walking tree.parent from every actual sampled haplotype. It excludes isolated non-sample nodes. Root's earlier occurrence counts from its tree.nodes traversal included isolated roots and are superseded.

| Parent | Trees | Samples | Connected internal occurrences | Unique connected internal IDs | Internal nodes/tree | Adjacent shared internal IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bio_paths_pilot_train_000 |955|280|266,445|266,445|279|0 throughout|
| bio_paths_pilot_train_001 |897|280|250,263|250,263|279|0 throughout|
| bio_paths_pilot_val_000 |946|280|263,934|263,934|279|0 throughout|
| bio_paths_pilot_val_001 |920|280|256,680|256,680|279|0 throughout|

The counts are root's factual observation audit, not recomputed here. Every connected internal occurrence is unique to its tree, rather than merely absent from the neighboring tree. These four inputs contain persistent sample IDs but no persistent internal node IDs.

Consequently, every full sample-to-root integer-ID path is unique to its tree/sample. The logically implied full-path count is 280 times tree count: 267,400, 251,160, 264,880 and 257,600. This follows from the connected-ID finding and the extraction key. It is not a second inspection of corpus/checkpoint arrays. Full-path deduplication cannot recover sharing that the exporter did not preserve.

## Exact producer and official export source

scripts/infer_biological_paths_relate_v1.py:371 and 379 invoke:

~~~text
RelateFileFormats --mode ConvertToTreeSequence -i pre_size -o pre_size
RelateFileFormats --mode ConvertToTreeSequence -i post_size -o post_size
~~~

No association option is supplied. The producer source SHA256 is a63a79b36a422f7346f2a35206f41b35fb440053edcfe4e080755c649ed737e9 and configuration SHA256 is 7da0bae298c0927fd0fce6a6ee320f0c51fc459b901de9c6f0d1e225cae05de9.

The existing config pins [include/file_formats/ConvertToTreeSequence.cpp](https://github.com/MyersGroup/relate/blob/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/file_formats/ConvertToTreeSequence.cpp#L338) at official commit b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74, SHA256 452298c9a5ad60159d9f86dd402f71798e8a88108f2bb217fcb941d28778bfc4. Freshly retrieved pinned source matches that hash. The official repository's current master resolves to the same commit and converter bytes.

Source mechanism:

- Lines 338-348 allocate the N sample nodes once, preserving .anc sample-age/order information.
- Lines 383-384 set node_const=tree_count*(N-1).
- Lines 419-427 append all N-1 internal nodes on every tree iteration.
- Lines 430-433 shift internal child/parent labels by node_const when writing edges.
- Internal mutation-node pointers receive the same offset at lines 390-398.
- Sorting/dumping at 443-451 does not identify equivalent ancestors.

For N = 280, each tree receives a new 279-node block. The formula accounts for root's connected counts and total node counts exactly.

The official [TreeSequence documentation](https://myersgroup.github.io/relate/input_data.html#TreeSequence), saved in the audit directory, says: “each tree is stored with new nodes in the tree sequence file format, leading to no compression.” It also states that information about how long branches persist and how many mutations map to a branch is lost in conversion.

clean_converted_tree at producer lines 279-343 clips/shifts genomic intervals, strips hidden annotations and metadata, sorts tables and explicitly checks that node count/sample mapping are unchanged. It does not relabel or merge internal nodes. A correction must target a new observation/representation condition, rather than pretending cleaning introduced the lack of sharing.

The code/reference commit is source evidence, not a signed assertion of the release binary's compilation identity. The release executable remains separately pinned at cb19cb6e277c27704de506f0e8155f8f53d9c3dc8d0f52615a059caa253332ee. Official documentation, source allocation and root's actual output audit independently agree on default behavior.

## There is no demonstrated association export flag

The inspected official RelateFileFormats dispatcher declares a generic --flag option, but dispatches ConvertToTreeSequence directly. Neither the binary nor text converter function reads flag or invokes branch association. Adding --flag 1 to the existing command is therefore not a supported identity repair in the inspected source.

ConvertToTreeSequenceTxt has the same tree-local allocation pattern. The inspected RelateExtract dispatcher has no shared-node tree-sequence export mode. No inspected official converter supplies a lossless association switch. This statement is scoped to the pinned/current official implementation and documentation, not every third-party program ever written.

Relate's equivalent-branch mechanism belongs to inference/branch bookkeeping:

- include/pipeline/FindEquivalentBranches.cpp:92 and 107 call AncesTreeBuilder::BranchAssociation on neighboring trees and store equivalent_branches maps temporarily.
- anc_builder.cpp initializes threshold_brancheq=0.95 at 364/382. BranchAssociation at 1440 onward uses descendant and parent-descendant correlations, tries near-exact matches at 1540-1555, then approximate matches above the threshold at 1573-1595.
- AssociateTrees at 669 onward propagates num_events and SNP_begin forward and SNP_end backward. It does not allocate persistent tskit coalescent-node IDs.
- AncesTree::AssociateEquivalentBranches in anc.cpp:1862 onward similarly associates branch bookkeeping with threshold 0.95.
- .anc node serialization stores local parent, branch_length, num_events, SNP_begin and SNP_end. Those fields preserve estimated branch-persistence information, rather than a supplied universal ancestral-event identity.

Branch equivalence is an inferred, partly approximate relationship between branches. It is not a theorem that the endpoint vertices are the same biological coalescent event. The default exporter does not consume those relationships to create shared nodes.

## Why arbitrary ID merging is not a correction

Removing tree_count*(N-1), equating local node ranks, hashing descendant clades, rounding times, or matching SNP spans alone would impose unverified identities. A recurring descendant set need not be the same coalescent event. Approximate 0.95 branch matches can even have different descendant sets.

A tskit node has one time across its genomic occurrences. If an inferred corresponding node has time 10 in one tree and 11 in another, sharing one tskit ID cannot preserve both times. A converter must either establish equal time, change the times under a disclosed model, retain tree-local nodes, or represent correspondence separately. Equal local topology does not solve that constraint.

A proposed shared-node conversion would need explicit provenance for every adjacent correspondence, one-to-one/injective mappings within each tree, globally acyclic parent relationships, valid shared time constraints, exact sample/age/order mapping, coordinate/interval/mutation joins, and measured topology/time differences. Relate's current converter also applies local time-order repairs before writing, so “exact original .anc times” and “identical current exported times” are different validation targets.

No available inspected flag is shown to preserve all per-tree topologies AND times while naturally supplying persistent ancestor IDs. A Relate-derived correspondence sidecar could retain original marginal geometry and record upstream estimated branch links, but it would be a new explicitly approximate adapter. It must not be silently renamed a lossless native coalescent-event representation. No such adapter or conversion is implemented here.

## ARGformer paper and source expectation

The published ARGformer article (DOI [10.1093/bioinformatics/btag438](https://doi.org/10.1093/bioinformatics/btag438)) at [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13501305/fullTextXML) has SHA256 53ab26710804f0b9ca6f355095893b365413caf56cf7bb722b39907f6db0619f, matching the earlier source audit. Section 2.2 explicitly says the simulated-genotype ARG was inferred using tsinfer+tsdate, without phasing/genotype error. The real-data branch uses the Wohns et al. unified genealogy. The reviewed paper contains no exact-word Relate occurrence or Relate conversion recipe.

The paper describes paths “exploiting shared coalescent events across paths” and says: “ARGformer leverages shared ancestral nodes to capture the topological intersections of the ARG.” It explains that this captures relationships across trees and ancestry changes induced by recombination.

At author commit f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596, [prepare_data_pretrain.py:40-46](https://github.com/AI-sandbox/ARGformer/blob/f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596/src/data/prepare_data_pretrain.py#L40) constructs a tuple of raw leaf-to-root node IDs. Lines 58-80 deduplicate that tuple across trees. Lines 127-152 assign one token per observed node ID. The author README says node tokens are specific to the ARG and requires deduplicating full paths before embedding. ARGDataset maps node_X to the corresponding vocabulary token and drops the extant token when skip_extant_tokens is true.

The project's official_argformer_bridge.py:109-129 matches that full-path extraction rule and preserves occurrence joins. Its vocabulary is namespaced to the observed tree hash. It has not invented shared ancestor IDs. The owned biological adapter supplies the unchanged default Relate export, then trains a separate native vocabulary/model per ARG.

The released code can numerically process tree-local IDs and does not assert a minimum cross-tree sharing rate. That is execution compatibility, not demonstration of the intended shared ancestral-node representation. Within-tree ancestors are still shared between haplotype paths. With extant tokens skipped, some distinct full paths can have identical ancestor inputs, as the author documentation acknowledges. Do not deduplicate those truncated inputs or reinterpret this as restored cross-tree ancestry.

No exact-word Relate mention was found in the pinned author's Python/Markdown/YAML/TOML/text files. The published tsinfer+tsdate family is the available native observation reference, not a documented alternate Relate converter.

## Specific next condition and comparison labels

Run a separately versioned native_ARGformer_tsinfer_tsdate_biological_v1 observation condition after a source-only producer plan and capability review. Inputs must remain the same observed phased genotypes and panel roles/dates, all 280 haplotypes, 96 pure references and their labels, 64 mixed haplotypes with unlabeled roles, 120 actual query haplotypes, class order and fixed physical output lattice.

Before any new preparation, freeze actual tsinfer/tsdate versions, native-compatible defaults, ancient/nonzero sample-time support, polarization/build conventions, variant/site/sample joins, inference/dating seeds, coordinates, coverage and failure policy. Do not silently zero ancient dates, change the panel, add truth, discard difficult samples or invent new target eligibility.

Use the common original 100 bp output coordinates and explicit shared physical scored support for the main paired comparison. Record any native-inference support difference separately. Query ancestry targets and all truth remain outside observation inference, dating, vocabularies, MLM and contrastive reference labels. The observed query haplotypes remain part of the unchanged 280-haplotype observation panel. Each new inferred ARG receives a fresh hash namespace, vocabulary, corpus and checkpoints. Current tree-local-ID vocabulary/checkpoints cannot be transplanted as if they described the new ancestors.

| Comparison | Interpretation |
| --- | --- |
| Native ARGformer on tsinfer+tsdate versus AP/ARGMix/candidate on default Relate, same observed genotypes/panel/output support | End-to-end method/inference-pipeline comparison. Differences include inferred topology/time/identity and cost, so model-only superiority is not identified. |
| Models evaluated on the same tsinfer+tsdate trees, with native input/label scopes disclosed | Same-tree representation/model comparison. Additional candidate/competitor preparation is a separately frozen cell, not automatically authorized by this audit. |
| Existing models on the same default Relate trees | Valid common-data marginal-tree comparison for AP/ARGMix/candidate. Existing ARGformer runs retain the tree-local-ID input condition label and upstream-adequacy limitation. |

Default Relate marginal-tree topologies/times remain usable by AP, native ARGMix and the candidate. Their current information does not require persistent anonymous ancestor IDs across trees. Candidate persistence is in sampled query states, with anonymous interiors reduced separately per tree. Native ARGMix uses per-tree observed topology/TMRCA plus its own GRU sequence. The identity finding alone does not invalidate those inputs.

Keep all current products and native execution receipts. Label existing ARGformer results, if later reported, within_ARG_transductive_on_default_Relate_tree_local_internal_ID_v1. This is a condition description, not an assertion that a run failed or had a particular ancestry score. Root's receipt already records upstream ancestral-correspondence adequacy under review and no admitted superiority claim.

Do not repeat current native fits to address this issue. Root should register the new observation producer and its concrete inference-family change, get independent review, then qualify actual observation-only conversion/dating and connected identity before native preparation/fitting. No current pilot is retroactively an exact paper reproduction.

## Scope and handoff

The audit directory preserves official source/docs, paper XML, retrieval provenance and machine-readable findings. All observed sharing counts are attributed to root's corrected receipt. No four-tree array audit was repeated.

Outstanding work is a source-only tsinfer+tsdate producer plan with observation-only conversion/date capability checks, followed by separately authorized actual inference/admission. It is not another native fit on unchanged tree-local IDs or a guessed Relate export flag.
