# Official ARGformer execution and task bridge v1

This source/CPU handoff preserves genuine ARGformer rather than substituting the existing feature GNN or ARGMix implementation. It does not run the full author model, install dependencies, train, launch CUDA, prepare an actual cohort or read planned held-out inputs/targets. The prior sealed workflow audit remains unchanged.

## Author identity and reviewed sources

- Official repository: https://github.com/AI-sandbox/ARGformer
- Pinned commit: `f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596`
- Read-only archive directory: `.runtime/author_sources/argformer_f7545be7bca0`
- Archive SHA256: `44c1b76f7806dfc3063d707eae1950601eb4caa3ebe2fa20277bb98e77f1890c`
- Hash of the sorted 60 author-file path/SHA records, excluding the locally generated archive receipt: `2295d86ace359ac1a7d0522c1001b6f2822b30cf9b16200517ea3473f234ca5b`
- Paper: Bonet et al., *ARGformer: learning on ancestral recombination graphs with transformers*, Bioinformatics 42, btag438, 2026, https://doi.org/10.1093/bioinformatics/btag438
- Full text read: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13501305/fullTextXML

The repository LICENSE is Apache-2.0 and the attention lineage credits ModernBERT/MosaicBERT. File-level notices in `src/bert_layers/normalization.py` and `model.py` identify the active RMSNorm implementation as Meta Llama 2 Community License. The released YAML selects `rmsnorm`. Preserve those notices and have root review the complete provenance before distribution. This bridge does not vendor author code.

`validate_source()` hashes every author file and rejects changed bytes, symlinks, another source root or a changed archive identity. The author source is not edited. The CPU probes compile only reviewed class/function definitions in memory and execute them on handmade tensors. They do not import an upstream CLI or run its top-level preparation/training code.

## Original biological task

ARGformer learns contextual representations of **ARG-specific node identifiers** along full leaf-to-root marginal paths. Shared ancestral node IDs across marginal trees and haplotypes carry the intended relationship information. Masked-node pretraining predicts observed node tokens, with paper masking probability 0.30. Supervised contrastive fine-tuning uses known **reference population labels**, a learned pooling head and normalized embedding similarities. It is not a whole-tree feature classifier, and it is distinct from ARGMix.

The paper is an actual local-ancestry competitor. Section 3.1 benchmarks three-way local ancestry against FLARE on 10,000 unique marginal paths from admixed individuals, using reference-embedding PCA/cluster assignment and nearest-neighbor ancestry labels. The paper reports comparable accuracy and higher AFR accuracy for retrieval. This is a published downstream LAI application, not a reason to exclude the method. The released `retrieve.py` exposes normalized cosine top-k indices/scores and optional label metadata. It does not supply an integrated calibrated ANA/CHG/WHG/EHG classifier with raw four-class logits.

Native simulation in §2.2 uses AmericanAdmixture_4B18, 1,600 samples (1,000 admixed and 200 each European/African/East Asian), a 30-Mb chromosome, flat recombination `1e-8`, mutation `1.25e-8`, and inferred tsinfer+tsdate genealogies without phasing/genotype error. Unique marginal paths are split 90/10 for pretraining. Three continental source ancestries are the native LAI ontology. This differs from AncientEurope/Relate and the four-source task.

The real-data branch uses ancient/present-day human genealogies from the Wohns et al. unified genealogy. Archaic and Americas/Oceania analyses retrieve top-5 reference paths and report row-normalized label votes and chance-corrected enrichment. The Americas reference labels are Americas, East Asian and Oceania. These enrichment shares are not calibrated admixture proportions. The repository contains no `.trees`, dataset `.pkl`, model `.pt/.h5/.safetensors` assets or direct ARGformer checkpoint download. Paper data availability refers to the article/supplement. Access to exact author data/checkpoints, inferred genealogy version and supplement-specific settings remains to be established rather than assumed available or absent.

## Transductive and inductive regimes

The earlier exclusion based solely on the candidate's node-ID-renaming invariance contract is superseded by the user's requested genuine biological comparison. Invariance is a property to report for the candidate, not a universal comparator requirement.

The paper/source supports a **within-ARG transductive representation task**: build one vocabulary over that observed ARG, self-supervise on unique observed paths (including unlabeled query paths if prespecified), and fine-tune contrastively using only observed reference population labels. Query ancestry/census targets remain external and never enter MLM or contrastive data. Splitting paths in the same ARG is not an independent-parent or held-out-subject transfer experiment.

A checkpoint from one ARG cannot be assumed to encode a fresh ARG. Its node-token embedding rows have graph-specific meanings. The author raw tokenizer raises on an unseen node. Reusing integer `node_123` from an unrelated parent would introduce accidental semantics, and mapping unknown nodes to padding would change the method. This bridge binds each vocabulary to an explicit ARG namespace and fingerprints it. Separate ARGs with identical numeric node IDs have different vocabulary identities and reject cross-use.

For an inductive new-ARG study, root must freeze an explicit adaptation regime: for example, initialize a new node table and perform observed-only MLM adaptation, with its compute and data access reported. That is transductive adaptation at prediction time, not unchanged frozen-checkpoint inductive transfer. A namespaced union vocabulary across training ARGs is another possible study, but new unseen ARG nodes still require a defined policy. No union-vocabulary or adaptation training launch is implemented here. The current frozen ID predictor forbids fitting, so this regime needs a reviewed protocol/runner amendment rather than silently fitting inside that predictor.

## Exact released configuration

| Item | MLM YAML | Contrastive YAML |
|---|---|---|
| Architecture | FlexBERT, 12 layers, 12 heads, pretrained *configuration* `bert-base-uncased` | Same backbone, `num_labels=3`, learned mean pooling head |
| Hidden/intermediate width | Inherited from fetched BERT-base config (normally 768/3072), not pinned numerically in YAML | Same dependency on fetched config |
| Sequence length | 128 | 128 |
| Node vocabulary | All sorted unique raw node IDs plus PAD=0, MASK=1, CLS=2, SEP=3 | Exact same vocabulary/checkpoint namespace required |
| Leaf token | `skip_extant_tokens=true` | `skip_extant_tokens=true` |
| Position/attention | Sans absolute position embeddings, RoPE, unpadded | RoPE, unpadded |
| Local/global pattern | Window 128, global every third layer | Window setting not explicitly present, inherited config defaults need capture |
| Norm/activation | RMSNorm epsilon `1e-6`, GELU, pre-norm, final norm | RMSNorm epsilon `1e-6`, GELU, pre-norm, final norm |
| Masking | Train .30, eval .15 | No MLM masking in retrieval fine-tuning |
| Optimization | AdamW LR `5e-4`, betas .9/.98, epsilon `1e-6`, decay `1e-5`, 5 epochs | AdamW LR `2e-5`, betas .9/.999, decay `1e-6`, 1 epoch |
| Batch/precision | Global and microbatch 512, AMP bf16, seed 17 | Global and microbatch 4096, fp32, seed 36 |
| Temperature | MLM objective | YAML .3, paper/code default .05: explicitly capture which is reproduced |
| Loss | `fa_cross_entropy` | Weighted supervised contrastive InfoNCE |
| Output/checkpoint | Composer `state.model`, vocab resized/padded to multiple of 8 | Composer model with encoder/pooler prefixes |

The YAML is a released example, not proof that every reported paper experiment used those defaults. Numerical BERT-base configuration bytes must be acquired and pinned before model construction. `pretrained_model_name` fetches configuration and does not establish imported language-pretrained weights. The author factories load a local trusted pickle vocabulary before resizing embeddings. `embeddings.py` maps checkpoint prefixes and loads `strict=False`, then checks missing encoder/pooler keys. Root should additionally review unexpected keys, shapes, exact vocabulary identity and actual reload equality for a study.

Unique paths are deduplicated **before** leaf skipping or truncation, including across selected splits. Distinct leaves/full paths can produce identical model inputs. Never deduplicate by embedding vectors or truncated token inputs. Store every full path's original leaf ID and all physical tree-interval occurrences. For local ancestry, retain local path embeddings and map them back to every original coordinate. `aggregate_embeddings.py` averages unique paths per haplotype and haplotype means per individual for global visualization. That averaging discards local interval distinctions and cannot replace a local readout.

`pooling=first_node` selects position 1 after CLS. With skip-extant true, this is the **first ancestor**, not the extant leaf. The paper's pretraining ablation text mentions extant-node pooling, so skip/pooling settings must be reconciled with the actual corresponding experiment/checkpoint. `mean` includes nonpadding special tokens. `head/head_norm` applies the trained contrastive pooling head and requires the corresponding fine-tuned checkpoint.

## Dependencies and honest fallback boundaries

Two supplied dependency specifications conflict. Choose and record one coherent author-compatible environment rather than mixing them or downgrading the shared project environment.

| Source | Torch / CUDA | Transformers | Composer / streaming | Other |
|---|---|---|---|---|
| `requirements.txt` | Torch 2.3.0, Triton 2.3.0 | 4.40.2 | mosaicml NLP/WandB >=.22,<.23; streaming .7.6 | OmegaConf 2.3.0, einops .8.0 |
| `pixi.toml` | Python 3.11.9, Torch 2.4.0 CUDA 12.4, CUDA toolkit 12.4 | 4.44.1 | mosaicml .24.1; streaming .8.0 | NumPy 1.26.4, tskit 1.0.0, torchmetrics 1.4.0.post0, Optimi .2.1, compilers, Ninja |

The present project runtime has Torch 2.6.0+cu124, NumPy 2.4.6 and tskit 1.0.3. OmegaConf, Transformers, Composer, streaming, torchmetrics, einops, FlashAttention and Optimi are missing. Its package inventory is in the source/CPU receipt. A complete official-model import/forward has **not** occurred.

The README calls FlashAttention optional for the base environment, but unchanged released MLM execution requires more than global SDPA fallback:

- RoPE classes import rotary implementations from FlashAttention and assert they exist. The unpadded rotary implementation imports a FlashAttention Triton kernel.
- Sliding-window attention is explicitly unsupported by the stock SDPA branch and raises when the local window is enabled.
- `fa_cross_entropy` is only registered when FlashAttention loss imports succeed.
- A global, non-RoPE, cross-entropy SDPA configuration is a disclosed backend/architecture variant, not a drop-in released-config reproduction.

The author README requests `flash_attn==2.6.3 --no-build-isolation`, with an already CUDA-enabled Torch build, CUDA/compiler toolkit and Ninja. H100 FlashAttention3 is optional and separate. Root should first create a project-contained isolated environment using the coherent Pixi versions or a reviewed equivalent, preserving the shared runtime. Root owns all installs/builds/GPU scheduling. After containment review, test unchanged author model import, then a tiny label-free GPU forward under the project launcher, and verify masked objective, contrastive pooler, checkpoint/vocabulary roundtrip and attention backend. A faithful stock RoPE/window backend would require explicit equivalent implementation/equivalence tests and should be disclosed as such.

## Containment review before upstream execution

The preparation script performs `os.makedirs`, `tskit.load` and pickle writes at top level, including a logs directory under the author source. It must not be imported/run unchanged. Root should construct a reviewed execution copy/config or use the bridge's observed-only extractor without editing the pinned archive. The preparation script is not a biological simulator or inference pipeline.

`main.py` and `sequence_contrastive.py` set `WANDB_DIR` to `~/wandb` when absent. Author YAMLs enable WandB and placeholder save/load paths. Factories call Hugging Face `from_pretrained` for config, and Composer/loaders can access remote storage. Before execution, root must set project-only caches/TMP/WANDB paths, disable external logging, freeze local/offline model configuration, resolve every input/output/checkpoint path inside the repo, and validate trusted pickle/checkpoint artifacts. No upstream scripts have been run in this handoff. The pinned author source must remain read-only, including bytecode writes.

## Common-data local-ancestry adapter

`official_argformer_bridge.py` supplies:

- `extract_observed_paths`: topology/node-ID-only extraction in author sample-then-tree order, full-path deduplication and physical occurrence joins. It accepts an already reviewed observed `TreeSequence` in memory and opens no data file.
- `build_vocabulary` and `tokenize_official`: per-ARG namespace/vocab checks, actual unchanged author tokenization methods on in-memory paths, explicit skip/truncation and original leaf metadata.
- `PathCorpus.readout_path_ids`: exact original query order and half-open global physical coordinates. It preserves repeated occurrences of a unique full path at different intervals.
- `downstream_reference_similarity_logits`: a separately labeled four-source adapter over extracted official path embeddings, using mean exponentiated cosine similarity per observed reference class and an explicit pre-frozen temperature. It uses no query ancestry targets and fits nothing. It is not the native top-5/PCA rule or an author supplied calibrated classifier. All four source classes need observed reference support.
- Reviewed CPU components: actual author global padded attention forward/backward; actual author contrastive loss with an explicitly toy encoder/pooler; actual author pooling and streaming top-k retrieval on handmade hidden states/embeddings. No toy encoder is returned as ARGformer.

Common comparisons must share the same biological observed genealogy, physical coordinates and ordered query/reference identities while retaining each method's native representation. Reference contrastive labels must be observed population labels only, not census ancestry on query paths. For the common task use the same 24 references and 64 queries unless root amends the panel. Report transductive data/compute separately from an inductive GNN or AncestralPaths model, and match selection/tuning budgets on development data. Original ARGformer three-source/30-Mb/tsinfer+tsdate reproduction is a separate branch from this AncientEurope/four-source/Relate adapter. AncestralPaths additionally has a native six-path ontology at census350 generations, whose four-source collapse is secondary. ARGMix's paper/native reference and feature settings also require separate capture. No native result should be relabeled as a common-task exact reproduction.

## Validation and next concrete step

Run on anogena with existing project Python and empty CUDA visibility:

```sh
source scripts/project_env.sh
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
.runtime/env/bin/python -m pytest -q -p no:cacheprovider tests/test_official_argformer_bridge.py
.runtime/env/bin/python scripts/check_official_argformer_source.py --synthetic-cpu --output FRESH_PROJECT_RECEIPT
```

The first focused run passed 10 tests in 2.64 seconds before two additional loss/overflow cases were added. The next run passed 12 tests in 3.74 seconds, and the final run with author pooling/retrieval coverage passed **13 tests in 3.11 seconds**, exit 0. The source/CPU CLI also completed, exit 0. Historical receipts, final exact source hashes and component evidence remain under `.runtime/tmp/official_argformer_source_cpu_v1/`. These are synthetic component/interoperability checks only.

The immediate action for root is to choose an isolated dependency stack and review the native-task/data/checkpoint acquisition and transductive access protocol, then authorize a contained full-model import and tiny official forward. The bridge is ready for that stage, but actual official-model training, checkpoint reproduction, biological evaluation, FLARE comparison and real-data hypothesis testing remain unperformed.

A final head distinction matters: FlexBertForSequenceClassification contains a class-logit layer, but the released ContrastiveModel.forward calls only the encoder and pooling head and applies similarity-based contrastive loss. It never calls that classifier. Therefore num_labels=3 in the YAML does not establish a trained three-class ancestry head. Use the trained embedding/pooler and native retrieval rule, or train and disclose a separate downstream classifier.
