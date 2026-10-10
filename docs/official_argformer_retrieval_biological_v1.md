# ARGformer corpus retrieval on biological development parents v1

This separately named readout searches the complete pure-reference path corpus within each observed ARG. It reuses completed native mean-head embeddings and preserves native checkpoints, original exports and the earlier same-coordinate mean-exp readout. No retraining, model inference, calibration or target-driven corpus selection occurs. This is an adapted biological C4 retrieval comparison, not reproduction of the paper's American-admixture task.

## Evidence and prespecified choices

Verified ARGformer full text, DOI 10.1093/bioinformatics/btag438, describes simulated LAI queries as unique admixed marginal paths retrieving reference-haplotype embeddings. The simulated LAI paragraphs do not state k, distance implementation, corpus curation or tie order. The main text explicitly uses top-5 unweighted row-normalized neighbor vote shares for archaic and Oceania-America real-population analyses. Those enrichment analyses are not the simulated LAI experiment.

The unchanged current author retrieve.py performs F.normalize(p=2,dim=1) on corpus and queries, computes dot products and returns descending top-k neighbors. Both its function and CLI default to k=1. k is configurable. It does not provide a separate LAI majority-vote implementation or a deterministic tie rule. Its torch.topk does not define stable tie order.

Before any retrieval outcomes, root accepted source-default k=1 as primary and paper-explicit k=5 as a prespecified sensitivity. Both use equal neighbor votes divided by k. No similarity temperature, class-density mean, class balancing, span weighting, softmax, pseudocount, probability floor or calibration is applied. k is never selected from VAL scores.

The adaptation resolves exact computed cosine ties by ascending original full-path ID, after descending score. C4 class-vote ties use the common scorer's first argmax in ANA,CHG,WHG,EHG order. These deterministic tie choices are disclosed implementation choices, not claimed original-paper tie behavior. FP32 NumPy BLAS dot products may differ in their last bits from the author's PyTorch/device-specific matrix products. A handmade rotated-vector check observed differing near-tie neighbor order despite cosine-score agreement within1e-7. The readout does not claim bit-identical native device results and uses the fixed declared CPU block shapes, threads and actual library receipt. This implementation is exhaustive and exact under its declared computed score and tie rule, with no approximate nearest-neighbor index.

Supplement retrieval through the linked EuropePMC supplementary endpoint did not complete. One attempt lost its SSH transport and one bounded curl attempt timed out after30 seconds. Those attempts are recorded. No missing supplement details are inferred and no exact simulated-LAI fidelity is claimed.

Evidence: [data/literature/argformer_btag438_source_check_v1/body.txt:49](/workspace-SR003.nfs2/ARG-GNN/data/literature/argformer_btag438_source_check_v1/body.txt:49) lines49–51 describes simulated LAI, while lines57–67 describes top-5 votes in real-population analyses. [.runtime/author_sources/argformer_f7545be7bca0/retrieve.py:13](/workspace-SR003.nfs2/ARG-GNN/.runtime/author_sources/argformer_f7545be7bca0/retrieve.py:13) lines13–16 normalizes, lines78–97 implements streaming dot/top-k, and lines118–148 sets configurable source-default k=1. [.runtime/author_sources/argformer_f7545be7bca0/README.md:135](/workspace-SR003.nfs2/ARG-GNN/.runtime/author_sources/argformer_f7545be7bca0/README.md:135) lines135–164 discusses unique full paths and single-ARG namespaces.

## Corpus, query paths and class adaptation

Each parent retains its own namespace parent_id:inferred_tree_sha256 and its own completed native MLM5 plus contrastive1 fit. Vocabularies, checkpoints, corpus rows and embeddings are never mixed across parents. The encoder remains within-ARG transductive. Pure-reference source labels drove native contrastive training. Query ancestry targets did not train that encoder and no query-supervised ancestry classifier is added.

The corpus contains every unique full leaf-to-root path whose actual inferred leaf is one of the 96 observed pure-source haplotypes. It includes paths from both native 90/10 path splits and all observed marginal trees, even when an interval contains no retained scoring center. Mixed64 and query120 leaves are excluded from the donor corpus. Original full-path identity includes the leaf before source tokenization skips it. Repeated physical occurrences of one full path contribute one corpus row. Distinct full paths remain distinct rows when truncation, skipped extant tokens or identical embeddings make their vector representations equal.

Each retained original100bp query center uses its exact prepared query full-path ID. All distinct query path IDs required by those centers are searched once, then their raw probabilities are copied back to every corresponding center. This optimization does not discard any retained query support or infer targets. The corpus is not restricted to references at the same coordinate. It is not formed by repeating references once per100bp center.

The original corpus pickle and source embedding raw-sequence pickle are hashed and read through a primitive-only unpickler during actual execution. Their full-path order must agree. Physical occurrences are checked as complete half-open partitions per actual leaf. Every query/reference center's prepared path ID must equal the original occurrence join. All280 observed leaves are retained in the full path corpus and true query/ref identities remain disjoint.

Completed native_path_embeddings.pt contains a float32 contiguous finite CPU tensor and native_pooling=trained_author_mean_head. Root metadata evidence gives VAL000 shape264880x768 and VAL001 shape257600x768, with file hashes matching completed fit receipts. Actual execution checks dimensions, native pooling, tensor hash and row order before retrieval. This source-only handoff did not deserialize these real tensor payloads.

Biological donor labels are 0 ANA,1 CHG,2 WHG,3 EHG, with24 pure haplotypes per class. Replacing the paper's three source populations with these four labels is a disclosed task adaptation. The unused three-class native classification layer is not used, retrained or reinterpreted. The embedding geometry comes from the preserved four-reference-class contrastive fit.

## Raw votes, support and scores

For query full path q, retrieve the k corpus path rows with highest source-normalized cosine. Raw C4 probability p_c(q) equals the number of retrieved rows with observed class c divided by k. For k1 this is exactly a one-hot vector. For k5 components are multiples of1/5. Saved cosine scores are retrieval similarities, not ancestry logits.

Both k1/k5 full query120 probability grids, original neighbor path IDs/scores and tie-count diagnostics are written and hashed for both VAL parents before the first external truth capability. A missing or inconsistent path/coordinate join fails the stage. It never removes bins, queries or denominators based on targets.

The frozen common scorer integrates prediction bins against complete external census C4 truth with exact physical intersections. It reports present64 primary, ancient56 secondary and all120 groups. Support is the existing retained-center bins clipped to observed coverage, with original2Mb normalization, no uncovered-tail extrapolation and no purity/eligibility filtering. The same frozen source functions supply accuracy, macro F1, balanced accuracy, confusion, Brier, calibration, dosage and tract/boundary summaries.

Accuracy and macro F1 are the principal retrieval endpoints. Raw NLL remains uncapped. If any positive truth mass receives zero probability, NLL is infinite, represented by a null numeric value plus nll_is_infinite and affected query-bp. No auxiliary soft probability or finite substitute is introduced. Macro F1 retains the common scorer's definition over classes with true support. Query haplotypes, bins and path rows are not independent biological replicates.

The original mean-exp same-coordinate readout is preserved as its own adapted endpoint. It is not relabeled published retrieval and is not used for choosing k, corpus, ties or scores.

Frozen scoring: [scripts/score_official_argformer_biological_v1.py:179](/workspace-SR003.nfs2/ARG-GNN/scripts/score_official_argformer_biological_v1.py:179) lines179–311 checks native fits and actual joins, lines373–405 admits truth after both seals, and [scripts/train_ancestralpaths_biological_v1.py:545](/workspace-SR003.nfs2/ARG-GNN/scripts/train_ancestralpaths_biological_v1.py:545) lines545–675 defines common full-span metrics. Native embedding row order is established in [src/arggnn/train_official_argformer_biological.py:731](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/train_official_argformer_biological.py:731) lines731–750 and full-path preparation in lines224–263.

## CPU cost and admission

The exact pure-reference corpora have90,816 and88,320 full paths for VAL000/VAL001. Metadata-only upper bounds for unique query paths, including mixed leaves, are174,064 and169,280. These conservative bounds imply15.81 and14.95 billion dot pairs, or24.28 and22.96 trillion floating-point operations at width768. Actual execution searches only query120 paths present on the retained grid and should be below those bounds. No wall-clock claim is made without execution.

The implementation holds CPU vector blocks and streams query batches128 against corpus chunks8192. A score block uses4,194,304 bytes, with a similarly sized selection workspace. It never allocates a full query-by-corpus distance matrix. Original embeddings are loaded once per VAL parent and released before the search after selected query/reference normalization. Parent stages execute sequentially.

BLAS, PyTorch and environment thread settings are explicitly limited to4, with Torch interop1. threadpoolctl records actual libraries and bounds actual BLAS pools. CUDA visibility must be empty and CUDA initialization is checked to remain false. Each parent is admitted only below40 billion exhaustive pairs. Insufficient corpus/support or cost admission fails without ANN substitution or target filtering. The stage uses the existing repository storage guard with2GB output headroom. No GPU, package installation or new native/model execution is required.

## Root execution

New files are scripts/score_official_argformer_retrieval_biological_v1.py, configs/official_argformer_retrieval_biological_v1.json and this document. Default preflight reads frozen source/config, completed JSON receipts and NPZ headers only. It hashes bound input files but does not deserialize native embeddings, corpus objects, joins or truth. Actual retrieval and scoring require the explicit execute flag, reviewed config hash and fresh output.

~~~bash
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" \
  scripts/score_official_argformer_retrieval_biological_v1.py

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" \
  scripts/score_official_argformer_retrieval_biological_v1.py --test-handmade
~~~

The default root inputs are exactly experiments/results/official_argformer_biological_readout_v1_sequence/root_inputs.json, SHA28abd1777ca982024cb3f527cfca64c591b5f54853a18064c65264a275561921. Metadata evidence is experiments/results/official_argformer_retrieval_biological_root_schema_v1/schema.json, SHAbf3a49e9738eb212e6032d8d518f6e1a82b73a4eeb4cd408dc5d464b7cb695b7. All four native preparation/fit receipts must remain passed and unchanged, although only the two VAL readouts are computed.

After review, root runs a dedicated fresh output:

~~~bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$ARGGNN_PYTHON" scripts/score_official_argformer_retrieval_biological_v1.py \
  --execute --expected-config-sha256 RETRIEVAL_CONFIG_SHA_FROM_SOURCE_AUDIT \
  --output experiments/results/official_argformer_retrieval_biological_v1
~~~

Failure receipts preserve partial products and traces. The runner rechecks policies, source, completed native bindings and original mean-exp exports afterward. It writes no prepared input, checkpoint or earlier readout. Retry outputs must be fresh. No final holdout is prepared or evaluated.

## Source-only verification and pins

Meaningful handmade tests cover streaming versus exhaustive scores across chunks, exact tie ordering, duplicate embedding rows, source normalization/cosine parity and zero vectors, pure-corpus role eligibility, repeated-center projection, and missing support/namespace rejection. Real embedding distances, model inference, truth and scores have not been executed by this authoring agent. The audit directory retains metadata preflight, handmade receipts, author inventory and supplement-attempt records.

ARGformer author commit is f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596. The immutable source inventory uses compact sorted file-digest JSON plus a newline in this readout. Pinned evidence and original files are listed below.

| File | SHA256 |
|---|---|
| [.runtime/author_sources/argformer_f7545be7bca0/README.md:1](/workspace-SR003.nfs2/ARG-GNN/.runtime/author_sources/argformer_f7545be7bca0/README.md:1) | a58d830dbc12925532d30f5cfe7b4ca44ddf3509c38a4d2824b37310613a7e2c |
| [.runtime/author_sources/argformer_f7545be7bca0/retrieve.py:1](/workspace-SR003.nfs2/ARG-GNN/.runtime/author_sources/argformer_f7545be7bca0/retrieve.py:1) | 520f1961c396432a875a40be253a01648e7debf1fad8c9a274a7522d3809e7ea |
| [configs/official_argformer_biological_pilot_v1.json:1](/workspace-SR003.nfs2/ARG-GNN/configs/official_argformer_biological_pilot_v1.json:1) | 84eaf8b6638f01f37b6519193b8e9d85880880123cdede002a17de80129f098e |
| [configs/official_argformer_biological_readout_v1.json:1](/workspace-SR003.nfs2/ARG-GNN/configs/official_argformer_biological_readout_v1.json:1) | 0758e9f2cdcf738d75ddcafdeb03a87d3be86f92836075f30cd2e55937efc96d |
| [data/literature/argformer_btag438_source_check_v1/body.txt:1](/workspace-SR003.nfs2/ARG-GNN/data/literature/argformer_btag438_source_check_v1/body.txt:1) | ca8b72e780bc885f8c53cc9d13bf7d147c0588742be2f1b31cebbf241640251b |
| [data/literature/argformer_btag438_source_check_v1/fulltext.xml:1](/workspace-SR003.nfs2/ARG-GNN/data/literature/argformer_btag438_source_check_v1/fulltext.xml:1) | 53ab26710804f0b9ca6f355095893b365413caf56cf7bb722b39907f6db0619f |
| [data/literature/argformer_btag438_source_check_v1/source_receipt.json:1](/workspace-SR003.nfs2/ARG-GNN/data/literature/argformer_btag438_source_check_v1/source_receipt.json:1) | 7bec3f9dde5848630ee2fddf8a4412daeec5d2ba52c88981e1e9f6d94ed51f87 |
| [scripts/score_official_argformer_biological_v1.py:1](/workspace-SR003.nfs2/ARG-GNN/scripts/score_official_argformer_biological_v1.py:1) | 82e031a9b22bf469ed044000c87a345537c1255ee398705137fa47f6d72b68e4 |
| [scripts/train_ancestralpaths_biological_v1.py:1](/workspace-SR003.nfs2/ARG-GNN/scripts/train_ancestralpaths_biological_v1.py:1) | 826a1e8c3053cf63b8374361f793903970dd75d44cbcd536755e9aaa3b985056 |
| [src/arggnn/official_argformer_bridge.py:1](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/official_argformer_bridge.py:1) | d6d81a078cff1bb132bc5040a62db7e728c439472b8093eba41b24a4c1edab80 |
| [src/arggnn/train_official_argformer_biological.py:1](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/train_official_argformer_biological.py:1) | edc290b9fbce093f4b844f7d07c4963c23bf24257f43c7d502c43a2753170b94 |

Author tree inventory SHA256: 6d569f1499aa4d4218522f5efafa4eb07beb19b319bf2213a67f85ec87fa04de.
