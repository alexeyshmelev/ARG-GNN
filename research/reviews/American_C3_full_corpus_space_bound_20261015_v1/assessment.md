# Full original C3 native corpus: compatible tighter storage bound

A substantially tighter source-level upper bound is available without removing or changing any native output. With the exact count metadata and the explicit invariants below, all existing native preparation outputs plus root log/metadata fit a conservative 5,942,029,370-byte aggregate bound. A proposed fresh 6,000,000,000-byte output+auxiliary reservation leaves 57,970,630 bytes above that bound. This is arithmetic and a source proposal, not measured output size, an admission or an implementation. The frozen v2 package cannot be launched under the smaller reservation unchanged.

The existing 12,246,172,077-byte forecast already uses actual T=316,672,003. It does not use 29*P as its token count. Its largest loose term is 256*P=6,547,094,272 bytes. The bound can be tightened by accounting for actual serialized containers, exact reference-view counts and tree-index integer opcodes. No history or failed charge is deleted or credited by this proposal.

## Evidence and exact counts

The original completed full-count receipt is research/reviews/American_C3_native_paths_capacity_source_v2/actual_counts_v1/completed.json, SHA d743c880b5c7773221aaa87da23a26307b0b7fbed6ce4b03217be3c256d2c86a. It binds the full original tree, observation, count-worker, native scientific source and original author functions. Its referenced per_sample_counts.json SHA74505550902e3bc0c990f956868bfe8149bfa68173b05e1217cab2b9d099254f contains counts only. All1600 sample indices and sample-node IDs are0..1599 in order. Summed totals match the completed receipt.

| Quantity | Exact count or fixed bound |
| --- | ---: |
| Unique paths P | 25574587 |
| Unique full-path tokens T | 316672003 |
| Sample/tree occurrences O | 122980800 |
| Vocabulary nodes V | 106769 |
| Maximum raw path length L | 29 |
| Fixed query focals F | 10000 |
| First600 samples' unique paths P_R | 9583977 |
| First600 samples' tokens T_R | 116080787 |
| Last1000 samples' unique paths | 15990610 |
| Last1000 samples' tokens | 200591216 |

The simulation source emits ordered AFR/EUR/ASIA reference samples followed by ADMIX query samples; inference preserves the original sample/roster order. The observed manifest binds counts200/200/200/1000 and reports identity verification. A successor must explicitly require, from the same existing input roster it already reads, that sample indices0..599 and nodes0..599 are pure_source_reference and600..1599 are query. This audit did not open the roster or any labels. This role-partition check is a prerequisite for using P_R/T_R, not an inferred query ancestry label. If it fails, stop without a fallback, reassignment or smaller corpus.

The author get_path_key begins with the sample node, so paths of different samples cannot collide. The count worker proves and checks that condition. Thus summing the first600 sample counts gives exact reference-view path/token totals before seed42 splitting. The train/val split partitions them; it does not multiply these totals. The query view contains exactly the same10000 preselected query focals and at most29*10000 tokens. Its actual token count is not claimed here.

## Bound derivation

Keep all six raw_sequences/vocab native pickle files: MLM, reference-only contrastive and query-focal views, each with its original train/val split. Also keep the seven vocabulary copies, two label pickles, six row-order NPZ sidecars, focal NPZ, full sequence-to-tree mapping and producer completion. The aggregate row count across the six raw files is N=P+P_R+F=35,168,564. The aggregate token count across those files and the mapping is at most 2*T+T_R+L*F=749,714,793.

Use CPython3.11 protocol5, exactly the current pickle.HIGHEST_PROTOCOL serialization, with ordinary builtin int/list/tuple/set/dict values and the existing dictionary/list insertion orders. Every raw node is nonnegative and below2**31, so its integer opcode uses at most5 bytes. The original vocabulary keys node_N have at most11 ASCII characters for this tree; 64*V per vocabulary copy remains a generous bound, with fixed headers/framing in the existing margin.

For each raw native dataset with n rows and t tokens, 5*t+8*n+64*V+16MiB is sufficient. A path list of length at most29 uses at most4 non-token bytes (EMPTY_LIST, MEMOIZE, MARK/APPENDS); the outer list adds at most2 bytes per1000 rows plus a constant. The8*n allowance dominates that structure, and16MiB covers framing, outer dictionary/special-token headers and fixed vocabulary overhead. This replaces the existing32*n planned-write coefficient without changing a serialized object.

For the full mapping, a key tuple adds at most3 bytes beyond its integer tokens. Each value set adds EMPTY_SET+MEMOIZE and MARK/ADDITEMS batch pairs. For P nonempty sets with O total entries, even allowing an extra empty final batch, the total tuple/set/dictionary overhead is bounded by 7*P+2*floor(O/1000)+2*ceil(P/1000)+a small constant. It is below8*P for these exact counts. The existing16MiB mapping margin covers framing and fixed headers. No Python in-memory dictionary/set allocation is charged as serialized bytes.

Each sample visits each marginal tree once. Since a key includes its sample node and every visit has a distinct tree index for that sample, the full mapping contains exactly1600 copies of each integer index0..76862 across its sets. The order of set iteration does not change integer opcode sizes. Their combined exact integer-opcode bound is

    K =1600*(256*2 +(65536-256)*3 +(76863-65536)*5)
      =404779200 bytes.

This replaces the loose5*O=614904000 term. A per-file mapping plan of5*T+8*P+K+16MiB preserves the same ordinary tuple-to-set native pickle. There is no compressed or packed replacement.

The six sidecars contain three int64 vectors per row plus one extra reference-class vector for each contrastive row, bounded by24*N+8*P_R bytes before archive/header allowances. Keep the original conservative5*P_R label allowance rather than tightening it to2 bytes per0/1/2 class. The original16*16MiB fixed margin is retained in full. Ten pickle files consume ten of those16MiB framing/header allowances; the remaining six cover the focal arrays, NPZ/NPY/ZIP headers and bounded deflate expansion, completion/directory overhead and unchanged auxiliary slack. Combined sidecar/focal raw arrays have less than922MB of vector payload, and a conservative1%+1MiB-per-NPZ expansion/header allowance plus completion/focal/directory metadata is below the remaining100,663,296 bytes. No filesystem payload is read or counted by this argument.

| Aggregate term | Upper-bound bytes |
| --- | ---: |
| Raw path integer opcodes, all datasets and mapping | 3748573965 |
| Dataset and mapping container allowance | 485945208 |
| Mapping tree-index integer opcodes K | 404779200 |
| Row-order int64 payloads | 920717352 |
| Reference labels, original5-byte allowance | 47919885 |
| Seven vocabulary-copy allowances | 47832512 |
| Original16*16MiB fixed margin | 268435456 |
| Original bounded root log | 16777216 |
| Original root metadata allowance | 1048576 |
| **Aggregate bound** | **5942029370** |

The serializer argument is source-based. The installed stdlib pickle.py SHA227b9550b1491f5ed3bf7b444a97f91113d47a3bb0e82fb49da24cbc255d407f was read as text, without import. The matching public CPython3.11.17 Modules/_pickle.c source was also read: SHA b8f6875c572368b62fcd2edeb7c094a77acd2aa58943d96e26b29485800d958f. It confirms1000-item batches,64KiB framing,9-byte frame headers, integer opcodes, protocol4+ set serialization and MEMOIZE behavior. The admitted existing CPython runtime must remain exact; this is not a serializer-independent bound for arbitrary protocols or replacement classes.

## Smallest compatible source proposal

Create a fresh successor, leaving frozen v1/v2 science, adapters and artifacts untouched. Retain the original author get_path_key, build_vocab_from_sequences and split_unique_sequences definitions, full iteration/deduplication and first-occurrence path order, all raw path lists, mapping keys and sets, vocabulary construction and token IDs, seed42 shuffle and int(.9*P) cut, source/reference labels, focal RNG seed92001 and the same10000 focal IDs/coordinates. Keep all existing native filenames, Python object schemas, pickle protocol and NPZ fields. No output file needs to be removed, redirected to a virtual reader or reconstructed by a model loader.

The minimal conceptual changes are only source/config budget plumbing and explicit checks:

1. Bind the original completed count plus its per-sample count ref, exact tree/observation/author/runtime sources, the full actual P/T/O/V/L values, P_R/T_R and the reference/query index partition. Check the same unchanged full counts at terminal completion. The original exact-count result remains historical evidence from the unchanged method, not a result generated by a new serializer.
2. Replace storage_bound and the two loose native pickle planned-write expressions with the derived bound: per raw dataset5*t+8*n+64*V+16MiB; mapping5*T+8*P+K+16MiB. Keep the label/vocabulary expressions unchanged. Retain original Writer.prepare/finish and original resource guards. The ordinary NPZ serialization remains np.savez_compressed; its prospective per-file budget should use a conservative deflate/header bound rather than rely on compression reducing the payload. This can be a local budget-only helper using the same prepare/write/finish pattern; it must not change common/guard sources or the saved arrays. Its extra allowance is already inside the retained fixed margin. Do not simply lower CAP while leaving the original12.228GB storage_bound precondition or per-file32/64 overhead checks in place.
3. Make a fresh inactive contract/consumer/config bind the new producer and6,000,000,000-byte output+auxiliary cap. Preserve existing policy, marker-repair authority, original mutex/accounting/open-scope checks, actual inference-close prerequisite, supervision/time/RSS/log limits, immutable reservation-prefix validation and failed-artifact retention. Root owns subsequent registration, source qualification, admission and execution. No monkeypatch to science or accounting is proposed.

The current integration requires all three views and two splits, and downstream native children use the raw files directly. Deleting the mapping/vocabulary would remove original author outputs. Omitting the contrastive/query raw files would require an explicit changed loader/view interface and contracts. Neither is needed for this6GB proposal. Existing in-memory keys/seqs/sets are not redundant durable files and deleting them would not reduce retained-storage accounting.

Using only the parent-reported projection943,821,569,048, the950GB ceiling has6,178,430,952 bytes of bookkeeping headroom. A6GB reservation would leave178,430,952 bytes before its new event metadata charge or any other changes. This is not a live eligibility check. Root must run its unchanged current guard before any admission. The original failed11.5GB charge and all other historical charges remain fully charged. This bound covers complete native preparation outputs and their declared auxiliary/log/metadata scope; it does not reserve later tokenized caches, fitting/checkpoints or embeddings.

No implementation, fixtures, scientific or guard imports, payload/roster/label/tree reads, live process/GPU/state/journal checks, inventories, admissions, execution, package actions or Git operations occurred. Only source bytes and named compact count/observation metadata were read, and this fresh assessment folder was written. Physical GPU4/5 steering is unchanged and this CPU source task uses no GPU.
