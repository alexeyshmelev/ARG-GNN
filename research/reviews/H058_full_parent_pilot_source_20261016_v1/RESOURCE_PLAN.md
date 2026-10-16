# H058 source resource counts

The count generator reads only two authenticated existing JSON metadata files and immutable source shapes. It opens no array/archive/tree/checkpoint/truth. Per-parent and aggregate counts are saved in source_resource_counts.v1.json, including all32 TRAIN and eight VAL parents.

| Count | Source value |
|---|---:|
| TRAIN trees |852..1,005 per parent|
| TRAIN native vertices |1,621,356..1,912,515 per parent|
| TRAIN original SNPs |12,807..13,801 per parent|
| TRAIN original CPU bank tensor upper bound |9,369,185,304 bytes aggregate|
| VAL original CPU bank tensor upper bound |2,383,918,196 bytes aggregate|
| TRAIN native FP32 target mass |335,108,256 bytes aggregate|
| Original H045 episode tensor upper bound |1,045,930,998 bytes over TRAIN|
| Largest original bank tensor upper bound |312,908,420 bytes|
| Largest one-arm packed CPU layout |77,011,560 bytes,15,391 Tensor objects|
| Largest packed named preparation allowance |187,454,304 bytes|
| F1 initial binding comparison scratch allowance |25,165,824 bytes|
| New96 native FP32 prediction products |1,023,007,104 bytes|
| Optional48 frozen-control products |511,503,552 bytes|

All952 phased haplotypes/686 queries/266 references and complete2Mb parents remain. Bank tensor dtype/shape rules give 100N-24T+8TH+164S+4T+8R+8Q+12H bytes plus SH times the original genotype element size. The accepted genotype range is1..8bytes; its actual dtype was not read. Bank upper bounds use8, four allele slots and actual metadata readout counts. This is a scientific-tensor count, not total RAM RSS.

Exact packed numeric storage is4*(7N-2T+2TH+4TQ+3TRref)bytes, with15T+1 Tensor objects. Only one parent/one arm route is live. No full-parent packed GPU .to exists. Each transient view transfers eleven int64 vectors plus its small group matrix, created inside original checkpoints. Inner marker blocks make additional transient views; they are counted rather than claiming eleven transfers total for all work.

The source prospectively rejects a hold over the reviewed4,096-node resource ceiling without dropping any vertex. The retained data are never reduced to fit. A V4096/B16/8groups/4alleles FP32 message occupies8,388,608bytes. A transparent32-message engineering allowance is268,435,456bytes; two transient layout/group views add394,624bytes. Actual full-parent hold maxima are recorded from the genuine packed result. Metadata average N/T is not silently promoted into a maximum-node proof.

The route allowance does not bound the entire model. Source counts include every full N-by32 state, both T-byQ-by32 residual list/concatenation copies, GPU query scatter, original full genomic GRU input/output and classifier concatenation. Additional nonreentrant checkpoints wrap original prepare/sweeps/leaf context; original site, native-hold and marker-block checkpoints remain nested. Counts expose10 full N32 named-state copies and a separate32N32+20TH32 planning allowance for staged autograd. These declared allowances are not a proven Torch2.6 saved-tensor, cuDNN workspace or allocator-fragmentation bound. No detached history or reduced corpus is used to make them small.

The original two CPU spawn workers use prefetch1 and no pinning. Each queued PyG parent temporarily includes up to31,223,232bytes of redundant edge_index before item release. CPU bank retention, original episode cache, packed preparation/Python scratch, decompression, workers and scoring all count toward the proposed original64GiB combined owned-RSS ceiling. The96/144 prediction matrix counts exclude evaluator target/intersection temporaries. Root's current actual process/RSS checks remain necessary.

The historical H050 first scheduled TRAIN004 continuation update measured14,621,469,184bytes allocated and20,451,426,304bytes reserved in a shared four-arm allocator. That is not an H058 measurement or all-parent bound.16GiB could fit after native packed processing and staged checkpointing, but source alone cannot certify the new backward/workspace/reserved peak. A reserved cap of16GiB could fail even when allocated tensors fit. The conservative reference retains H050's original28GiB reserved ceiling. ROOT currently plans a16GiB enforced ceiling to coexist with its declared other allocations; the inactive template exposes16GiB as a configurable operational cap. This does not establish fit. ROOT supplies actual capacity/lease/guard enforcement, and a capacity failure must remain a real partial failure.

The first real scheduled TRAIN004 step records its complete retained update's timing, memory, gradients and actual optimizer counter. It is update1 of960 in each new arm. Root can use those real facts when choosing later resource caps; no smaller parent/SNP/query set or discarded optimizer update is authorized. A later larger parent can still exceed the first-parent observation, so every scheduled update remains guarded.

There is no measured fit, throughput estimate, lease availability claim or numerical qualification in this source count package.
