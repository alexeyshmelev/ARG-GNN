# Independent source audit: H028 observed variant residual v1

Frozen source and metadata support the stated development discovery pilot. No blocking implementation defect was found. This is a source/metadata readiness judgment, not a performance result or paper verdict.

## Checked contracts

- Original VCF parsing joins its 140 ordered diploid names and phased allele indices to the original 280-haplotype roster. The runner joins graph/emitter samples by sample ID and haplotype, preserves query order, and verifies native sample dates, coverage, centers, bins and retained indexes (module lines 21–128; runner lines 151–238).
- Independent compact-roster checks passed for all four parents: 120 query, 96 pure-source and 64 unlabeled mixed haplotypes, identical source/inferred/emitter identity-role-date-class metadata, younger counts [24,16,22,22] and older counts [0,8,2,2]. Query/mixed source classes are null. Actual VCF and tree payloads were not inspected.
- Categorical evidence uses (count + 0.5)/(called + K*0.5), retains all declared nucleotide categories and treats missing query calls as unavailable. If a globally present donor class has no call, its site is omitted for every class in that cohort. Stable window softmax includes only available classes. The author's two-site hand example is analytically consistent: likelihood ratios 5 and 5 give fields 25/26 versus 1/26 (module lines 131–240).
- Windows use original parent [center−5000, center+5000) coordinates, coverage clipping and original 1kb inverse-density weights. Features are prepared parent-wide before tile slicing, so windows cross artificial 250kb boundaries (module lines 182–240; runner lines 214–217 and 287–288).
- Older ANA is globally absent, excluded from the cohort denominator and exactly zero before and after centering. It does not restrict final ANA logits. Both fields share younger/older C4 ordering and exact observed-age centering. Graph context remains the containing-tree, class-balanced native affinity (module lines 243–279; graph helper lines 82–99 and 167–197).
- The observation preparation reader rejects truth/oracle paths and preparation writes its feature seal before the TRAIN target-opening function is called. No target object or binding enters prepare_v1. Supervised TRAIN emitter/teacher output is used only to recompute the common frozen output skip, never as residual-head input (runner lines 108–148, 239–259, 312–358).
- All arms allocate the same zero-initialized bias-free 8→4 head. Frozen logits and contexts are detached. Seed 101, FP32, 10 epochs and 16 TRAIN updates per epoch give 160 updates per arm. Strict NLL improvement keeps the earliest tie. Full predictions are sealed before the unchanged external VAL scorer reads C4 targets (module lines 298–313; runner lines 359–433; inherited scorer lines 566–575).
- The original external target manifest and binding metadata identify two TRAIN, two VAL and zero TEST parents. Original census350 JC69 simulation provenance and C4 order are inherited and pinned. Target arrays were not inspected.

## Interpretation and nonblocking precision notes

1. Fixed arithmetic fusion is a declared heuristic in shared class/cohort coordinates. The fields are not jointly calibrated probabilities. Genotype composite evidence over 10kb and graph affinity on containing trees compress observations differently, and graph inference also used mixed64. A positive or negative fusion result would apply to this tied-head fixed-mean representation, not all possible fusions or an isolated topology mechanism.
2. There are 32 allocated parameters per arm. The four weights attached to the identically zero older ANA input receive no data gradient. Cohort centering also constrains the fields. Use “32 parameters” rather than “32 active independent parameters.” Capacity is still matched across arms.
3. The 1GB output guard fails safely on overrun, but the author's 850MB estimate is compression-dependent. Per parent, raw logits plus probabilities alone require about 76.8MB. Up to 20 such products can coexist transiently: baseline2, old current/best up to12, and new epoch seals6 before pruning. This is approximately 1.54GB before compression. Do not describe 850MB as an uncompressed worst-case guarantee.
4. Temporal donor proxies, only 12 older donors, linked and paired observations, exact-age transductive centering, in-sample TRAIN emitter/teacher predictions, one seed, and repeatedly reused two-parent development VAL remain limits. NLL selection does not establish accuracy improvement or independent generalization.

## Freeze and activation

The completed author packet, both new source files, config and protocol match their reported hashes. The final audit is bound to the hashes in receipt.json. The first source inspection occurred while the packet was being finished and was not separately hash-sealed; no claim is made about an earlier construction snapshot. The final packet added its receipt and protocol without changing the final reviewed source/config hashes.

Root still owns the active input, inherited source closure, runtime, storage reservation and launcher. The pre-preparation inputs_v1 reader intentionally consumes all source_sha256 entries, so that root-controlled closure must contain sources/config/receipts and no truth/oracle payload. Existing target metadata may be bound without opening target arrays. No active root input or live execution was audited here.

All project work and the two audit artifacts stayed on anogena. No source edit, installation, Git operation, model launch, GPU use, original VCF/tree payload, prediction/weight payload or query truth access occurred.
