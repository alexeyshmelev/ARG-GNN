# Independent review of submitted v2 amendment

Verdict: **BLOCKED, source-contract only**. The amendment resolves the BH full-pool target-statistics omission and the GST per-model paper/shared source learning-rate discrepancy. All 217 initial source-contract inputs remain unchanged.

The AncestralPaths/GNOMix paragraph and machine fields misstate the primary source. Supplement Note 3, printed p34, describes four query populations: GBR, Bronze Age, Neo and Yam. It does not describe four modern individuals. BAA is excluded from queries. GNOMix uses logistic regression as its base and xgboost as its smoother, rather than alternative model variants. The 0.02 cM window belongs to GBR queries and 0.2 cM to Bronze Age/Neo/Yamnaya queries. Bind the published five 200 Mb sequences, evenly spaced sites and EHG/WHG/CHG/Ana references while leaving unavailable raw identities, sample IDs, exact versions, seeds and splits unknown.

The machine amendment also points to a nonexistent initial `review_receipt.json`; the actual initial artifact is `receipt.json`. Effective v1+v2 retains the nonblocking 640 “distinct” validation-set wording, although only generator draws were verified.

Submitted v2 SHA256s are preserved in `receipt.json`, which also holds the complete primary GNOMix excerpt, exact input hashes and check results. This review read source/paper/metadata only and used PDF text extraction to stdout. It acquired no dataset, opened no biological/model/score array, executed no author code and changed no source, runtime, ledger or Git state.
