# Independent source review: native full-state resume v5

Verdict: **REQUEST_CHANGES_SOURCE_ONLY**,one source-contract P1. No execution authorized.

Candidate handoff SHA256:da864b6600c3826f8ad66ee3c7589c1b2cb961289bb7db3a58cb5046ddc2c296.
Candidate manifest SHA256:1b45642f350ee58ae55d3c1861a28c521d1cb02851ece926a24b08469d66dd14.

## F1: Scope ROOT's required drain qualification

amendment_contract_v1.py:272 requires all_prior_owners_drained_and_future_writes_revoked=True. Yet:157–163 explicitly preserves v3 complete_owned_roster_known=False and owned_roster_truncated=True. The qualifier schema/template and recipe do not explicitly restrict that all-prior assertion to genuine current-v4 drain,fresh exact known-prior PID/group absence and cooperative old-route revocation while older unknown descendants remain unknown.

ROOT cannot truthfully establish complete historical ownership/drain from this evidence. Leaving the broad flag false fails qualification; setting it true risks an unsupported historical all-drained assertion. Publish a fresh sealed contract/template/recipe with separate bounded claims,evidence refs and explicit retained older uncertainty. Add fake cases accepting truthful scoped facts and rejecting broadened history. This is the source blocker;missing future runtime receipts are not source failures.

## Checks passed

All165 distinct packet/manifest refs,including148 inherited source/compact metadata refs,match and Python sources parse. Candidate hashes were rechecked before seal. The ROOT6500 compact proof is bound exactly;no checkpoint payload was read.

Original7GiB whole,6576734208 scientific output,805306368 AUX and6442516480 checkpoint-section ceilings remain. The original native run applies soft/hard RLIMIT_FSIZE2GiB before serialization. Old checkpoint stays immutable;at most2 new files yield3 global regular checkpoints. Fake census cases accept the full prospective6313174016-byte checkpoint peak and reject a third new file,oversize,new child escape,old retained-byte drift and AUX overflow. Actual retained-scope fit still belongs to ROOT.

The actual installed Composer load_checkpoint and _restore_checkpoint AST functions were exercised with fake State/runtime/payload via the candidate's private FD download callback. Complete serialized state reaches the original restoration branch;77 fake Adam states at6500,timestamp,scheduler and RNG are checked before any training. Seven missing/wrong-state cases reject. Installed State.load_state_dict and original Trainer RNG/epoch spin/batch skip paths were inspected. The full restore code object and native run code object are retained;the downloader substitution is I/O-only. This is software evidence,not actual numerical/runtime restoration.

The four operational config fields are disclosed. Original5ep/227990 objective,batch/micro512,amp_bf16,seed17,workers8,compile and500ba save events remain. Nested Timestamp lookup adaptation preserves original full final epoch/update/model/optimizer/alias checks.6500 remains partial.

Original32+2GiB native,C326GiB and4GiB headroom capacity contract remains. Fake16GiB external boundary and above-threshold failure prove bounded actual-or-unknown evidence emitted synchronously before cleanup;failed filesystem persistence remains explicit. Entire4096-identity30s cleanup finally is AST-identical to v4;10 controller/status/attachment bodies are unchanged. Original48h anchor,absolute expiry,strict fresh-entry cutoff and cleanup/evidence margins remain.

## Future actual work

After F1 is corrected and source review passes,ROOT still owns genuine v4 stdout/outer evidence binding,narrowly scoped quiescence/revocation,whole-parent retained capacity and fresh actual GPU0 fit,then explicit one-attempt authority. A new factual metadata projection may bridge receipt field names only while preserving worker-15/raw15/outer0 failure and original receipt provenance.6500 is not full native fit. Actual restore,new progress,full227990 completion and genuine new wait/drain/outer remain unestablished here.

Candidate unchanged. Only source/compact metadata and fake fixtures in the owned review directory were used. No payload,scientific import,process/GPU/census/accounting/authority probe,Git,package or launch action occurred. Structured refs are in review.v5.json.
