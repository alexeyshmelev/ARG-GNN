# Independent AP biological v2 sampler audit

No blocking findings in the reviewed v2 bytes. This is a source-only audit:
no truth arrays or VAL scores were opened, and no competitor code, model fit,
GPU job, installation or Git operation was executed.

TRAIN-only v1 evidence confirms Neo accepted 23,731/25,000 rows, including
11,231/12,500 WHG rows. Its pass8 stride 19,683/256 admits only site0 among
10,147 SNP rows and adds zero. V1 has no completed seed or optimizer fit.

The v2 source pass loop and initial stride match v1 AST after excluding only
added stop-reason annotations. Candidate order, caps, repeated-pass replacement
and source safeguards remain unchanged. Each cohort unconditionally contributes
its actual accepted rows; stalled quotas disclose shortfalls and later cohorts,
including Yam, continue. There are no fabricated/padded rows or increased caps.

Hash-bound v1 helpers retain the native Dense model, truth capability separation,
prediction sealing and scoring. Model, fitting settings, seeds, physical support,
C4 collapse and metric config remain identical. Explicit guards now enforce
metric equality, approved shortfall-policy identity and cohort insertion order.
The runner binds all four existing observed feature caches and metadata to the
preserved failed receipt, completed inference manifest and actual node/query/
physical grid/site identities. No feature extraction reruns; v1 output writes
are rejected. Cache and metadata hashes were independently rechecked.

Root's check-only admission and genuine fit remain pending. Actual training totals,
later cohort shortfalls, fitted weights, optimizer receipts and VAL results must
come from those executions. The reduced realized count is a disclosed method
adaptation; no exact paper/CNN or final holdout claim is justified by this audit.

See review_receipt.json for the exact reviewed runner/config/doc hashes,
v1_TRAIN_quota_diagnosis.json for preserved evidence and static_candidate_review.json
for the candidate loop comparison.
