# Fresh-development candidate fitting

The dedicated runner is train_anchor_development.py. It copies the verified candidate training mechanics into an independent module and does not modify active pilot dependencies. Production admission requires exactly64TRAIN/16VAL/TEST0, fixed training seeds101/202/303, and development-v1 namespace. There are exactly eight candidate configurations: plain/age geometry, kernel depth1/2 and learning rate0.0003/0.001. Other shared dimensions and budgets remain source width32/depth2, latent8,32anchors, three branches,48epochs/patience8 and zero rate/mixture weight decay.

A concrete execution config must bind completed source, boundary and age manifests/preparation hashes. The CLI additionally requires the exact config SHA256 and rejects an unlisted seed. One seed/config subset may be scheduled per invocation, but missing configurations/seeds make the search incomplete. Root freezes the concrete config after preparation and owns scheduling.

Validation scatters each tile back to the exact parent positions/query identities. Selection uses equally weighted parent NLL from float64 stable log-softmax of uncalibrated raw logits. All saved validation predictions are raw float64 logits, with parent IDs, positions, queries and protocol SHA. Historical capped NLL is separately available. Checkpoint ties retain the earliest epoch. Family configuration selection across all three seeds remains a later frozen search-synthesis step.

The run stores a compact manifest binding and ordered parent IDs rather than duplicating the large geometry manifest. The original dataset manifest remains content-bound in the dedicated data directory. Source dependencies are hashed before and after fitting. Synthetic tests exercise extreme NLL, coordinate scattering, production freeze guards and a real tiny CPU fit/checkpoint/prediction round trip. No fresh or final cohort was trained for these tests.

## Independent audit corrections, 2026-10-10

Root added safe model-ID components and immutable configuration snapshots before any new runner GPU use. Every produced artifact retains the exact startup config digest. A configuration change during loading/fitting prevents completed results from being sealed. Synthetic adversarial tests cover absolute/traversal/control-character IDs and real config edits during dataset loading or validation. The focused combined runner suite passed36tests in8.31s. The independent auditor replayed both drift reproducers and verified rejection without completed results. This supersedes earlier authoring runner/test hashes above.

src/arggnn/train_anchor_development.py SHA256 89bb74a03ad8736594bcf662ccc5e96a5f40f9867315b58fa4b80c3f5df73f7f

tests/test_train_anchor_development.py SHA256 95567d2b82e2937db641d64b31abf4956985e0d7fdae3bbd5ad8cf093270b31a

## Current fresh execution extension

The current runner also binds generic-selector family/seed/completion metadata, retains failure receipts and applies strict deterministic development settings. Selected and reloaded NLL values are recorded separately and must agree. See docs/development_execution.md and its source receipt for current hashes. Earlier source hashes remain historical verification snapshots.
