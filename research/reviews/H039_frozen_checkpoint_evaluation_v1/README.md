# Root commands

After the actual all-three checkpoint group receipts for passes 5 and 10 exist:

```sh
cd /workspace-SR003.nfs2/ARG-GNN
.runtime/env/bin/python research/reviews/H039_frozen_checkpoint_evaluation_v1/root_evaluation_driver.py --admit
.runtime/env/bin/python research/reviews/H039_frozen_checkpoint_evaluation_v1/root_evaluation_driver.py --launch --gpu 3
```

For an optional pass-5-only evaluation after its all-three frozen checkpoint receipt exists:

```sh
.runtime/env/bin/python research/reviews/H039_frozen_checkpoint_evaluation_v1/root_evaluation_driver.py --admit --passes 5
.runtime/env/bin/python research/reviews/H039_frozen_checkpoint_evaluation_v1/root_evaluation_driver.py --launch --passes 5 --gpu 3
```

Root selects GPU 2 or 3 under the existing v2 authorized-sharing launcher. Admission never inspects GPU inventory and never launches. The worker requires the genuine held lease, exact process title, original qualification and the 50 GB CUDA cap. The configured narrow marker helper is used for every storage guard and write.

The two capabilities have separate job/output/admission folders (`admitted_pass5` and `admitted_pass5_10`). Admission reads checkpoint group JSON metadata and checks checkpoint file existence; it does not load checkpoint, observed or truth arrays. The actual worker verifies every checkpoint byte hash and its model/pass/training provenance before reading VAL observations. Exactly eight VAL truth bindings are authorized; TRAIN and outer array reads are denied.

Actual launcher output is saved in the chosen admission folder's `launch.log`. The fixed `root_wait.json` records the actual direct launcher wait and preserved errors. Failed or interrupted outputs stay charged; no automatic reserve or execution replay occurs.

Worker: src/arggnn/evaluate_directed_clade_exchange_train_stage_v1.py.
Prospective rationale: prospect.md.
Author checks: source_receipt.json.
The source-only author did not import/run the worker or driver, reserve storage, inspect live resources, read observation/checkpoint/truth arrays, launch GPUs or commit.
