# GST repository deep audit

This audit inspected current main and dev, every one of the 18 reachable public commits, and their source/configuration history. Current main has 166 blobs. The previously staged runtime had 16 selected files. The full history index contains 764 text-file entries and 194 unique text blobs, including 52 Python variants. All selected historical text fetches were verified against Git blob IDs. This was static inspection, with no author-code execution or training.

## Recovered material

- Deleted `gst_v2.py` is public at commit `d8fd56a797998527b32e85b6b7469bf307161d60`, SHA256 `62b5ea04ef1ec0903fdf2ac65b3220e10ca0ae287559f6ec5a0cbb049cd0773c`. It has a node FFN, LayerScale and separate graph/set gates. Its presence resolves a current-head missing-module claim, not the choice of exact publication architecture.
- Historical `train_gnn_benchmark.py` and `train_gnn_benchmark_fixed.py` load official PyG CIFAR10/MNIST splits and build graph sets. The fixed version uses size 10 and five seeds. The variable version trains on sizes 5/10/20 and evaluates single graphs. The paper describes sizes 5/10 and three seeds. No exact saved CIFAR table artifact was found among the indexed histories.
- The full current tree includes saved configurations and raw result summaries that were absent from the selected runtime. These are useful source evidence, not new experimental results.

## Actual executable paths and remaining differences

`eval_synthetic.py` and `eval_rxn_center.py` import the tracked current `gst.py`. The former generates graph sets directly. The latter names the public USPTO15K source and fully defines split parsing, atom features, edit-derived labels, training, validation selection and scoring.

Current `eval_react.py:108` and `eval_synthetic_v2.py:139` import an absent current `gst_v2.py`. The module is recoverable historically, but the current scripts did not coexist with that historical file. Combining them is an adaptation requiring explicit qualification. It is not a recovered unchanged author experiment.

The current module uses one set gate. The historical v2 uses two gates and extra local/normalization/scaling operations. The latter must not be called exact paper code merely because the paper also describes two gates. No model was constructed or trained in this audit.

The paper specifies learning rates 0.0001 for GST and 0.001 for its baselines. The native synthetic/USPTO commands expose one all-model learning rate. The BH runner uses Adam whereas paper prose specifies AdamW for non-CIFAR tasks. These need named conditions instead of silent edits.

## Saved results are not the printed table

The committed USPTO configuration uses the first 1,000 retained training reactions, five seeds, one layer and LR 0.001. Its committed TEST AP is 0.6089/0.6195/0.6125/0.6158 and its uncertainty is a Student-t CI. Paper Table 3 first-block AP is 0.839/0.858/0.822/0.819 with mean/std and different GST parameter counts. This is direct evidence that the available saved run is a separate pilot, not a reproduction of that table.

## Practical conclusion

The repository contains actionable native synthetic and USPTO pipelines and recoverable historical CIFAR and v2 source. The candidate comparison remains unfinished project integration. Other paper/source settings need resolution or an explicitly declared reconstruction. Original-table reproduction is not established, but the absence of a finished project benchmark cannot be attributed simply to missing author code.

Fresh metadata, source census, all 18 commit identities, response hashes and source indexes are preserved beside this report. No tags, releases or issues were returned by the fresh public metadata endpoints. Public code history has no declared license file in inspected trees. This is a bounded public-source audit, not a claim about unpublished/private assets.
