# TGB annual trade diagnostic source readiness

A causal annual-trade diagnostic is feasible as a separate adaptation. The task, source version, URL, compressed size, reported licence, metric and split formula are pinned. Execution readiness remains conditional on an authorized archive audit, especially edge/label year alignment. No dataset was downloaded.

## Official release and task

| Item | Verified evidence |
|---|---|
| Repository | [TGB commit 3b83e5cfb2010743fd4f7a80a01b0e19dfcf78cb](https://github.com/shenyangHuang/TGB/tree/3b83e5cfb2010743fd4f7a80a01b0e19dfcf78cb), committed 2026-07-27 |
| Versions | `py-tgb` 2.3.0; `tgbn-trade` dataset version **1**, as declared in pinned source |
| Task | Annual directed agriculture trade among nations; predict queried nations' next-year destination trade proportions, with 255 destination dimensions |
| Download | [Official archive](https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbn-trade.zip), listed in `DATA_URL_DICT` |
| Size | HEAD 200: **12,780,237 bytes**, ETag `623fc179f3f5cc2df64b9f07e887befd`, Last-Modified 2026-07-12. Zero body bytes read. ETag is not authenticated SHA256; uncompressed size is unknown. |
| Licence | Dataset **MIT**, explicitly reported in [TGB Appendix B](https://arxiv.org/html/2307.01026v2). Code is separately MIT; its exact `LICENSE` is retained. Archive licence contents were not inspected. |
| Raw filenames | `tgbn-trade_edgelist.csv` and `tgbn-trade_node_labels.csv`, expected by the loader |
| Published counts | Table 1: 255 nodes, 468,245 edges, 32 timestamps. The description says 1986–2016, containing 31 calendar years. Actual range/counts require checking. |

The official constructor downloads by default. Do not import/run native examples as a metadata check.

## Splits metric and tied timestamps

The core loader uses `q70,q85=np.quantile(edge_timestamps,[.7,.85])`, then train `t<=q70`, validation `q70<t<=q85`, test `t>q85`. Ratios are approximate because complete tied years stay together. Exact calendar cutoffs require the raw timestamp histogram.

The evaluator calls `sklearn.metrics.ndcg_score(y_true,y_pred,k=10)` over all 255 destination scores. Default predicted-score ties are averaged, verified in the declared minimum scikit-learn 1.2.2 source. Native examples average countries within a label year, then years equally. Pin the actual execution dependency version.

Yearly rows have no meaningful internal order. Forecast year τ from completely released years **<τ**. Seal all 255-by-255 predictions before target source selection/label join. Consume every edge at τ only after that forecast. Target-year row permutations must not change predictions. Released validation/test observations may update streaming memory, while their targets must not update fitted parameters.

## Native baselines and timing risk

Unmodified pinned author sources are retained.

| Baseline | Exact behaviour and entry |
|---|---|
| `PersistantForecaster` | `modules/heuristics.py`: last released vector, zero cold start. Entry `examples/nodeproppred/tgbn-trade/persistant_forecast.py`. Misspelling is upstream. |
| `MovingAverage` | Same module, example `window=7`: **EMA** `(6*old+new)/7`, first observation copied. It is not a rolling seven-item mean. |
| TGN and DyRep | Official trade entry scripts retained. TGN uses memory/embedding/time width 100, neighbour size 10, batch 200, Adam LR 0.0001 and 50 epochs. No model was constructed. |

The label API accepts `cur_t>=pending_label_time`, but examples trigger at `query_t>label_t` and consume one pending label per call. TGN advances `label_t` before filtering a crossing batch; earlier batches may already contain edges with the target year. This is a **source-backed causality risk conditional on actual CSV alignment**, not a demonstrated archive leak.

The archival `generate_aggregate_labels` function copies years greater than 1986 without shifting timestamps. Its file has a comment-only `main` and fails Python parsing. It is not the runtime loader, should not regenerate this release, and does not establish how the hosted archive was produced.

Use one disclosed whole-year prediction/release scheduler for all future methods, preserving native algorithms and cold-start behaviour. Flush every eligible final target year. Record scored periods per split. If scheduling/support differs from examples, call it a causal diagnostic using TGB v1, not an exact paper reproduction or unqualified leaderboard result. Native heuristic examples use PyG's split helper, while TGN uses dataset masks; freeze and compare their resulting periods explicitly.

## Observation derived partition and witness interface

Retain the public 255-country catalogue as schema with a frozen mapping hash. Include one leaf per country each year, including unseen/inactive countries. Do not use future activity flags, a future-present panel intersection, or learned country-ID features.

At the end of τ−1, form row-normalized outbound trade profiles from released edges only, with separate missing-history and zero-mass flags. Build a **complete single-linkage dendrogram** using total-variation distances. Simultaneous distance ties remain polytomies. Missing-history leaves have distance 1 to others and remain singleton below the proposed fixed cut 0.25. This is a preregistration candidate, not a validated optimum. Naive distance construction has cubic work and quadratic distance storage in panel size.

The persistent witness is a country leaf. Internal merge nodes are anonymous clusters. Merge height means dissimilarity, duration means calendar years. There are no reference populations, mutation support, generations, bp widths or 350-generation cut. Forecasting requires forward-only history. The current 952-leaf H050 bundle is not reusable unchanged.

[causal_partition_contract_v1.py](/workspace-SR003.nfs2/ARG-GNN/research/reviews/TGB_trade_crossdomain_readiness_20261016_v1/causal_partition_contract_v1.py) implements an inert metadata validator and schedule helpers. It checks all 255 leaves, complete connected topology, exact cut membership, missing-leaf singleton policy and strict causal fields. It implements neither a model nor a dendrogram builder. Twenty-one synthetic cases and the inert default CLI pass. Metadata assertions do not prove upstream feature provenance.

## Remaining readiness conditions

A separately authorized archive audit must pin SHA256, safe ZIP manifest/uncompressed size, headers, duplicates, country mapping, year histogram, normalization and edge/label alignment. Resolve 32 timestamps versus 31 described years, establish exactly which calendar target each label represents, and freeze cutoffs and scored years.

Then review the causal builder/scheduler and baseline adaptations on identical information/support, pin dependencies, and register the narrow diagnostic. Controls should include persistence, EMA, causal whole-tree/GRU and triple/marginal/leaf memory. One network over a few decades offers limited independent evidence.

[data_contract.proposed.inactive.json](/workspace-SR003.nfs2/ARG-GNN/research/reviews/TGB_trade_crossdomain_readiness_20261016_v1/data_contract.proposed.inactive.json) separates verified facts and unresolved fields. Source locators/hashes and retrieval receipts are adjacent. No utility/novelty claim, dataset download, installation, model/job/GPU action, guard/accounting call or Git operation was made.
