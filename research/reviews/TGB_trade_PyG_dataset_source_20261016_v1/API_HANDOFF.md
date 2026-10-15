# Reusable causal trade-tree Dataset and encoder source

This inactive source packet supports a future ordered diagnostic. The author
reads no real payload, starts no worker/model/fit/GPU job, and grants no execution
authority. All future training is restricted to GPU0. Existing sources and cache
handoff remain frozen.

## Audited chronology and catalogue

Root's payload audit establishes 255 countries, edge years 1986–2016 and target
years 1987–2016. Same-year label profiles equal same-year edge profiles. Forecast
tau therefore uses complete edge years strictly below tau. The frozen H052 split
is TRAIN 1987–2009, VAL 2010–2013, TEST 2014–2016.

CountryCatalogueV1.from_official_mapping(mapping) requires the exact official
first-appearance country -> ID dictionary, in insertion order, and its audited
compact UTF-8 JSON hash:
18dfd6a8a6a3a739a2f1653b1a5172e4dbf564c1073063608cc4d8373c972273.
The full catalogue is fixed public schema. No per-country future-presence feature,
source-country embedding, internal persistent identity or ARG semantics is added.
This packet contains no real country names, ZIP/CSV/network reader or label API.

## Data API

Modules use package
research.reviews.TGB_trade_PyG_dataset_source_20261016_v1.

    catalogue = trade_tree_core_v1.CountryCatalogueV1.from_official_mapping(mapping)

    def load_observed_year(year):
        # Root supplies this complete observed year only, no target-file access.
        # weights [255,255], source_present [255] bool.
        return trade_tree_core_v1.observed_year_v1(year, weights, source_present)

    dataset = trade_tree_pyg_v1.CausalTradeTreeDatasetV1(
        catalogue, load_observed_year, forecast_years=tuple(range(1987,2017)))
    loader = trade_tree_pyg_v1.make_trade_tree_loader_v1(
        dataset, batch_size=1, num_workers=0)

observed_year_from_edges_v1(year, rows, catalogue) also accepts in-memory
(source ID,destination ID,weight) rows, rejecting duplicate directed pairs.
Root authenticates the loader's actual edge lineage. This code verifies shapes,
finite nonnegative mass, source presence, requested year and catalogue order.

The constructor reads no observation arrays. get requests only complete years
less than its forecast. Observed arrays are lazily copied/memoized; built graphs
are cached and returned as clones. The profile is each country's latest observed
outgoing source row in the released prefix, matching the accepted persistence
information. Positive rows are normalized; observed zero mass and never-observed
history remain separate flags. Last-observed year and staleness in calendar years
are retained. Missing leaves have age zero plus their separate missing flag.

The public Dataset requires the official catalogue/year domain. A private core
Dataset and explicit synthetic catalogue helper support the root fixture only.
Every graph contains all countries, without target-source selection. The loader
is ordered and never shuffles/drops forecast periods. Positive worker counts are
only a future caller option; none is started here.

## Complete single linkage and ties

All-pair total variation is half L1 over fixed destination columns, with FP64
reductions over stored normalized FP32 profiles. Row-block computation avoids an
[N,N,N] temporary. Never-observed countries have distance 1 to every other
country, including other missing histories, and remain singleton below cut 0.25.
Observed zero mass stays explicit and is not treated as missing.

Kruskal component merges at each exactly equal computed FP64 height are
simultaneous. Each connected set of pre-height components becomes one genuine
polytomy, with no arbitrary binary tie refinement. Every country leaf and every
distinct-height internal merge remains: 256–509 total nodes and N-1 hierarchy
edges. The cut never prunes the encoder hierarchy. Root height is the genuine
maximum merge distance, which can be less than 1; no artificial super-root is
added. The prior proposed validator's blanket root-height-one rule is not reused.

Leaf order is the catalogue order. Internal integers are anonymous local pointers
within one annual tree and convey no cross-year identity. Heights are TV and
durations are calendar years; no ancestry, mutation, generation or bp quantities
are invented.

## PyG Batch and encoder interface

Graph leaf arrays: leaf_profile, leaf_missing, leaf_zero_mass,
leaf_last_observed_year, leaf_staleness_years. Node arrays: true FP64 merge height,
descendant leaf count, child count, leaf flag and up/down dependency levels.
edge_index contains all child -> parent hierarchy edges. leaf_node_index,
root_node_index and cut_node_index are node pointers; country_witness_index
contains fixed catalogue IDs. observation_sha256 hashes the causal snapshot and
full hierarchy.

TradeTreeDataV1 explicitly offsets all node pointers and edge_index by num_nodes
in Batch. Country witness IDs retain their values. A fixed [1,31] released_year_mask
collates as [graphs,31], avoiding variable-length tuple truncation. Full profiles
and leaf metadata concatenate in graph/country order.

    batch = next(iter(loader))
    schedule = whole_country_tree_encoder_v1.make_full_tree_schedule_v1(batch)
    batch = batch.to(device)       # Root selects authorized CPU/logical device.
    schedule = schedule.to(device)
    encoder = whole_country_tree_encoder_v1.WholeCountryTreeEncoderV1(
        num_destinations=255, hidden_channels=64).to(device)
    encoded = encoder(batch, schedule=schedule)

Prepare the schedule on CPU after Batch collation and reuse it for that same
unchanged graph/batch. Ragged node/child/rank tensors transfer explicitly, avoiding
per-level device synchronization, dense padding and full-node scatter buffers.

The source defines shared MLPs, native PyG SumAggregation with child-degree
normalization, and complete upward/downward passes. Leaf inputs are profile,
missing, zero-mass and log1p age. Structural inputs are observed height,
descendant count, degree and leaf flag. Country IDs are metadata only.
CountryEncodingV1 returns node_state, leaf_state, country_witness_index and
leaf_graph_index. No temporal scan/GRU, prediction head, loss, optimizer, training
loop or fitted output is implemented. The author constructed/forwarded no encoder.

## H051 boundary

aligned_country_cuts_v1(a_graph,b_graph,c_graph) returns membership vectors in
identical catalogue order. Root may supply them to H051 make_plan_v1 with an
authenticated target-free source record, using fixed/count gates. Country leaves
align years; equal internal integers across years imply no identity. No
edge-continuity witness is supplied.

encoded.leaf_state provides full-tree leaf forcing; node_state[cut_node_index]
provides leaf-expanded cut-node features. Root must freeze that choice, temporal
bootstrap/carry/update, readout/loss and paired controls before a diagnostic. No
future edges or targets may define a/b/c membership or forcing.

## Focused fixture and verification limits

Author validation is syntax/AST/metadata only. Actual synthetic execution is
root-deferred, using a new receipt outside this sealed packet:

    .runtime/env/bin/python3.11 -B research/reviews/TGB_trade_PyG_dataset_source_20261016_v1/focused_synthetic_fixture_v1.py --execute-fixture-root --output NEW_RECEIPT.json

Default mode imports no NumPy/Torch. Explicit mode is CPU-only and uses tiny
synthetic profiles, PyG Data/Batch, schedules and nonlearned sums. It checks
strict-prefix access, staleness/missing/zero flags, complete hierarchy, simultaneous
ties, row/country/destination permutations, cache isolation, witness alignment,
Batch offsets, complete schedules and final-period retention. It creates no
worker/encoder/forward, reads no real data, and fits/scores no model.
No actual fixture PASS is claimed by the author.

Root still owns actual catalogue/materialization, full-data construction, encoder
forward/autograd/host-fit qualification, and any later registered GPU0 experiment.
This is reusable source, not a measured model or novelty result.
