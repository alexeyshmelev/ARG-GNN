# Biological GST source audit

This is a source-only integration review, not a paper-review verdict or an
experimental outcome. The dedicated new files implement admission and a
biological harness around the unchanged existing GST architecture.

## Architecture and provenance

src/arggnn/gst_baseline.py remains unchanged at SHA256
68d7283b99dbabd6c9a3254f8a9bb71272b1288ccea25a6821610804c354cc73.
Its GSTAdaptation already returns position-by-query-by-class tensors through
query_node_index, so no new per-query head, pooling substitute, or architectural
rewrite was needed. The existing independent paper-equation reconstruction
and inspected official source commit are documented in docs/gst_baseline.md.
No original-domain numerical reproduction is claimed.

The exact existing study factory branch fixes cross-attention, span token
weighting, and paper gating. H009's factual score-free selection metadata
identifies the fixed width32/depth2/dropout0/LR0.001/WD0.0001 control. The
separate eight-cell engineering grid is not selection evidence for this task.
Width64 with the same depth, dropout, and optimizer policy was explicitly
prespecified by the primary agent as a capacity sensitivity before biological
GST outcomes.

## Input and target contract

The frozen H020 serializer writes whole observed graphs as a plain dictionary
of GraphSequence tensors and labels_in_input false. The GST integration has
no preparation command and never loads a Kron bank. source47 only projects
the all-node tensor from 48 to 47 columns by dropping original column7.
The edge, graph, identity, clade, interval, query, position, readout, and window
tensors are retained. Float64 coordinates are not cast through float32.

Runtime admission validates the completed prepared source/config hashes,
four parent joins, original tree counts and IDs, exact readout indexes,
retained original centers, all sampled leaves, complete both-direction tree
edges, original coordinates, pure96 classes, and actual-query flags.
The manifest's original_trees_preserved field is an integer count, as defined
by the frozen serializer, rather than a boolean. Source review caught and
corrected this interpretation before any execution.

One complete existing tile is one model context. Full Q184 inference is
followed by the exact actual-query120 index selection. Mixed64 outputs have
no direct loss targets. Handmade gradients demonstrated that direct output
loss is confined to actual120 columns. Mixed or other unscored observations
can still affect the learned context, which is intended.

The method-specific difference is explicit. GST uses F47 on all observed
nodes and full topology. H020 encodes leafF47 with Kron response operators.
A shared task and target/support contract does not make these internal
aggregation inputs identical.

## Fit and evaluation contract

Root-only execution reuses H020's GPU qualification, deterministic policy,
observed parent/role joins, storage admission, external C4 capabilities,
and full_span_score definitions. The new module imports numerical packages
only in handmade helpers or actual fit paths. The default source path has no
Torch or NumPy dependency, as verified in an isolated subprocess that
rejects those imports.

Both arms have 30 epochs, patience8, AdamW0.001/decay0.0001, gradient clip1,
and fixed seeds101/202/303. Complete TRAIN tiles each receive one update per
active arm. All active arms and both VAL parents have predictions sealed
before each epoch's scoring. Selection uses the unweighted mean of two
parent present64 exact-support raw-probability NLL values, with earliest
epoch ties. Selected checkpoints must show changed learned weights.
No calibration fit, probability floor, posthoc renormalization, or TEST
parent is present.

Outputs must be fresh and runtime failures are preserved. Source/config,
observed/prepared manifest, and whole-graph hashes are rechecked before
completion. The root-supplied snapshot and direct accounting reserve
23.52 GB for one two-arm seed, plus50 GB concurrent headroom, under450 GB.
No per-epoch root scan is introduced.

## Validation evidence and limitations

check_01 records the successful default source preflight.
check_02 records the first eight passing handmade checks.
check_03 retains one failed added metadata test. Its fixture removed ID119
instead of ID0, then reinserting119 restored the original ID union, contrary
to the test's intended missing-ID condition. Only the test fixture was
corrected. check_04 records all nine checks passing afterward.

The checks establish the source and handmade semantics only. No actual
biological graph or bank tensor, truth array, trained weights, native model
fit, score output, or TEST data was loaded. No simulation, preparation,
native execution, optimizer fitting, GPU work, installation, Git operation,
or frozen-file edit was performed by this subagent. Runtime memory,
qualified real-data fitting, and biological performance remain for the
primary agent to establish through actual registered execution.

The source handoff is ready for primary-agent review and scheduling after
the current native ARGMix work. An out-of-memory or source/data binding
failure must remain a preserved failed cell. It cannot be resolved by
thinning or replacing the existing GST architecture within this protocol.
