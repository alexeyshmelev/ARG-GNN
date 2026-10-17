# Frozen TreeLoom scientific payload read trace

Conclusion: the frozen bounded TreeLoom trainer does not dereference a native
raw-path/token corpus, vocabulary, row-order corpus metadata or path-occurrence
mapping. Preparation JSON is authenticated and consumed as provenance only.

| Frozen new module | Exact route and reads |
|---|---|
| __init__.py | Inert namespace; no reads. |
| borrowed_gpu0_v1.py | configure_controller_binding_v1, _registry_v1, controller_v1, attach_v1: source/lease/registry/process identity and lock metadata; no scientific corpus. |
| bounded_execution_v1.py | BoundedExecutionV1 and train_step_v1: in-memory observed bank/episodes/cut/focals/targets; no file I/O. |
| contract_v1.py | validate_input:43-104 authenticates pinned source/config/review/runtime/storage/acceptance/barrier JSON. It never recursively opens refs carried inside native completion JSON. |
| parent_v1.py | execute_v1:21-143 authenticates launcher/control metadata and reads owned process/output evidence; no scientific corpus. |
| parity_v1.py | synthetic_parent_v1:10-55 and run_parity_v1:57-102 use synthetic in-memory observations; no real corpus. |
| targets_v1.py | TrainTargetJoinV1.join:51-97 opens fixed focal registry and separately genuine TRAIN truth ZIP/NPZ after observation seal. |
| worker_v1.py | run_v1:70-75 opens observed/native JSON. Nested load_parent:80-108 calls original observation loader then full observed preparation. Checkpoint payload I/O is output-only provenance from this trainer, never native corpus. |
| gpu0_inventory_v1.py | GPU0InventoryV1/inventory_v1 use NVML resource metadata; no scientific data. |

The transitive source reader is
src/arggnn/american_c3_gnn_dataset_v1.py:load_observation:28-92:
observed manifest, roster and preparation JSON(:30); complete inferred tree_ref
(:67); complete phased genotypes_ref(:74); positions_ref(:75); allele rows
alleles_ref(:79); focal registry(:83). Its preparation checks(:31-34) require
observation_ref, focals_ref, parent_id, genealogy_kind, class_order and
query_ancestry_truth_opened=False. No schema/status/native-corpus-ref requirement.
common_v1.bound:26-28 hashes each directly consumed ref; it does not recursively
hash the refs embedded in JSON. evaluate_v1.registry:29-40/load_npz:24-26 read
all focal NPZ fields and validate the six required identity fields.

american_full30mb_prepare_v1.prepare_from_loaded_observation_v1:22-123 loads no
files; it consumes every original observed SNP(:60-64) and every native tree
(:67-80), and joins exact focal holds/samples(:88-105). Native preparation ref
is copied into bundle/metadata provenance. Genuine TRAIN truth is an additional
scientific payload; saying only trees and focals would therefore be incomplete.

Source-only inspection established the dependency finding. The compact fixture
uses only a tiny constructed RAM tree sequence and source/config pins. It is not
a real TRAIN parent rerun or a large-data equivalence measurement.
