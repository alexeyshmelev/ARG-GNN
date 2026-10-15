# Private-owned CPU cache successor

This is a narrow source-only repair. It launches no workers or training, modifies no old bound source, and preserves the failed artifacts. All future experiments are restricted to GPU0 by the user. This CPU cache selects no GPU and authorizes no run; root must provide a fresh GPU0 launcher, binding and admission.

## Evidence and likely mechanism

The preserved failed_training.json reports RuntimeError "unable to mmap 3672 bytes ... Cannot allocate memory (12)" and no complete parent checkpoint. The final drain receipt records 623 retained cache entries and 5,502,663,930 logical bytes across four parents, with genuine worker joins after failure. Root reported vm.max_map_count = 65530. No VMA count was captured before exit. Retained shared-storage mapping exhaustion is plausible, not a proven diagnosis; RAM, cgroup, address-space and other causes have not been excluded. The author made no host, process or resource probe.

AmericanC3BlockV1 is a PyG Data container with 21 MutationRoutedWholeTreeInputsV1 bank fields and four extra block tensors. Four bank fields are level tuples: upward_levels, upward_children, upward_parent_rank and downward_levels. The other 21 top-level fields are ordinary tensors. Each block can therefore have 21 plus the sum of those tuple lengths tensor-storage occurrences; aliasing can reduce the number of distinct storages. A block is not a single shared allocation.

Multiprocessing transports these tensors in shared CPU storages. persistent_workers._take removes its pending/task bookkeeping before returning the received block. The old RAM provider _remember retains that exact block for the immutable cache lifetime. The unchanged stream later clones model inputs, but that does not release the incoming storages still held by the cache. Source refs and failure metadata hashes are recorded in source_pins.v1.json; failure_diagnosis.v1.json separates observations from the unproven mechanism.

## Narrow repair and unchanged contracts

private_owned_tensors_v1.clone_block_private_v1 copies each unique incoming untyped CPU storage once using a private CPU byte-tensor clone. It rebuilds views against the new storage with the original dtype, shape, stride and storage_offset. Tensor-object and storage/view alias relationships within each block are preserved. Nested tuple/list/dict contents retain their structure and metadata is deep-copied. The PyG container is independently copied, including its store, before replacing every item. The scalar ownership report retains no incoming block, tensor or storage reference.

The original full feature digest and logical byte count are recomputed after cloning and must match the already validated transport digest/size before commit. CPU, strided and no-grad contracts remain. The sum of unique backing-storage bytes must fit the original logical byte charge; an unexpected oversized backing view fails before allocation instead of silently changing layout. Source construction uses per-block arrays; this packet does not claim a real bank's physical storage has been measured.

private_owned_cache_v1 subclasses the exact old provider/factory with SHA 7de3ce76df415f20056bdba0ef2b4a2fe48ea1938b270fa0fa4384719ac23936. It changes receipt ownership only:

1. _remember copies and authenticates the block, then delegates the original cache/accounting/RSS commit.
2. _read returns the cached private block to the original seal, so the seal's temporary NumPy focal-row view cannot retain incoming IPC storage.
3. Scalar private-storage diagnostics are appended to the original diagnostics and cleared on cache release.
4. The successor factory selects this provider and binds its helper/provider sources. Original cache caps, all-parent registration, 20 seal plus 600 update releases, persistent workers and genuine final joins remain inherited.

No original module global or method is patched. Feature values and metadata, tensor layouts, feature digests, block order, full-parent recurrence/checkpoint equations, seeds, TRAIN/VAL, optimizer, and the 30 epoch / 600 update budget remain unchanged.

Bounds remain 2 GiB logical per parent, 40 GiB aggregate logical, 88 GiB main RSS, 120 GiB aggregate RSS, 128 GiB root RSS, two workers with 8 GiB each, and 128 MiB blocks. One incoming block, private copy and temporary contiguous hash allocation fit within the existing 1 GiB block-duplicate planning allowance when the original checked block bounds hold. Retained private backing bytes cannot exceed the unchanged logical charge. This does not establish measured host fit or zero anonymous VMAs. No sysctl or multiprocessing sharing-strategy change is made.

## Root integration

Use module research.reviews.C3_private_owned_RAM_cache_source_20261016_v1.tracking_factory_successor_v1, class TrackingObservationProviderFactoryV1. Its API matches the old tracking factory:

- Constructor: (*, operational_caps, immutable_cache_caps, resource_check).
- Call: (plan_ref, *, parent_id, worker_tmp_directory, cache_bytes).
- Property: execution_binding.
- Method: finalize(*, training_failed).

Bind the new tracking, private provider and storage helper together with the frozen original provider, persistent provider/workers, digest, dataset/model sources and actual runtime before admission. This cache contains no physical GPU4/5 eligibility check to adapt; device authorization belongs to root's fresh GPU0 successor.

Keep the old tracking/provider/root contract/canonical shim, old active route and failed result directory unchanged. Existing source pins cannot silently refer to new bytes. Prepare new refs, request, caps, output scope and an authorized GPU0 path. Any later run must use the original seed/config after root's genuine terminal, revocation and capacity gates. The failed cache has no complete parent checkpoint and supplies no reusable learned/cache state. root_integration_delta.inactive.v1.json is inactive and grants no retry authority.

## Focused tensor fixture

focused_storage_fixture_v1.py defaults to inert metadata and does not import Torch. Under root authorization, use the existing Python and a new receipt path:

    .runtime/env/bin/python3.11 -B research/reviews/C3_private_owned_RAM_cache_source_20261016_v1/focused_storage_fixture_v1.py --execute-fixture-root --output NEW_FIXTURE_RECEIPT.json

This fixture uses tiny synthetic shared CPU tensors in a real PyG Data container and the exact AST-extracted original digest functions. It includes mixed dtypes, level tuples, noncontiguous and offset aliased views, an empty view and metadata. Checks cover values/digests/layouts, disjoint private storage, preserved internal aliases, incoming mutation isolation, cached mutation detection, independent metadata and the backing-byte bound. It creates no worker/process, executes no model/GPU call and opens no real observation, genotype, tree or checkpoint. It invokes no guard, admission or resource probe.

The actual tensor fixture remains root-deferred under this source-only assignment. Only syntax and source/metadata checks were performed; no runtime tensor PASS is claimed.
