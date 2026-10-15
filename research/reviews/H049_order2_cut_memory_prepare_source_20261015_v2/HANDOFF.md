H049 fresh prep-v2 repair — source-only root handoff

The preserved v1 attempt failed before any parent recipe was sealed: numpy.iinfo(dtype).max is an integer property, but frozen encode_v1 called it. The only codec change is removal of that call's parentheses. decode_v1 and compact_upper_v1 are byte-identical to the original. All original files, failed output and charge remain intact; root has revoked all future v1 preparation launch/resume/write capabilities.

The new CPU-only execution path is prepare_v2.py → context_factory_v2.py, driven by this folder's root_driver.py. These additive copies bind the corrected codec and actual new preparation producer. Completion metadata explicitly records preparation_execution_version=2 and the corrected codec_ref. No scientific cut construction, observed support, formats, serialization, decoder, schedule or resource caps changed.

Fresh scopes:
- Output: experiments/results/H049_CPU_order2_cut_memory_prepare_v2
- Admission: research/reviews/H049_order2_cut_memory_prepare_source_20261015_v2/admitted_prepare/
- Auxiliary and TMPDIR/TMP/TEMP: its fresh worker_aux/ scope
- Reservation key: H049/H049-order2-prepare-seed101-v2-CPU

The metadata-only plan remains 241,945,331 output bytes under the unchanged 242,221,056-byte reservation/cap. Config and request remain inactive, with root_resource_caps_reviewed=false and execution_enabled=false. Root must create reviewed active copies and explicitly authorize the fresh retry. No admission, guard, launch, real scientific payload read, GPU or Git operation has occurred.

The one authorized focused handmade CPU codec fixture passed: one 953-node star tree / 952 sampled leaves, including one age>350 singleton. Every logical cut tensor roundtrips identically through both repaired and original byte-identical decoders; uint16 and uint32 overflows reject before casting. It made zero optimizer updates and initialized no CUDA. Both inert CLIs and the public metadata preparation plan also returned zero. All 123 source pins match and all inherited 119 pins are unchanged. Exact deltas and full checks are retained beside this file.

The sealed original GPU packet remains frozen. A separately authenticated GPU follow-on will bind this new preparation producer, codec and source manifest, and will require the actual successful v2 completion and actual direct wait/drain; no old-lineage substitution is authorized. Existing memo2 bindings remain unchanged; any namespace-memo integration remains a separate authenticated source review.
