# Exact observed genotype loading

The new reader changes only the conversion of observed VCF text into the existing integer genotype and allele-count arrays. It uses NumPy byte operations for ordinary phased calls and retains the original scalar interpretation for unusual accepted representations, including explicit -1, signed integers, leading zeros, whitespace and Unicode digits. GT can occur anywhere in FORMAT. Auxiliary values may vary in length. Unphased, malformed or out-of-range calls receive the original rejection rather than a new interpretation.

The full adapter retains the original public-input receipt, checksum, inferred-tree, coordinate, sample identity, reference-label, query-mask and sample-age checks. A source-level check confirms that every line outside the replaced parser block is exactly the existing loader. The original source hash is pinned in qualification. No float calculation, model input definition, simulation, loss, sample order or native competitor implementation changes.

## Qualification

`qualification.json` records 11,173 genotype differential cases, 1,750 complete VCF cases, 17 complete-loader contract cases and an exact source-contract check. Successful outputs must match shape, dtype and every byte. Errors must match exception type and message. Resource callback boundaries must also match. Cases cover missing and multiallelic calls, unusual accepted integer spelling, GT field placement, duplicate GT fields, invalid phase and allele values, malformed columns, repeated or reversed positions, untranslated sample order, retained-site subsets, reverse native-site assignment, header variants, coordinate offsets, invalid metadata and callback boundaries at 4,096 sites.

A bounded real observed VCF test checks 2,048 actual rows across all 952 haplotypes, retaining every site or every third site. The first measurement was 9.14 times faster with all sites retained and 7.77 times faster with a sparse join. Those are parsing measurements on that input, not model training speedups. Repeated measurements are recorded rather than promised as universal.

`qualify_full_train.py` additionally compares the entire actual TRAIN000 VCF using its actual retained-site join. The immutable inputs are hashed before use and the VCF again afterwards. Reference and candidate run sequentially under a 24 GiB address-space limit, each output array is hashed in full, and arrays are released between implementations. No scientific dataset copy or inferred-tree load is needed. `full_train_qualification.json` holds the completed result. The progress file and log are operational evidence only.

## Integration boundary

Use `adapter.load_relate_observation` as the observation loader inside a new pinned wrapper for our preprocessing routes. Bind `adapter.py` and `parser.py` plus the unchanged original dependency closure in the execution manifest. If an existing route refers to the original loader through its module, a scoped wrapper may substitute that one function in its own new worker process before invoking the existing workflow. The new route must never overwrite author source files, the frozen original module, or an already-running worker.

The same arrays can be supplied to the existing GST adaptation without changing the native GST implementation. Existing author data loaders for ARGformer, ARGMix or AncestralPaths do not use this replacement. This is not permission to alter those implementations.

This version creates no persistent cache and consumes no additional scientific storage. Peak scratch use is one VCF line and its byte/index arrays. It removes the per-sample Python loop on common input. It does not remove serial dependencies between trees or promise permanent saturation of a GPU. Any end-to-end speed claim still needs the parent pipeline measurement.

## Full TRAIN000 result

All 1,335,446 retained SNPs across 952 haplotypes match exactly in genotype and allele-count dtype, shape and complete SHA256. The original reader took 422.10 seconds and the vectorized reader took 46.41 seconds, a 9.10 times parsing speedup. Peak process RSS across both sequential passes was 1.79 GiB. All resource callbacks were identical and the original observed input hash was unchanged. This is a full actual TRAIN input parsing result, not an end-to-end model training benchmark.

## Explicit European prediction wrapper

`prediction.prediction_factory` accepts the original prediction arguments plus `prefetch_next_parent=True`. It creates a new source-pinned execution spec. It leaves the original modules and active workers untouched. `preparation.py` is the original complete parent preparation with exactly three changed import destinations, selecting the qualified serial tree, episode and cut implementations. Every other preparation line is checked against the original source.

The current parent is loaded and admitted for prediction first. At most one following parent may be prepared in one background thread. Admission requires current process RSS plus the original 128 GiB preparation headroom and 64 GiB prediction headroom to fit within the original ceiling of at most 512 GiB. If that admission fails, the next parent remains synchronous and starts after the current parent is released. Runtime resource checks retain the original RSS cap. These are the existing measured headroom rules, not a new mathematical upper-bound proof for arbitrary inputs.

There is no completed-parent cache, no eager opening of all parents and no modification of training order. A serialized callback prevents races in resource logging. The producer is joined on success or failure. Cooperative cancellation interrupts it at existing resource-check boundaries when a current prediction fails. No detached preparation thread remains after context cleanup.

`prefetch_qualification.json` records varied preparation timings and admission decisions, exact delivery order, a maximum of two live parents, release of all bank references, no leaked threads, background exceptions, current-prediction exceptions, cooperative cleanup, empty/single-parent behavior, memory admission boundaries, callback serialization and untouched Python/NumPy/Torch RNG. Complete mocked-wrapper execution compares the previous and new product arrays with prefetch both disabled and enabled. The only changed product field is its truthful wrapper source reference. The test explicitly does not claim a real model execution or full-pipeline timing result.

Source checks preserve the original model checkpoint loading, recurrent execution adapter, `predict_logits`, softmax and product construction. Parent-level full input qualification and independent review remain the production gates. Applying this wrapper requires a new reviewed queue execution binding or legitimate future continuation. It does not authorize an in-place edit of an active task.
