# H018 independent serialized preparation read-back

This new source-only verifier closes the parsing/comparison implementation gap recorded as H018-SR-01. Actual H018 read-back remains unexecuted. Existing bridge, wrapper, inputs, artifacts, guards and journals are unchanged.

Public API: serialization_readback.verify_h018_serialized_v1(basis_bytes, actual_output_bytes). Both arguments are dictionaries of bytes already read by root. This module performs no file, network, process, guard, native, Torch or model I/O. The public API has immutable H018 pins/counts and no fixture or contract override. Only the synthetic checks use the private engine with a tiny synthetic contract.

Root invocation, after its own capacity/admission and limits:

    basis = {name: root_bounded_stable_read(dict(ref))
             for name, ref in module.ORIGINAL_BASIS_REFS.items()}
    actual = {name: root_bounded_stable_read_actual_output(
                  module.OUTPUT_DIRECTORY + "/" + name)
              for name in module.OUTPUT_NAMES}
    receipt = module.verify_h018_serialized_v1(basis, actual)
    receipt["verifier_ref"] = <exact reviewed serialization_readback.py ref>
    receipt["preparation_actual_wait_ref"] = <genuine root direct-wait/drain receipt>
    root_guarded_durable_save(<separate compact receipt scope>, receipt)

The root read/save helpers above are deliberately placeholders, not runnable authority. Root must bind this exact verifier source in its original-guard consumer/runtime closure and enforce canonical non-symlink files, stable bounded reads, resource/deadline caps, complete writer absence/drain and durable receipt writing. If an existing admission does not bind this new verifier, obtain a fresh appropriate current credit-v3 admission; never edit or rebase an old reservation or its initial pins. Do not monkeypatch production constants or call the private synthetic engine for H018. Do not add a receipt inside the ten-file preparation directory, whose exact file set is checked. Use appropriate separately accounted compact control/auxiliary authority and retain all original output files.

Required immutable basis keys and refs are exposed as ORIGINAL_BASIS_REFS: original source_VCF and source_QC, planned_haplotype_roster.json, planned.sample, planned.sample_ages.generations, qualified.regional.map, map_qualification.json, metadata_completion and metadata_independent_result_validation. VCF/QC cap is 80 MB each, map cap 2 MB, remaining basis files 64 KiB. The actual ten-file output bundle is capped at 32 MB including 131,072 bytes slack. The verifier rechecks every basis SHA and every written asset SHA/path/size, then independently parses original VCF/QC and actual output bytes. Root must guard reads before constructing either dictionary. Source bounds do not establish real input size, memory, time, storage capacity or admission. Any actual call needs root's bounded CPU/memory/time supervision.

The comparison covers all 94,892 original candidate identities and exactly 4,148 fixed QC-selected identities in their original order. It parses the source GT and pan_troglodytes independently of the bridge, reconstructs all 40 biological nucleotides from actual haps, checks phase/order and AA/DA XOR recoding, and compares every audit field and code vector. It also parses the actual sample/age/map files and binds exact original diploid/haplotype IDs, R16/Q24 order, fractional calBP1950/28 dates, original C4 donor roster and piecewise map/local coordinates. It verifies the original notice, provenance, report counts/digests and completion claims. No additional QC threshold, intersection, monomorphic filter, site rescue or genotype/HS mutation is used.

Success returns a compact preparation-serialization receipt, including actual output hashes, exact basis refs, admission/completion identities, count/digest results and verified monomorphic-detail artifact. Full monomorphic identities stay in the unchanged, independently checked bridge_preparation.json; the compact receipt carries their count/hash. A nonzero monomorphic count can pass preparation read-back and still blocks native admission. Exceptions leave identity unqualified. Never convert exceptions to zero unsupported counts, drop offending sites or edit output to pass.

The original metadata child's OS exit remains unknown, original wait0 unavailable and original 5 MB charge retained. Neither source checks nor a successful read-back certify capacity, admission, direct OS wait, descendant drain, native inference, tip-date serialization, native sample-node/site/coverage/chronology round-trips, HS, GPU, model prediction or ancestry accuracy. All native/HS/GPU permissions stay false. Root must require the genuine preparation wait/drain and this actual read-back before any later promotion. Native fractional-age serialization policy and all native round-trips remain separate prerequisites.

Synthetic checks only:

    python3 -B research/reviews/H018_bridge_serialization_readback_source_v1/synthetic_checks.py

The checks build all inputs in memory. They import only this new pure verifier and standard library. They do not read real H018 payloads or import/execute the bridge, guard or pipeline. Two positive cases exercise ancestral-ALT flips, phase, fractional dates and retained monomorphic/no-monomorphic conditions. Negative cases rehash corrupted output assets before comparison, so phase swaps, forged source audits, dropped monomorphic sites, order/identity changes, date rounding, map substitution, count changes and false wait/native claims must fail semantic validation. The public API also rejects the synthetic basis. See synthetic_checks.json for the recorded result.
