# H018 v4 internal-coordinate verifier diagnosis

The single failing v4 fact is a definite error in the source-derived verification oracle. It does not yet establish that the corrected verifier will accept every retained native coordinate. Root should reverify preserved v4 outputs with the one-line candidate, without rerunning native inference or altering a historical receipt.

## Actual failure and other facts

All eight native commands genuinely exited0 and their owned groups drained. The controller genuinely exited1 after facts verification; original completed_ref is null. Native facts have exactly one failure, no_unreviewed_internal_time_changes, with39 mismatching internal-node IDs among195 internal nodes in five trees. The original facts record mismatch identifiers, not expected/actual time values or ULP differences.

All512 site identities,20,480 phased biological nucleotide calls, six monomorphic patterns, exact fractional tip-date serialization/bounds, exact sample-node dates, sample count/order, complete SNP/tree support and strict native edge chronology passed. The old oracle reported no float32 nonpositive branch or converter chronology-adjustment trigger. These facts remain preserved; no successful overall qualification is inferred from the native exit codes.

## Exact type error

Pinned anc.hpp line31 declares Node.branch_length as double. anc.cpp line52 parses .anc branch lengths with %lf into that double. anc.cpp lines540–550 reconstructs coordinates into vector<float>. For each internal node, line546 adds a float child coordinate to the double child branch_length, applies std::max in double, then assigns the result to the float parent coordinate. The float child is promoted for the addition; the branch is not first rounded to float.

The frozen Python verifier instead uses:
`max(f32(coord32(c)+f32(lengths[c])) for c in children[node])`

Its inner f32(lengths[c]) discards binary64 branch precision too early. The candidate uses the source-typed formula:
`f32(max(coord32(c)+lengths[c] for c in children[node]))`

The converter passes those float coordinates directly to the binary64 node-table API (ConvertToTreeSequence.pinned.cpp lines419–424), so exact equality should compare each stored node time to the correctly reconstructed float32 value represented as binary64. Tip sample nodes follow their separate double-age path and are unchanged.

A finite decimal example has child float32 value36.123451232910156 and branch token1234.6. Premature branch rounding predicts1270.723388671875; the source-typed calculation gives1270.7235107421875. They are adjacent float32 values. Additional examples and the binary32 hexadecimal encodings are in rounding_counterexamples.json. These are synthetic arithmetic demonstrations, not measurements of the39 retained discrepancies.

## Candidate scope and source limits

fixture_and_verify_corrected_v1.py differs from the original pinned helper by exactly that one arithmetic line. make_fixture, rounded_age_token and verify_native remain identical. No native code/binary, input, source date, genotype, tree, node time, output file or comparison tolerance was changed. The candidate retains exact internal-time equality, strict chronology and every prior site/date/support check.

The C++ sources are pinned to b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74. Exact installed binary/build identity remains unproven. The corrected arithmetic is source-supported; an actual complete comparison against retained native output is still required. Passing that comparison would validate this exercised serialization/arithmetic path, not prove a complete source/build identity.

## Biological meaning

This is a synthetic format/runtime qualification. The isolated failing oracle does not show lost alleles, phase changes, failed monomorphic support, changed tip dates or invalid edge chronology; those specific checks passed. The premature-rounding error can falsely classify ordinary source arithmetic as an unreviewed internal-time edit. It neither establishes biological ancestry accuracy nor qualifies the real2074-site/277-monomorphic cohort. Native internal coordinates still have the author's float32 precision; the correction does not upgrade them to double precision.

## Root-owned verification-only next action

Use the new candidate in a separately bound, bounded verification-only root context. Read and SHA-pin the preserved v4 .anc/.mut/.trees and actual five synthetic input assets. Validate those inputs against the original fixture_expected.json and fixed fixture/profile. Load the preserved .trees with the pinned existing tskit runtime, then run the entire candidate verify_native function on the same bytes/object. No native subprocess is needed or authorized by this proposal.

Record all corrected facts, original/candidate source refs, original eight native wait0/drain refs, original genuine controller wait1, the new verifier's actual completion/direct wait, and every old/new/actual internal-time value/bit pattern for the39 original mismatch IDs. Require exact corrected equality for all195 internal nodes and rerun all remaining checks unchanged. A persistent mismatch or any newly detected failure remains unsupported; do not add a blanket ULP allowance.

Write the verification result in a fresh root-owned output under this audit folder. Do not replace native_facts.json, controller_result.json, controller-wait evidence or any native file, and do not fabricate completed.json in the original failed scope. All historical reservations stay fully charged. The original v4 controller remains a failed execution even if a later corrected verification qualifies its preserved native outputs.

The existing empirical packet requires an originally successful qualifier controller/completed record and therefore remains blocked. Accepting a successful new verification-only promotion would require an explicit root-registered gate amendment that preserves original wait1 and links the separately completed corrected verifier. Do not substitute the new verifier wait0 into the old controller receipt.

Only retained compact synthetic facts/controller evidence, source files and stdlib/source/synthetic checks were read or run here. No retained native tree sequence, empirical array, model/GPU/native/process probe, guard/accounting or Git action was performed. Root owns actual verification.
