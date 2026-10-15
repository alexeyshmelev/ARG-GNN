# H018 observed2074 native vector failure diagnosis

The original strict qualification remains failed. This is a real biological nucleotide/phase export mismatch, not an AA/DA oracle or site-order error. No native rerun, site filtering, mutation reattachment, input/output edit, chronology change or relaxed biological check was performed.

Of2074 original sites,198 polymorphic sites differ at692 of82960 ordered haplotype nucleotide cells. All277 monomorphic sites match exactly; no calls are missing.1876 exported site vectors match. The mismatch cells comprise264 reference and428 query calls.430 unphased diploid genotype cells also differ, so this is not solely an exchange of phase within adjacent haplotype pairs.

An independent stdlib parser reconstructs the carrier descendants of every native .anc branch and applies the actual pinned converter rule to every .mut row. Predicted export vectors agree with all2074 recorded .trees-derived vectors, with zero discrepancies. Original .haps AA/DA decoding, polarization-equivalence source REF/ALT decoding and independently serialized input vector digest agree at every site. The verifier correctly compares biological nucleotides in unchanged sample order.

| Native .mut category | Sites with export error | Wrong exported calls | Native branch semantics still wrong |
| --- | ---: | ---: | ---: |
| One branch, unflipped |178|178|178 sites /178 calls|
| One branch, flipped |7|278|2 sites /2 calls|
| Multiple branches, unflipped |11|175|0|
| Multiple branches, flipped |2|61|0|
| Total |198|692|180 sites /180 calls|

Native inference and converter export contribute different failures:

1. The author builder source sets thr=int(0.03*N), equal to1 for40 haplotypes. MapMutation accepts a clade when its carrier mismatch count is at most this threshold. After applying all native branch records and their flip semantics symbolically,180 sites retain exactly one wrong original observed call each.178 are unflipped; the two flipped sites are indices1160 and1491. These are genuine approximate inferred mutation/clade mappings.
2. The converter creates each site's ancestral state from mutation_type[0], inserts a derived mutation only when branch.size()==1 and does not consult the flipped flag. It therefore ignores all13 multibranch mappings and misinterprets all9 flipped mappings, with two sites shared between these sets. Full native branch/flip semantics recover all original calls at13 multibranch sites and five of seven flipped singlebranch sites.18 sites fail exclusively during export; two additional flipped sites combine export semantics and approximate native mapping.
3. is_not_mapping=0 is not a certificate of exact matching. The writer sets that field solely from whether branch_count exceeds1.

The complete exact site/mismatch/branch/flip classification is in site_diagnostics.csv. diagnosis.json is a compact machine-readable summary. input_source_refs.json pins the retained facts, native .anc/.mut/.trees, original observed bridge files, verifier and public C++ sources. checks.json records6951 exhaustive row/source assertions; diagnosis_stderr.log is empty. These are read-only Python/stdlib checks, not native qualification reruns.

The retained355 trees,27690 edges, original fractional tip dates, strict chronology, complete sample connectivity, exact native topology/support and genuine9996712bp span can remain positive structural facts. They do not establish lossless genotype export or biological ancestry accuracy. An inferred tree need not perfectly fit recurrent/error SNP patterns. A future graph-only condition using the native inferred topology with separate immutable original observed SNP features would require its own registered contract and evaluation. That would be a new condition, not promotion or repair of this failed strict export gate.

The original controller exit1, failed facts, null completion, artifacts and full charge remain unchanged. Exact installed binary source-build identity remains unproven. Root owns all execution and any subsequent protocol decision.
