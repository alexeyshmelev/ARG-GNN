# Existing Relate path and prospective handoff

Existing callable path:
src/arggnn/prepare_expanded_full_panel_v1.py:171 execute_parent_v1(... stage="relate" ...)
calls src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py:534 infer_parent.
That function loads scripts/infer_biological_paths_relate_v1.py via helpers():156.

Pinned existing tool recipe, Relate1.2.4:
1. RelateFileFormats --mode ConvertFromVcf --haps INPUT/input.haps --sample INPUT/converted.sample -i INPUT/input
2. Canonicalize .sample IDs against the exact observed diploid header/order, then GenerateSNPAnnotations with --haps, --sample, --poplabels and -o.
3. bash -x RelateParallel.sh -m1.25e-8 -N30000 --threads8 --memory2 --seed REGISTERED_relate_initial --haps INPUT/input.haps --sample INPUT/input.sample --sample_ages INPUT/sample_ages.txt --map INPUT/input.map --annot INPUT/input.annot -o pre_size
4. bash -x EstimatePopulationSize.sh -i pre_size -o post_size -m1.25e-8 --threads8 --num_iter5 --threshold0.5 --years_per_gen28 --seed REGISTERED_relate_size_base --poplabels INPUT/input.poplabels --noplot
5. RelateFileFormats --mode ConvertToTreeSequence -i post_size -o post_size.

Exact source argv are at infer_parent lines555–575. The existing source/tool hashes are in source_manifest.v1.json relate_documentation_refs. These commands document the existing8-thread inference recipe; they are not authorized by the one-CPU generation reservation. A later root inference request must explicitly register its own exact argv, runtime, resource caps and seeds.

The old producers cannot be called unchanged for prospective1611. They impose476diploids/952haps, old observed filenames/manifest and schema, a uniform2Mb map, explicit957-field .haps checks, an old filtering helper, and a cleaner that clips support and resets a local origin. The expanded wrapper seals old geometry and inputs; inventing old counts or a tsinfer tag would be false. No new Relate implementation is included here.

Prospective inference must read only the shared observed bank and map. Private simulation.trees, census nodes, sample-allocation truth and target labels must never be inputs. The generation already writes standard Oxford observed.haps and observed.sample, so use them directly after exact validation rather than an unnecessary VCF roundtrip that can filter/reorder sites. Record authentic argv, tool hashes, registered inference seeds, exit statuses and direct owned wait/drain evidence.

Required joins and coordinates:
- Generation internal x = source_bp -16,051,347 on[0,35,178,458). VCF/Oxford haps POS=x+1; source_bp=x+16,051,347. REF=A and ALT=C encode observed binary0/1. Keep exact phased haplotype order; all3,222 observed columns must match variants.tsv/haplotypes.tsv.
- .sample has1,611 explicit diploid IDs; .haps column2 is variant ID and column3 is POS1. Build sample ages in that diploid/phase order from the exact roster; no guessed node offsets. Native returned sample-node IDs must be read from actual NODE_IS_SAMPLE insertion order and authenticated against header, canonical .sample, every phased .haps column, and actual sample times. Emit a bound roster and explicit haplotype_id→observed_node_id mapping; private simulation-node IDs remain a separate mapping.
- The persisted simulation map is0-based: 0≤x≤L, cumulative cM preserved. Relate uses the haps POS1 domain: its map positions must be x+1, endpoints1 andL+1; preserve the exact cumulative-cM values and outgoing slope, including terminal0. Header adaptation alone is insufficient unless the+1 coordinate shift is also done and receipted. No uniform-map substitution, extrapolation or interval shortening.
- Raw RelateFileFormats conversion retains POS1 site coordinates. Normalize each site by subtracting1 and each physical edge boundary by subtracting1 under a separately qualified full-support policy. Raw first edge0 produces a negative left boundary after shift; clipping that single artificial negative prefix to0 is a documented boundary transform. Do not create unsupported terminal topology. The raw endpoint is typically last observed SNP+1, notL+1. A fullL bank is valid only if genuine native support covers all[0,L), or a separately reviewed explicit inference policy provides real full support. Do not extend the last tree, invent flanking edges, drop sites, reset origin to first SNP or silently reduceL.
- Verify actual normalized site positions/genotypes/phase against every retained observation, exact full span and edge coverage, all samples, positive raw chronology required by the model, sample ages, allele/flip semantics and actual node IDs. Conversion time changes or unsupported terminal flanks require resolution, not a completion assertion.

The generation external-truth command accepts only a complete prospective1611_inferred_bank_receipt_v1 with files exactly observed_trees, roster, observed_node_mapping, sequence_length_bp35178458, true_genealogy_used_as_input=false, and:
  export_position_encoding="POS=rebased_bp+1"
  normalized_tree_coordinates="rebased_bp_zero_based_half_open"
  map_relate_transform_receipt_sha256=ACTUAL_AUTHENTICATED_TRANSFORM_RECEIPT_SHA256

That transform receipt must bind actual source map, shifted Relate map, raw inferred tree, normalized tree, native command/software receipts, ID joins and measured support. A metadata assertion alone cannot qualify the path. Common target midpoint sampling and external census350 truth follow actual inference; this generation wrapper performs neither.
