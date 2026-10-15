# H018 variable-panel H042 interface feasibility

The unchanged learned H042 body can compute on a generic40-haplotype bank once explicit panel/query guards and data preparation contracts are replaced in a separate source adapter. The live H042/H046 sources are unchanged. This proposal is inactive. It does not prepare an empirical bank, read native graph/observedGT bytes, load a real checkpoint, train or apply a model to empirical data.

flag_interface_proposal.py subclasses the original H042 flag model. Its constructor calls the same inherited H038 initializer with expected_haplotypes=40 and adds the identical rank8 adapter. forward_holds copies the original function with only952/686 guards replaced by declared40/24 counts and an error-message change. The exact encode_sites, prepare_tree_base, two PyG tree sweeps, flag injection, leaf context, bidirectional GRU and classifier methods are inherited unchanged. All ordered learned state keys/shapes and30788 parameters match the original952 model. A strict transfer between two freshly initialized synthetic model instances needs no weight reshaping. This is interface evidence, not evidence about real checkpoint predictions.

interface_checks.json records the actual tiny synthetic CPU/FP32 result. Two complete40-leaf trees, four synthetic SNPs and R16/Q24 metadata produce finite[2,40,4] and[2,24,4] outputs, with exact query order. The original H042 full952 model gate and flag wrapper still reject the small bank. The preparation fixture is explicitly synthetic, uses fixture=True only in the test, and is not a production reader bypass. No trained weight file or empirical genotype/graph was opened.

## Exact fixed contracts requiring separate replacement

| Source | Current fixed assumptions | Required H018 contract |
| --- | --- | --- |
| mutation_routed_full_panel_observations_v1.py:15–16,52–90,106–111,118–170 |952 haps,686queries,266 labeled proxies with50/28/102/86 per class;476 subjects;48parents/32TRAIN8VAL8outer;2Mb offset coordinates; Present182/ancient504 strata |40 haps,24queries,16 labeled reference haps with4/4/4/4;20subjects; separate empirical condition, no simulated parent registry or cohort metrics |
| mutation_routed_full_panel_fast_reader_v1.py:158–162 and full_panel_cache_v1.py:34–45,66 |952/686 bank validation and fixed scoring/query cache geometry |New exact40/Q24 bank/cache closure; retain original datatype/topology checks |
| anonymous_clade_flag_control_v1.py:198–207,254–280 |952/686 flag wrapper, constructor and forward count gates |Fresh adapter uses generic exact-clade arrays and declares40/24; learned body unchanged |
| train_anonymous_clade_flag_control_v1.py:160–200,373–386,430–531 |Fixed cache/registry,686-query RAM shape, original scoring targets and metrics |Do not reuse TRAIN/VAL worker or metrics. Root must separately freeze empirical native-hold/readout/seal contract |
| mutation_routed_parent_loader_v1.py:176–228 |Native sequence_length equals coverage width; offset equals coverage left; raw sample times equal source ages; full interval extents |Separate native-minus1/genuine-support reader with exact author date-rounding verification |
| mutation_routed_tree_observations_v1.py:429–543 |280/120 or full952 production count rewrite; float64 sample-date equality; allowed sites outside coverage dropped |Explicit production40/Q24 preparation, retain every2074 original site and277 constants, zero drops, no fixture bypass |
| H046 worker/model integration |Full952/686 marker caches, continuous flank tensors, scoring and RAM shapes |Not adapted or qualified here; this proposal covers the H042 flag body only |

## Reader and feature contract

reader_view.inactive.json pins root's graph-only view and all reviewed source/metadata refs. reader_contract_proposal.py provides pure roster/date/coordinate validation helpers only. No path-based empirical reader, checkpoint loader, writer, launcher or admission boundary is implemented.

The persistent original order is20 adjacent diploid pairs. Reference haplotypes0–15 have observed source labels ANA,CHG,WHG,EHG, four haplotypes per class. Queries16–39 retain reference_class=-1 and query ancestry truth remains absent. Labels are reference proxy annotations, not inferred query truth. Preserve exact sample-index/native-sample identity mapping through every tree; do not infer it from population tables, genotype agreement or a new sort.

Original source ages remain calBP1950 divided by28 and keep their fractional binary64 metadata. Native .anc/.trees author clocks use the reviewed exact six-decimal policy. The new reader must verify native tip times equal the exact parsed author rounding and source-derived bounds, preserve raw native clocks and original source ages separately, then verify any FP32 equality required by the original bank validator. Existing raw float64 equality cannot simply be reused. No time adjustment or tolerance relaxation is proposed; if FP32 equality fails, stop and review a separate contract.

Use only native nodes sampleflags/time and edges parent/child/left/right. Do not iterate native variants or read mutation, ancestral-allele, population or genotype tables. Original frozen observedGT supplies all SNP features separately. This does not reattach mutations or alter trees. Branch concordance summaries are the existing observed SNP-versus-topology discrepancy features computed from originalGT and native descendants; they are not a promise of perfect clade fit.

Preserve feature semantics: site tuple19 =8 donor log frequencies,8 availability masks,panel log frequency,site entropy and called flag. Donor axes are youngerANA/CHG/WHG/EHG followed by olderANA/CHG/WHG/EHG, with age threshold350 generations, alpha0.5 and existing common-cohort callability. Globally absent groups remain masked, with no synthesized donor. Panel frequency/entropy use the observed40 panel. The6 leaf fields are four reference indicators,unknown flag and normalized observed age. Node3 and edge/gate4+5 fields retain their original definitions, including descendant fraction divided by actualH40. No genotype allele-index embedding or native ancestral polarization is added.

Original per-site biological nucleotide identity and phase must be checked against the immutable bridge/readback. Encode categorical alleles consistently within each site, including277 observed monomorphic sites, without filtering, jitter, genotype repair or imputed labels. All2074 sites must lie in genuine support before preparation; the existing general coverage filter must never silently drop one.

## Coordinate view

Native site positionp is local bp1. Model regional bp0 is p-1; absolute bp0 is20000000+p-1. Translate native interval endpoints by-1 and restrict the logical view to genuine support[427,9997139). The raw first native endpoint0 would translate to-1 and must not create supported flanking sequence. The root view's genuine native domain[428,9997140) maps to this exact regional domain. It spans9996712bp, with427bp left and2861bp right unrepresented. No full10Mb claim is allowed.

No native table is edited. The reader may intersect first/last logical interval endpoints with the declared genuine support while preserving all355 marginal holds and all2074 sites. It must certify unchanged tree topology, node times and persistent sample mapping. The pure coordinate helper rejects an empty/dropped hold or any outside-support original SNP. SNP membership and readout lookup use half-open transformed intervals consistently. Root still needs to register a separate empirical readout grid or native-hold output policy; original2Mb truth-scoring bins are not an empirical accuracy protocol.

## Material limits and remaining root work

The actual empirical reader/preparer remains unimplemented and unqualified. Root must review source/data closure, source/native dates, coordinate membership, stable40-sample mapping, zero drops, bounded memory/output, readout/seal format and genuine process evidence before any empirical bank/model action.

The observed pre-size fixedN30000 graph is a distinct inference condition from the post-size TRAIN workflow. The16 reference haplotypes versus266 TRAIN proxies change donor frequencies, sampling noise, age-channel support and graph geometry. The24 ancient Denmark_Neolithic queries differ from686 mixed simulated query cohorts. The2074 ascertained observed SNPs across10Mb, including277 monomorphic-in-panel sites, differ from simulated2Mb mutation density/ascertainment. Native-hold lengths, whole-span GRU history and topology errors may also shift. Keep original trained feature normalizations and no artificial2Mb scan resets; transfer validity remains untested.

The existing genotype-export gate stays failed. Root's separate graph-only view is not lossless mutation export. No empirical ancestry accuracy, calibration, demographic reconstruction or PCA is validated or claimed. No model application, training, checkpoint read, GPU, guard, storage admission or Git action is authorized by this source packet.
