# Biological estimand and model input fields v1

This handoff describes the frozen biological simulation and candidate implementation from source and configuration. It contains no execution or score claim and does not transfer the earlier engineering cohort's estimand to this experiment. The proposal contract is not the execution authority. The execution JSON and the candidate JSON define the implemented biology, panel and fields.

## Census ancestry estimand

Time is measured in generations before the present. For query haplotype q and physical position x, traverse the raw simulated marginal tree upward from q to its flagged census ancestor at time 350. The target is that ancestor's population, mapped to C4 as follows.

| Census population ID | Simulated name | C4 index | Manuscript label |
|---:|---|---:|---|
| 8 | Ana | 0 | ANA |
| 7 | CHG | 1 | CHG |
| 4 | WHG | 2 | WHG |
| 3 | EHG | 3 | EHG |

This is ancestry at the 350-generation census. It is not the query's sampling population, observed cohort name, sampled reference identity or a posterior generated from the inferred tree. The code requires a census ancestor at exactly 350 and rejects other census populations.

The backward admixture events are present_bronze at 166 into Neo/Yam with probabilities 0.5/0.5, BAA at 169 into CHG/Ana with probabilities 0.25/0.75, Yam at 177 into CHG/EHG with probabilities 0.5/0.5, and Neo at 259 into WHG/Ana with probabilities 0.25/0.75. Thus all query admixture pulses precede census time 350, while the WHG/EHG merge at 600 and CHG/Ana merge at 800 occur afterward. Their ancestors NE/WA merge into trunk at 1500. These fixed times come from the original full author age file and remain fixed after panel reduction. Population IDs 0–10 are present_bronze,Yam,Neo,EHG,WHG,NE,BAA,CHG,Ana,WA,trunk, with initial size parameters 50000,20000,50000,3000,2000,15000,20000,3000,5000,15000,15000 in the same order.

The external six-path encoding is 1 Ana_nonBAA, 2 CHG_nonBAA, 3 WHG, 4 EHG, 5 Ana_viaBAA and 6 CHG_viaBAA. The actual implementation adds four to base path 1 or 2 when the query's cohort is BAA. This cohort convention is confined to external target construction. C4 collapses paths with vector [0,1,2,3,0,1], eliminating that path distinction. The candidate predicts C4 directly.

The frozen simulation is a 2,000,000bp discrete genome, Hudson ancestry, uniform recombination 1e-8 per bp per generation and JC69 mutations at 1.25e-8. Contig name 22 does not imply empirical GRCh37 coordinates. Only two TRAIN and two VAL parents are defined, with no TEST parent in this pilot.

Source: [src/arggnn/biological_paths_simulation.py:20](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_paths_simulation.py:20) lines 20–42 and [src/arggnn/biological_paths_simulation.py:173](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_paths_simulation.py:173) lines 173–205 define the events and class order. [src/arggnn/biological_paths_simulation.py:262](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_paths_simulation.py:262) lines 262–290 define the exact target and six-path collapse. [src/arggnn/biological_paths_simulation.py:302](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_paths_simulation.py:302) lines 302–327 define the simulator and observation separation. The author event-age formulas are in [.runtime/author_sources/ancestralpaths_77db3cfae100/simulate.py:50](/workspace-SR003.nfs2/ARG-GNN/.runtime/author_sources/ancestralpaths_77db3cfae100/simulate.py:50) lines 50–80. The frozen execution config is [configs/biological_paths_execution_pilot_v1.json:21](/workspace-SR003.nfs2/ARG-GNN/configs/biological_paths_execution_pilot_v1.json:21).


## Exact observed roles and ages

The 140 diploids supply 280 consecutive haplotypes. A diploid at roster index d has slots 2d and 2d+1. Slots below are zero-based roster indexes, not an assumption about inferred node IDs. Reference-role samples precede query-role samples within each cohort.

| Cohort and sampling population | Role | Haplotypes | Roster slots | Age in generations |
|---|---|---:|---|---|
| present_day / present_bronze | actual query | 64 | 0–63 | 0 |
| Bronze_Age / present_bronze | mixed reference only | 16 | 64–79 | 120,130,134,137,141,145,151,161 |
| Bronze_Age / present_bronze | actual query | 16 | 80–95 | 135 |
| BAA / BAA | mixed reference only | 8 | 96–103 | 136,136,168,168 |
| BAA / BAA | actual query | 8 | 104–111 | 155 |
| Neo / Neo | mixed reference only | 32 | 112–143 | 148,166,169,169,173,175,181,183,188,194,195,198,202,212,223,237 |
| Neo / Neo | actual query | 16 | 144–159 | 200 |
| Yam / Yam | mixed reference only | 8 | 160–167 | 159,165,173,175 |
| Yam / Yam | actual query | 16 | 168–183 | 170 |
| WHG / WHG | pure reference, class 2 | 24 | 184–207 | 216,225,228,242,251,267,274,284,295,324,329,425 |
| EHG / EHG | pure reference, class 3 | 24 | 208–231 | 176,190,195,200,205,212,247,259,293,296,341,389 |
| Ana / Ana | pure reference, class 0 | 24 | 232–255 | 227,251,252,255,261,268,272,276,288,296,296,310 |
| CHG / CHG | pure reference, class 1 | 24 | 256–279 | 184,185,185,311,329,334,347,347,352,355,355,473 |

Each listed reference age belongs to one diploid and repeats for its two haplotypes. Query ages are constant within the listed cohort. Reference dates are the sorted original author cohort dates at index floor((j+0.5)m/n), where m is the original cohort size and n the selected reference-role diploid count. The frozen JSON retains the original author line for every selected date. A stdlib source/config check reproduces these dates from the age text without importing the simulator.

Only the 96 pure reference haplotypes have observed source-class labels, 24 each. Query120 and mixed64 have null reference class. Pure reference ages may exceed 350, including WHG425, EHG389 and CHG352/355/473. They remain observed source proxies and are not assigned a query census target. Every actual query is younger than the census. Sample age and donor class are assumed observed panel information, not inferred from query targets.

The names Bronze_Age, BAA, Neo and Yam select the simulation sampling population and date schedule. They are not extra input one-hots in the candidate. In particular, mixed64 are not assigned their cohort's mixture proportions or local ancestry classes.

Source: [src/arggnn/biological_paths_simulation.py:29](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_paths_simulation.py:29) lines 29–40 and [src/arggnn/biological_paths_simulation.py:63](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_paths_simulation.py:63) lines 63–97 define the panel, ordering and quantile rule. [configs/biological_paths_execution_pilot_v1.json:368](/workspace-SR003.nfs2/ARG-GNN/configs/biological_paths_execution_pilot_v1.json:368) contains all 280 frozen records. [src/arggnn/biological_candidate.py:232](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_candidate.py:232) lines 232–255 validate classes, pair roles and present64/ancient56 query counts.


## Whole tree, boundary partition and scored outputs

Let m(i) be the actual inferred node ID assigned to roster slot i by the verified converter mapping. In every complete observed marginal tree, all 280 sampled leaves are retained along with all internal nodes and edges. The Relate converter sample flags, checked haplotype column order and observed ages define m. The candidate does not substitute original simulation node IDs for inferred IDs. This source-only review specifies the exact roster-to-node mapping procedure and exact role slots. It does not assert that an individual runtime mapping equals the identity mapping without inspecting its separate completed receipt.

The boundary bank is sorted by actual persistent sample ID. Its query/reference lookup restores the declared input orders, so numerical ID values serve only identity joins.

| Set | Exact roster slots | Role in candidate |
|---|---|---|
| B280 | 0–279 | full persistent leaf boundary |
| Q184 | 0–183 | all non-pure leaves in coupled dynamics |
| R96 | 184–279 | pure observed reference sources |
| actual query120 | 0–63,80–95,104–111,144–159,168–183 | supervised and reported predictions |
| mixed64 | 64–79,96–103,112–143,160–167 | latent dynamics states with no label or score |

Q184 and R96 exhaust B280 and are disjoint. Their node vectors are m applied to those slot lists. The model output order follows Q184. Because the dynamics slots are 0–183 here, actual-query output indexes are exactly 0–63,80–95,104–111,144–159,168–183. The output tensor is explicitly indexed to those 120 outputs before loss or sealed prediction. Mixed64 can transmit information through dynamics, but their actual-query flag is zero and they never enter loss, checkpoint selection or metrics.

R96 is stored in roster order WHG, EHG, Ana, CHG. The feature and output class order remains ANA, CHG, WHG, EHG. These orders must not be conflated.

Each raw graph uses eight base fields and 40 fixed observed-reference summaries, giving 48 whole-tree columns. Leaf-interface construction eliminates internal nodes into the response operator while retaining all 280 leaves. It excludes raw parent-branch column 7, leaving 47 selected leaf fields. All 280 leaves enter the same graph-local BoundaryResponseGNN. The model partitions its encoded context into Q184 and R96 and emits the Q184 outputs before selection of the actual query120.

Source: [scripts/infer_biological_paths_relate_v1.py:246](/workspace-SR003.nfs2/ARG-GNN/scripts/infer_biological_paths_relate_v1.py:246) lines 246–259 establishes m. [scripts/train_ancestralpaths_biological_v1.py:177](/workspace-SR003.nfs2/ARG-GNN/scripts/train_ancestralpaths_biological_v1.py:177) lines 177–192 checks the full actual mapping and preserved observed attributes. [src/arggnn/biological_candidate.py:232](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_candidate.py:232) lines 232–267 defines Q/R/output indexes. [src/arggnn/kron_interface.py:404](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/kron_interface.py:404) lines 404–423 sorts the boundary and selects fields. [src/arggnn/kron_interface.py:501](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/kron_interface.py:501) lines 501–519 enforces an exhaustive disjoint partition. [src/arggnn/anchor_kernel.py:389](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/anchor_kernel.py:389) lines 389–412 and [src/arggnn/biological_candidate.py:653](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_candidate.py:653) lines 653–665 establish context and output selection.


## All 47 fields and exact formulas

Use zero-based bank columns 0–46. The whole-tree source columns are identical for base 0–6. For class c in [ANA,CHG,WHG,EHG] and summary j in 0–9, whole column 8+10c+j becomes bank column 7+10c+j. Whole column 7 is excluded.

For an observed tree T, let R_c be its 24 pure labeled reference leaves, N_c=24, and N=96. Let g(t)=log(1+t)/log(10001). The implementation first stores g(observed node age) as float32 in raw column 6. Define z(u,r) as that stored encoded value at MRCA_T(u,r). A reference descended from u has MRCA(u,r)=u, including a reference's self relation. These are encoded absolute MRCA ages, not query-adjusted time differences or raw-generation distances.

For each node u and class c, define D_c(u) as the number of R_c leaves descended from u, a_c(u)=min over r in R_c of z(u,r), h_c(u) as the number attaining that minimum, mu_c(u)=mean z(u,r), and s_c(u)=sqrt(max(N_c^-1 sum over r in R_c of z(u,r)^2-mu_c(u)^2,0)). Define a_*(u)=min_c a_c(u).

The kernel density is d_(c,sigma)(u)=N_c^-1 sum over r in R_c of exp(-z(u,r)/sigma). Kernel evidence k_(c,sigma)=d_(c,sigma)/sum_b d_(b,sigma), at sigma=0.05,0.1,0.2 in encoded-time units. Source code computes this with logaddexp and class log normalization to avoid underflow.

| Bank column | Base field | Formula at leaf u | Permitted information |
|---:|---|---|---|
| 0 | observed reference ANA | 1[u in R_ANA] | known pure donor class only |
| 1 | observed reference CHG | 1[u in R_CHG] | known pure donor class only |
| 2 | observed reference WHG | 1[u in R_WHG] | known pure donor class only |
| 3 | observed reference EHG | 1[u in R_EHG] | known pure donor class only |
| 4 | actual query flag | 1[u is one of actual query120] | declared prediction role |
| 5 | observed sample flag | 1[u is a sampled leaf] | sample status, constant 1 in B280 |
| 6 | encoded age | float32(g(observed age(u))) | observed sampling age |

The next table enumerates every remaining column. Each row is repeated in source-major order for the four allowed pure-reference categories.

| Summary j and field | ANA | CHG | WHG | EHG | Formula | Permitted information |
|---|---:|---:|---:|---:|---|---|
| 0 descendant_reference_fraction | 7 | 17 | 27 | 37 | D_c/N_c | observed topology and known pure donor labels |
| 1 panel_reference_fraction | 8 | 18 | 28 | 38 | N_c/N, exactly 1/4 | known reference panel composition |
| 2 minimum_encoded_tmrca | 9 | 19 | 29 | 39 | a_c | observed MRCA ages to R_c |
| 3 mean_encoded_tmrca | 10 | 20 | 30 | 40 | mu_c | observed MRCA ages to R_c |
| 4 std_encoded_tmrca | 11 | 21 | 31 | 41 | s_c | observed MRCA ages to R_c |
| 5 source_minimum_reference_fraction | 12 | 22 | 32 | 42 | h_c/N_c | exact within-source minimum ties |
| 6 global_nearest_reference_vote | 13 | 23 | 33 | 43 | h_c 1[a_c=a_*] / sum_b h_b 1[a_b=a_*] | exact global reference minimum ties |
| 7 kernel_evidence_scale_0.05 | 14 | 24 | 34 | 44 | k_(c,0.05) | fixed kernel of observed MRCA ages |
| 8 kernel_evidence_scale_0.1 | 15 | 25 | 35 | 45 | k_(c,0.1) | fixed kernel of observed MRCA ages |
| 9 kernel_evidence_scale_0.2 | 16 | 26 | 36 | 46 | k_(c,0.2) | fixed kernel of observed MRCA ages |

Nearest votes count individual references in minimum-time ties. They do not allocate equal weight to tied classes unless tie counts are equal. Minimum-reference fraction is normalized within a source and differs from the global nearest vote. For an unlabeled terminal query or mixed leaf, all descendant fractions are zero. At a pure reference leaf, the self relation gives descendant fraction 1/24 in its own class. Panel fractions are constant 1/4 here.

The adapter computes summaries in float64 from the stored float32 encoded ages, then casts its appended fields to the raw feature dtype float32. Leaf-bank preparation stores those values in float64 and neural input transfer uses float32. Thus log encoding is not reconstructed from pristine floating-point ages when computing the summary fields. Ties and minima use the actual stored encoded values. The formulas above represent that implemented encoding and rounding path.

No feature reads a census flag, hidden population annotation, native six-path target, query C4 target, oracle tree or truth-derived purity. Allowed donor labels are ancestry-related observed panel information. The inferred tree and ages intentionally contain ancestry signal. The separation claim concerns unavailable query ancestry targets, not statistical independence from ancestry.

Source: [src/arggnn/biological_candidate.py:367](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_candidate.py:367) lines 367–400 defines base columns and adapter settings. [src/arggnn/reference_encoder.py:23](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/reference_encoder.py:23) lines 23–31 fixes summary order. [src/arggnn/genealogy_baselines.py:49](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/genealogy_baselines.py:49) lines 49–60 establishes the stored encoded-time input and reference validation. [src/arggnn/reference_encoder.py:103](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/reference_encoder.py:103) lines 103–167 implements all summary formulas and dtype casts. [src/arggnn/reference_encoder.py:169](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/reference_encoder.py:169) lines 169–175 appends source-major features. [src/arggnn/biological_candidate.py:560](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_candidate.py:560) lines 560–579 checks selected fields, actual-query flags and neural dtype transfer.


## Other observed inputs outside the 47-field vector

Raw parent branch g(age(parent)-age(u)) is constructed in whole-tree column 7, with root value zero, and is excluded from leaf_x. This exclusion does not discard branch geometry. Plain edge resistance is 1e-4 times the decoded observed duration. The two age branches use observed endpoint ages with Phi_s(t)=4t-3s atan(t/s), s=350 or 1400, and resistance 1e-4[Phi_s(parent age)-Phi_s(child age)]. Internal-node elimination preserves the 280-leaf response. All branches share the plain full-boundary mean-diagonal normalization. The constant age scale 350 is a declared operator parameter, not a census target read.

The 16 clade landmark indicators are also separate GraphSequence metadata. They use pure-reference labels only to select four roster-ordered landmarks per class, at within-class haplotype offsets 0,7,15,23. They are not among the 47 leaf fields. The current BoundaryResponseGNN receives leaf_x and the plain normalized boundary response, not clades, cohort names, numerical IDs, genomic positions or interval widths. Later temporal/anchor operators separately receive observed physical intervals and fixed readout coordinates.

Primary graphs are post-five-iteration Relate inferences from observed genotypes and ages. Simulation removes census nodes and truth annotations before drawing mutations. The inferred graph cleanup removes populations, individuals, migrations, provenance and metadata again. Genome positions use the declared local-to-original offset and original bp/2,000,000 normalization. Eight 250kb tiles retain complete marginal trees and all leaves. Fixed original 100bp centers and clipped bin support do not depend on ancestry labels. External targets enter only actual-query loss and authorized scoring, after feature/bank construction.

Source: [src/arggnn/biological_paths_simulation.py:223](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_paths_simulation.py:223) lines 223–259 and [scripts/infer_biological_paths_relate_v1.py:292](/workspace-SR003.nfs2/ARG-GNN/scripts/infer_biological_paths_relate_v1.py:292) lines 292–317 remove truth annotations. [src/arggnn/biological_candidate.py:341](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_candidate.py:341) lines 341–393 constructs landmarks, geometry and coordinates. [src/arggnn/kron_interface.py:84](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/kron_interface.py:84) lines 84–100 defines plain resistance. [src/arggnn/age_interface.py:41](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/age_interface.py:41) lines 41–99 and [src/arggnn/age_interface.py:288](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/age_interface.py:288) lines 288–350 defines age branches and shared normalization. [src/arggnn/boundary_memory.py:138](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/boundary_memory.py:138) lines 138–156 specifies the graph-local encoder. [src/arggnn/biological_candidate.py:479](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_candidate.py:479) lines 479–484 fixes biological operator parameters.


## Ready-to-use LaTeX fragment

The fragment below uses zero-based neural column indexes and describes source-defined inputs. It makes no execution, fit-quality or superiority claim.

~~~latex
We define local ancestry for a query haplotype at position $x$ as the
population of its simulated ancestor at a census 350 generations before
the present. Census population IDs 8, 7, 4 and 3 map to ANA, CHG, WHG and
EHG. This target differs from the query's observed sampling cohort.
The panel contains 280 haplotypes, comprising 96 pure source references,
64 unlabeled mixed samples and 120 queries. The dynamics partition has
$Q=184$ non-reference leaves and $R=96$ pure reference leaves.
Only the 120 declared queries receive supervision or contribute to
reported predictions. Mixed samples have zero query flag and no source
label. Pure references can predate the census and retain their observed
source-proxy identities.

Each complete inferred marginal tree is reduced onto all 280 sampled
leaves. The input vector has 47 fields, comprising four pure-reference
indicators, the actual-query and sample flags, encoded observed age, and
ten observed-reference summaries per source. The parent-branch field is
excluded from the leaf vector while observed branch geometry remains in
the response operator. No query ancestry target, census annotation,
cohort one-hot or numerical node ID enters the vector.

Let $R_c$ contain the $N_c=24$ pure references of source $c$ and let
$z_{ur}$ be the stored encoded observed age of
$\mathrm{MRCA}(u,r)$, using $g(t)=\log(1+t)/\log(10001)$.
Write $D_c$ for the descendant reference count, $a_c=\min_{r\in R_c}z_{ur}$,
$h_c=\#\{r\in R_c:z_{ur}=a_c\}$, $\mu_c$ and $s_c$ for the mean and
standard deviation of $z_{ur}$, and $a_*=\min_c a_c$.
For $\sigma\in\{0.05,0.1,0.2\}$, define
\[
 d_{c,\sigma}=\frac{1}{N_c}\sum_{r\in R_c}\exp(-z_{ur}/\sigma),
 \qquad
 k_{c,\sigma}=\frac{d_{c,\sigma}}{\sum_b d_{b,\sigma}}.
\]
The sums and minima use inferred topology and ages together with known
pure-reference labels. They do not use the query's census ancestry.

\begin{table}[t]
\centering
\small
\caption{Observed input fields with zero-based neural column indexes.
Source columns follow ANA, CHG, WHG and EHG order.}
\begin{tabular}{lrrrr}
\hline
Base field & \multicolumn{4}{c}{Column} \\
\hline
Pure-source one-hots & 0 & 1 & 2 & 3 \\
Actual-query flag & \multicolumn{4}{c}{4} \\
Sample flag & \multicolumn{4}{c}{5} \\
Encoded observed age & \multicolumn{4}{c}{6} \\
\hline
Summary & ANA & CHG & WHG & EHG \\
\hline
$D_c/N_c$ & 7 & 17 & 27 & 37 \\
$N_c/96$ & 8 & 18 & 28 & 38 \\
$a_c$ & 9 & 19 & 29 & 39 \\
$\mu_c$ & 10 & 20 & 30 & 40 \\
$s_c$ & 11 & 21 & 31 & 41 \\
$h_c/N_c$ & 12 & 22 & 32 & 42 \\
$\frac{h_c\mathbf{1}[a_c=a_*]}
 {\sum_b h_b\mathbf{1}[a_b=a_*]}$ & 13 & 23 & 33 & 43 \\
$k_{c,0.05}$ & 14 & 24 & 34 & 44 \\
$k_{c,0.1}$ & 15 & 25 & 35 & 45 \\
$k_{c,0.2}$ & 16 & 26 & 36 & 46 \\
\hline
\end{tabular}
\end{table}
~~~

The manuscript should preserve three distinctions: the target is census population rather than cohort, Q184 is a dynamics partition rather than 184 labeled queries, and 47 leaf fields differ from 48 whole-tree fields. Older engineering-cohort results and native baseline feature contracts are separate conditions.


## Source pins

The AncestralPaths source commit is 77db3cfae100641c35218b87e1451e14f3074666. Hashes below identify the inspected bytes. Candidate dependencies match its frozen source_sha256 entries. All source citations use these pinned versions.

| Source or config | SHA256 |
|---|---|

| [configs/biological_paths_execution_pilot_v1.json](/workspace-SR003.nfs2/ARG-GNN/configs/biological_paths_execution_pilot_v1.json:1) | aca989588f46993dd1636f697cb3bac80d00af69ce738c06719d104075bb21c9 |

| [src/arggnn/biological_paths_simulation.py](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_paths_simulation.py:1) | f0f4520b8fdfd4074bc78024073dd5b331ecac996911f40af787b6df68f213e8 |

| [.runtime/author_sources/ancestralpaths_77db3cfae100/simulate.py](/workspace-SR003.nfs2/ARG-GNN/.runtime/author_sources/ancestralpaths_77db3cfae100/simulate.py:1) | 4ae51e2af2a7cc16393d2611d6745e47f63ed37d12c15631456abc135e6cd969 |

| [.runtime/author_sources/ancestralpaths_77db3cfae100/example/mesoneo_ages_pop.txt](/workspace-SR003.nfs2/ARG-GNN/.runtime/author_sources/ancestralpaths_77db3cfae100/example/mesoneo_ages_pop.txt:1) | 3cadb16abbc0a1913eadb183d324eadc26c035d61e5116cd58675f52a942b7de |

| [configs/biological_candidate_pilot_v1.json](/workspace-SR003.nfs2/ARG-GNN/configs/biological_candidate_pilot_v1.json:1) | 4151e7e156db3f188512bcc738b43597407d7707bc77c420509165c2e9bfc2f1 |

| [src/arggnn/biological_candidate.py](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/biological_candidate.py:1) | 6b3a25d0e59026cab7189032eb41221feb7d1bdf69e15f6c737a672dfb711dd3 |

| [src/arggnn/reference_encoder.py](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/reference_encoder.py:1) | 039f3b25383ea92f243090c2eaf592df590768fea7abf41d15fd7c34a7715138 |

| [src/arggnn/genealogy_baselines.py](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/genealogy_baselines.py:1) | 388e84c183413a79db8e0e75c86f1ce202b4bb9999473b729be045d7459dc9f8 |

| [src/arggnn/kron_interface.py](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/kron_interface.py:1) | 09ec6286c185e7aa2b13d76dcbe8fea24b3a1c785a6619ac9bc5464e0de66d69 |

| [src/arggnn/age_interface.py](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/age_interface.py:1) | 15f9615a84a230386d8f71f5a317a46051c395f75408c79a6213f1b07f101f3d |

| [src/arggnn/boundary_memory.py](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/boundary_memory.py:1) | a056cd10d1c848d7dabdc1daf646548876fd90e8f489667c46a374ac28cd11dc |

| [src/arggnn/anchor_kernel.py](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/anchor_kernel.py:1) | 2b8cc1c1726f746f961c0d63a4870e806a1a9d3d6e1fa6724dc0bda39967b083 |

| [src/arggnn/data.py](/workspace-SR003.nfs2/ARG-GNN/src/arggnn/data.py:1) | fb4e4b3ad8744731e6d2c5966cc2679f9f2d59dfd079facc280d69fbc61ef654 |

| [scripts/infer_biological_paths_relate_v1.py](/workspace-SR003.nfs2/ARG-GNN/scripts/infer_biological_paths_relate_v1.py:1) | a63a79b36a422f7346f2a35206f41b35fb440053edcfe4e080755c649ed737e9 |

| [scripts/train_ancestralpaths_biological_v1.py](/workspace-SR003.nfs2/ARG-GNN/scripts/train_ancestralpaths_biological_v1.py:1) | 826a1e8c3053cf63b8374361f793903970dd75d44cbcd536755e9aaa3b985056 |
