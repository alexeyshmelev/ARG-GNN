# Exact source excerpts

All paths are relative to `/workspace-SR003.nfs2/ARG-GNN`; source hashes are in `sources.json`. Code is read as text only.

## `.runtime/author_sources/ancestralpaths_77db3cfae100/simulate.py:16–35`

```text
16: with open(args.ages, 'r') as sample_times:
17: 	for line in sample_times:
18: 		line=line.strip()
19: 		field=line.split('\t')
20: 		if field[0]=="Bronze_Age":
21: 			bronze_times.append(int(field[1]))
22: 		if field[0]=="BAA":
23: 			baa_times.append(int(field[1]))
24: 		if field[0]=="Yam":
25: 			yam_times.append(int(field[1]))
26: 		if field[0]=="Neo":
27: 			neo_times.append(int(field[1]))
28: 		if field[0]=="WHG":
29: 			whg_times.append(int(field[1]))
30: 		if field[0]=="EHG":
31: 			ehg_times.append(int(field[1]))
32: 		if field[0]=="CHG":
33: 			chg_times.append(int(field[1]))
34: 		if field[0]=="Ana":
35: 			ana_times.append(int(field[1]))	
```

## `.runtime/author_sources/ancestralpaths_77db3cfae100/simulate.py:38–112`

```text
38: N_bronze = 50000
39: N_Yam = 20000
40: N_baa = 20000
41: N_whg = 2000
42: N_ehg = 3000
43: N_ana = 5000
44: N_neo = 50000
45: N_chg = 3000
46: N_A = 15000 #Ancestor of WHG and EHG
47: N_B = 15000 #Ancestor of CHG and Neolithic farmers
48: N_present = 50000
49: 
50: #Time of events
51: T_bronze = max(bronze_times)+1
52: T_Yam = max(yam_times)+1
53: T_neo = max(neo_times)+1
54: T_baa = max(baa_times)+1
55: T_near_east = 800
56: T_europe = 600
57: T_basal = 1500
58: 
59: demography = msprime.Demography()
60: demography.add_population(name="present_bronze", initial_size=N_present) #0
61: demography.add_population(name="Yam", initial_size=N_Yam) #1
62: demography.add_population(name="Neo", initial_size=N_neo) #2
63: demography.add_population(name="EHG", initial_size=N_ehg) #3
64: demography.add_population(name="WHG", initial_size=N_whg) #4
65: demography.add_population(name="NE", initial_size=N_A) #5
66: demography.add_population(name="BAA", initial_size=N_baa) #6
67: demography.add_population(name="CHG", initial_size=N_chg) #7
68: demography.add_population(name="Ana", initial_size=N_ana) #8
69: demography.add_population(name="WA", initial_size=N_B) #9
70: demography.add_population(name="trunk", initial_size=N_B) #10
71: 
72: demography.add_admixture(time=T_bronze, derived="present_bronze", ancestral=["Neo", "Yam"], proportions=[0.5,0.5])
73: demography.add_admixture(time=T_neo, derived="Neo", ancestral=["WHG", "Ana"], proportions=[0.25,0.75])
74: demography.add_admixture(time=T_baa, derived="BAA", ancestral=["CHG", "Ana"], proportions=[0.25,0.75])
75: demography.add_admixture(time=T_Yam, derived="Yam", ancestral=["CHG", "EHG"], proportions=[0.5,0.5])
76: demography.add_population_split(time=T_europe, derived=["WHG", "EHG"], ancestral="NE")
77: demography.add_population_split(time=T_near_east, derived=["CHG", "Ana"], ancestral="WA")
78: demography.add_population_split(time=T_basal, derived=["WA", "NE"], ancestral="trunk")
79: 
80: demography.add_census(time=350)
81: 
82: present_sample = [msprime.SampleSet(91, time=0, population="present_bronze", ploidy=2)]
83: bronze_samples=[]
84: Baa_samples=[]
85: neolithic_samples=[]
86: Yam_samples=[]
87: CHG_samples=[]
88: Ana_samples=[]
89: WHG_samples=[]
90: EHG_samples=[]
91: for i in range(0,len(bronze_times)):
92: 	time=bronze_times[i]
93: 	bronze_samples.append(msprime.SampleSet(1, population="present_bronze", time=time, ploidy=2))
94: for time in baa_times:
95: 	Baa_samples.append(msprime.SampleSet(1, time=time, population="BAA", ploidy=2))
96: for i in range(0, len(neo_times)):
97: 	time=neo_times[i]
98: 	neolithic_samples.append(msprime.SampleSet(1, time=time, population="Neo", ploidy=2))
99: for i in range(0,len(yam_times)):
100: 	time=yam_times[i]
101: 	Yam_samples.append(msprime.SampleSet(1, time=time, population="Yam", ploidy=2))
102: for i in range(0, len(chg_times)):
103: 	time=chg_times[i]
104: 	CHG_samples.append(msprime.SampleSet(1, time=time, population="CHG", ploidy=2))
105: for time in ana_times:
106: 	Ana_samples.append(msprime.SampleSet(1, time=time, population="Ana", ploidy=2))
107: for time in whg_times:
108: 	WHG_samples.append(msprime.SampleSet(1, time=time, population="WHG", ploidy=2))
109: for time in ehg_times:
110: 	EHG_samples.append(msprime.SampleSet(1, time=time, population="EHG", ploidy=2))
111: 
112: samples = present_sample + bronze_samples + Baa_samples + neolithic_samples + Yam_samples + WHG_samples + EHG_samples + Ana_samples + CHG_samples  #Can multiply each by the number of samples needed for each population
```

## `.runtime/author_sources/ancestralpaths_77db3cfae100/simulate.py:119–145`

```text
119: #Simulate chromosome 3 only
120: tree_sequence_rep = msprime.sim_ancestry(num_replicates=10, recombination_rate=1e-8, demography=demography,samples=samples, sequence_length=2000000)
121: 
122: 
123: for j, tree_sequence in enumerate(tree_sequence_rep):
124: 	ts=msprime.sim_mutations(tree_sequence, rate=1.25e-8)
125: 	name=str(args.out)+"_"+str(j)+".trees"
126: 	ts.dump(name)
127:  #Saving tree.sequence object to desired directory
128: 	with open(str(args.out)+"_"+str(j)+".vcf", "w") as vcf_file:
129: 		ts.write_vcf(vcf_file)
130: 
131: ages=tree_sequence.tables.nodes.time[0:tree_sequence.get_sample_size()]
132: np.savetxt(str(args.out)+".ages", ages)
133: 
134: poplab=open(str(args.out)+".poplabels", 'w')
135: nodes=tree_sequence.tables.nodes
136: pop_names=tree_sequence.tables.populations
137: print("sample", "population","group", "sex", sep=' ', file=poplab)
138: for i in range(int(tree_sequence.get_sample_size()/2)):
139: 	if nodes.population[(i*2)]==0:
140: 		if nodes.time[(i*2)]==0:
141: 			print("tsk_"+str(i), "present_day", "present_day", "NA", sep=' ', file=poplab)
142: 		else:
143: 			print("tsk_"+str(i), "Bronze_Age", "Bronze_Age", "NA", sep=' ', file=poplab)
144: 	else:
145: 		print("tsk_"+str(i), pop_names[nodes.population[(i*2)]].metadata['name'], pop_names[nodes.population[(i*2)]].metadata['name'], "NA", sep=' ', file=poplab)
```

## `.runtime/author_sources/ancestralpaths_77db3cfae100/relate.sh:1–9`

```text
1: output=$1
2: i=$2
3: path_to_relate=$3
4: 
5: ${path_to_relate}/bin/RelateFileFormats --mode ConvertFromVcf --haps ${output}_${i}.haps --sample ${output}_${i}.sample -i ${output}_${i}
6: ${path_to_relate}/bin/RelateFileFormats --mode GenerateSNPAnnotations --haps ${output}_${i}.haps --sample ${output}_${i}.sample --poplabels ${output}.poplabels -o ${output}_${i}
7: ${path_to_relate}/scripts/RelateParallel/RelateParallel.sh -m 1.25e-8 --threads 8 -N 30000 --sample_ages ${output}.ages --haps ${output}_${i}.haps --sample ${output}_${i}.sample --map genetic_map_simulated.txt -o ${output}_relate_${i} --annot ${output}_${i}.annot
8: ${path_to_relate}/scripts/EstimatePopulationSize/EstimatePopulationSize.sh -i ${output}_relate_${i} -m 1.25e-8 -o ${output}_relate_popsize_${i} --threads 8 --num_iter 5 --poplabels ${output}.poplabels
9: ${path_to_relate}/bin/RelateFileFormats --mode ConvertToTreeSequence -i ${output}_relate_popsize_${i} -o ${output}_relate_popsize_${i}
```

## `.runtime/author_sources/ancestralpaths_77db3cfae100/genetic_map_simulated.txt:1–3`

```text
1: pos COMBINED_rate Genetic_Map
2: 0 1 0
3: 2000000 1 2
```

## `.runtime/author_sources/ancestralpaths_77db3cfae100/README.md:19–21`

```text
19: ## Input
20: 
21: Population ages file: To specify the number and ages of each sample to simulate and train over a sample ages file is required. This should be a two column file, tab separated, with the population label and age in units of generations, one line per sample. The population labels should be one of *present_day, Bronze_Age, Neo, Yam, WHG, EHG, CHG, Ana, BAA* (caps-sensitive). These are populations from the model of European population structure. This ![figure](Model_schematic.pdf) describes the relationships of these ancient groups and present day Europeans. Ancient genomes from all of these groups are publically available.
```

## `src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py:28–30`

```text
28: COHORTS = ("present_day", "Bronze_Age", "BAA", "Neo", "Yam", "WHG", "EHG", "Ana", "CHG")
29: COUNTS = (91, 81, 7, 151, 13, 51, 43, 25, 14)
30: CLASSES = {"Ana": 0, "CHG": 1, "WHG": 2, "EHG": 3}
```

## `src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py:62–89`

```text
62: def roster():
63:     """Author order and original within-cohort age order; GBR input rows ignored."""
64:     ages = {}
65:     source = confined(AUTHOR + "/example/mesoneo_ages_pop.txt")
66:     if sha(source) != "3cadb16abbc0a1913eadb183d324eadc26c035d61e5116cd58675f52a942b7de":
67:         raise ValueError("Author age input changed")
68:     for line_number, line in enumerate(source.read_text().splitlines(), 1):
69:         cohort, age = line.split()
70:         ages.setdefault(cohort, []).append((int(age), line_number))
71:     rows = []
72:     diploid = 0
73:     for cohort, count in zip(COHORTS, COUNTS):
74:         dates = [(0, None)] * 91 if cohort == "present_day" else ages[cohort]
75:         if len(dates) != count:
76:             raise ValueError("Full author cohort count changed: " + cohort)
77:         population = "present_bronze" if cohort in ("present_day", "Bronze_Age") else cohort
78:         for age, line_number in dates:
79:             role = "pure_source_reference" if cohort in CLASSES else "query"
80:             for haplotype in (0, 1):
81:                 rows.append({"sample_index": len(rows), "diploid_index": diploid,
82:                              "sample_id": f"tsk_{diploid}", "haplotype": haplotype,
83:                              "observed_cohort": cohort, "sampling_population": population,
84:                              "age_generations": age, "role": role,
85:                              "observed_reference_class": CLASSES.get(cohort),
86:                              "author_age_line_1based": line_number})
87:             diploid += 1
88:     if len(rows) != 952 or sum(r["role"] == "query" for r in rows) != 686:
89:         raise ValueError("Native full-roster arithmetic differs")
```

## `src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py:183–220`

```text
183:         "runtime": {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
184:                     "msprime": msprime.__version__, "tskit": tskit.__version__, "numpy": np.__version__},
185:         "output_root": "data/ancestralpaths_full_panel_discovery_v1",
186:         "output_cap_bytes": 4000000000, "native_step_headroom_bytes": 1000000000,
187:         "maximum_parents_per_invocation": 1, "default_parent_id": "ap_full_train_000",
188:         "root_seed": 101, "splits": {"train": 3, "discovery_holdout": 2, "val": 0, "test": 0},
189:         "parent_registry": parents(),
190:         "seed_scheme": "SHA256 ASCII arggnn|ancestralpaths_full_panel_discovery_v1|root=101|split={split}|parent={000..}|purpose={purpose};first8bytes big%(2**32-1)+1 for simulation;%(2**31-100)+1 for Relate;size uses base+1..5 and final reestimate reuses base+5",
191:         "roster": {"cohort_order": list(COHORTS), "diploid_counts": dict(zip(COHORTS, COUNTS)),
192:                    "diploids": 476, "haplotypes": 952, "query_haplotypes": 686,
193:                    "pure_reference_haplotypes": 266, "sample_template_sha256": digest(sample_rows),
194:                    "within_cohort_age_order": "original_age_file_order",
195:                    "present_day": "hardcoded91_diploids_age0_author_GBR_rows_ignored",
196:                    "sample_identity": ["parent_id", "sample_id", "haplotype"]},
197:         "demography": simulation.build_demography().asdict(),
198:         "simulation": {"length_bp": 2000000, "recombination_rate": 1e-8, "mutation_rate": 1.25e-8,
199:                        "ancestry_model": "hudson", "mutation_model": "JC69",
200:                        "discrete_genome": True, "record_migrations": False, "census_time": 350,
201:                        "mutation_on_census_stripped_genealogy": True, "score_bin_bp": 100,
202:                        "cleaned_oracle_exported": False},
203:         "relate": {"native_tool_release": "1.2.4",
204:                    "installation": ref(".runtime/tools/relate/installation.json"),
205:                    "source_coordinate_reference": native["source_coordinate_reference"],
206:                    "tool_sha256": native["tool_sha256"], "initial_effective_haploid_size": 30000,
207:                    "mutation_rate": 1.25e-8, "initial_threads": 8, "size_threads": 8,
208:                    "chunk_memory_gb": 2, "population_size_iterations": 5,
209:                    "size_tree_mutation_threshold": 0.5, "years_per_generation": 28,
210:                    "population_size_plotting": False, "mutation_rate_normalization": False,
211:                    "final_branch_length_reestimate": True, "timeout_seconds_per_step": 7200,
212:                    "primary_condition": "post_five_size_iterations_primary"},
213:         "observation_contract": {"Relate_input_whitelist": list(OBSERVED_FILES),
214:                                  "truth_or_oracle_reads_in_Relate": False,
215:                                  "ancestry_labels_in_observed_files": False,
216:                                  "native_sample_ages_preserved_in_generations": True,
217:                                  "reference_classes": CLASSES, "query_reference_class": None,
218:                                  "VCF_position": "integer_zero_based_mutation_position_plus1",
219:                                  "contig": "22", "build": "synthetic_no_empirical_GRCh37_claim",
220:                                  "native_inferred_sample_IDs": "actual_sample_flags_in_verified_haps_order",
```

## `src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py:323–353`

```text
323: def generate_parent(config, parent, output, guard):
324:     import msprime
325:     import numpy as np
326:     import tskit
327:     simulation, _ = helpers()
328:     rows = roster()
329:     # Same source grouping: one91-diploid present SampleSet, then ancient diploids.
330:     samples = [msprime.SampleSet(91, ploidy=2, population="present_bronze", time=0)]
331:     samples += [msprime.SampleSet(1, ploidy=2, population=rows[i]["sampling_population"],
332:                                  time=rows[i]["age_generations"]) for i in range(182, 952, 2)]
333:     guard(config["native_step_headroom_bytes"])
334:     truth = msprime.sim_ancestry(samples=samples, demography=simulation.build_demography(),
335:                                 sequence_length=2000000, recombination_rate=1e-8,
336:                                 model="hudson", discrete_genome=True, record_migrations=False,
337:                                 random_seed=parent["seeds"]["ancestry"])
338:     if list(truth.samples()) != list(range(952)):
339:         raise ValueError("Native952 sample order changed")
340:     for row in rows:
341:         node = truth.node(row["sample_index"])
342:         if node.time != row["age_generations"] or simulation.POPULATIONS[node.population] != row["sampling_population"]:
343:             raise ValueError("Simulated date/population order differs")
344:     for i in range(0, 952, 2):
345:         if truth.node(i).individual == tskit.NULL or truth.node(i).individual != truth.node(i + 1).individual:
346:             raise ValueError("Diploid pairing changed")
347:     targets = simulation.external_truth_intervals(truth, rows)
348:     if targets["native_paths_1based"].shape != (truth.num_trees, 686):
349:         raise ValueError("Full native truth shape differs")
350:     clean, mapping = simulation.strip_truth_annotations(truth)
351:     observed = msprime.sim_mutations(clean, rate=1.25e-8, model=msprime.JC69(),
352:                                     discrete_genome=True, record_provenance=False,
353:                                     random_seed=parent["seeds"]["mutations"])
```

## `src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py:378–401`

```text
378:         "truth_interval_boundaries_not_permitted_as_inference_windows": True}, guard)
379:     edges = np.arange(0, 2000001, 100, dtype=np.float64)
380:     npz_write(obs / "scoring_coordinates.npz",
381:               {"bin_edges_bp": edges, "positions_bp": (edges[:-1] + edges[1:]) / 2}, guard)
382:     json_write(obs / "sample_manifest.json", manifest, guard)
383:     names = [r["sample_id"] for r in manifest[::2]]
384:     text_write(obs / "sample_order.txt", "\n".join(names) + "\n", guard)
385:     text_write(obs / "sample_ages.generations", "\n".join(str(r["age_generations"]) for r in manifest) + "\n", guard)
386:     text_write(obs / "population_labels.txt", "sample population group sex\n" + "".join(
387:         f'{r["sample_id"]} {r["observed_cohort"]} {r["observed_cohort"]} NA\n' for r in manifest[::2]), guard)
388:     guard(config["native_step_headroom_bytes"])
389:     with (obs / "genotypes.vcf").open("x") as stream:
390:         observed.write_vcf(stream, ploidy=2, individual_names=names, contig_id="22",
391:                            position_transform=lambda p: np.asarray(p, dtype=np.int64) + 1)
392:     guard(0)
393:     positions = observed.tables.sites.position
394:     if not np.array_equal(positions, positions.astype(np.int64)):
395:         raise ValueError("Mutation SNP coordinates are not discrete")
396:     json_write(obs / "site_coordinate_manifest.json", {
397:         "build": "synthetic", "contig_id": "22", "length_bp": 2000000,
398:         "rows": [{"site_id": i, "tree_position_0based": int(p), "VCF_position_1based": int(p) + 1}
399:                  for i, p in enumerate(positions)]}, guard)
400:     text_write(obs / "uniform_recombination_map.txt",
401:                "Position(bp)\tRate(cM/Mb)\tMap(cM)\n1\t1\t0\n2000001\t1\t2\n", guard)
```

## `src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py:563–575`

```text
563:     run(["bash", "-x", native["scripts/RelateParallel/RelateParallel.sh"], "-m", "1.25e-8",
564:          "-N", "30000", "--threads", "8", "--memory", "2",
565:          "--seed", str(parent["seeds"]["relate_initial"]), "--haps", str(inputs / "input.haps"),
566:          "--sample", str(inputs / "input.sample"), "--sample_ages", str(inputs / "sample_ages.txt"),
567:          "--map", str(inputs / "input.map"), "--annot", str(inputs / "input.annot"),
568:          "-o", "pre_size"], "relate_initial")
569:     pre_refs = {name: ref(str((work / name).relative_to(ROOT))) for name in ("pre_size.anc", "pre_size.mut")}
570:     run(["bash", "-x", native["scripts/EstimatePopulationSize/EstimatePopulationSize.sh"],
571:          "-i", "pre_size", "-o", "post_size", "-m", "1.25e-8", "--threads", "8",
572:          "--num_iter", "5", "--threshold", "0.5", "--years_per_gen", "28",
573:          "--seed", str(parent["seeds"]["relate_size_base"]), "--poplabels", str(inputs / "input.poplabels"),
574:          "--noplot"], "five_population_size_iterations")
575:     run([f, "--mode", "ConvertToTreeSequence", "-i", "post_size", "-o", "post_size"], "convert_post_size")
```

## `src/arggnn/prepare_expanded_full_panel_v1.py:23–24`

```text
23: SPLITS = (("train", 32), ("val", 8), ("outer", 8))
24: PURPOSES = ("ancestry", "mutations", "relate_initial", "relate_size_base")
```

## `src/arggnn/prepare_expanded_full_panel_v1.py:53–69`

```text
53: def parent_registry_v1():
54:     rows = []
55:     for split, count in SPLITS:
56:         for index in range(count):
57:             seeds, domains = {}, {}
58:             for purpose in PURPOSES:
59:                 domain = f"arggnn|expanded_full_panel_v1|root=101|split={split}|parent={index:03d}|purpose={purpose}"
60:                 digest = hashlib.sha256(domain.encode("ascii")).digest()
61:                 modulus = 2**31 - 100 if purpose.startswith("relate") else 2**32 - 1
62:                 seeds[purpose] = int.from_bytes(digest[:8], "big") % modulus + 1
63:                 domains[purpose] = digest.hex()
64:             rows.append(dict(parent_id=f"ap_expanded_{split}_{index:03d}", split=split,
65:                 downstream_role={"train": "TRAIN", "val": "INTERNAL_VAL", "outer": "UNTOUCHED_OUTER"}[split],
66:                 replicate=index, seeds=seeds, seed_domain_sha256=domains,
67:                 relate_size_iteration_seeds=[seeds["relate_size_base"] + offset for offset in range(1, 6)],
68:                 relate_final_reestimate_seed=seeds["relate_size_base"] + 5,
69:                 evaluation_authorized=False if split == "outer" else None))
```

## `src/arggnn/prepare_expanded_full_panel_v1.py:77–94`

```text
77: def build_config_v1():
78:     """Original biology, age roster and native recipe; new registry/splits only."""
79:     config = deepcopy(original.default_config())
80:     prior, metadata_refs = _prior_streams_v1()
81:     registry = parent_registry_v1()
82:     config.update(kind=KIND, status="source_only_registry_frozen_no_generation", output_root="data/expanded_full_panel_v1",
83:         default_parent_id="ap_expanded_train_000", splits=dict(SPLITS), parent_registry=registry,
84:         seed_scheme="SHA256 ASCII arggnn|expanded_full_panel_v1|root=101|split={train,val,outer}|parent={000..}|purpose={purpose};first8 big-endian modulo original simulation/Relate ranges;sizebase+1..5;finalreestimate reuses+5",
85:         expansion=dict(source_adapter=original.ref(SOURCE), reused_producer=original.ref(original.SOURCE),
86:             original_recipe_config=original.ref(original.CONFIG), prior_seed_metadata=metadata_refs,
87:             parent_registry_sha256=original.digest(registry), previously_registered_unique_seed_values=len(prior),
88:             new_unique_seed_values=48*9, stream_collision_count=0,
89:             generated_parent_count=0, existing_parents_reused=False,
90:             changed_conditions=["independent_parent_registry", "split_counts", "output_root"],
91:             unchanged_conditions=["476diploid_author_roster", "all_original_sampling_ages", "author_demography", "2Mb_JC69_rates", "complete686query_C4_targets", "952observed_haplotypes", "Relate1.2.4_initial_and5size_iterations"],
92:             target_policy="TRAINloss;INTERNAL_VALselection_afterpredictionseal;8outer_reserved_no_evaluation_or_tuning",
93:             inherited_native_sampler_metadata="retained_original3sequence_stride_receipt;H038wholeforestdoesnotusefocalsampler",
94:             full_panel_model_preparation_included=False))
```

## `scripts/infer_biological_paths_relate_v1.py:118–162`

```text
118: def filter_vcf(lines, names, position_join):
119:     """Observed-only biallelic segregating nucleotide/MAC1 filtering."""
120:     header, kept, counts = [], [], {"input": 0, "multiallelic_or_nonnucleotide": 0, "monomorphic": 0}
121:     saw_header = False
122:     previous = 0
123:     for line in lines:
124:         if line.startswith("#"):
125:             if line.startswith("#CHROM"):
126:                 if line.rstrip().split("\t")[9:] != names:
127:                     raise ValueError("original VCF diploid order differs")
128:                 saw_header = True
129:             header.append(line.rstrip())
130:             continue
131:         if not saw_header:
132:             raise ValueError("VCF has no ordered sample header")
133:         fields = line.rstrip().split("\t")
134:         if len(fields) != 9 + len(names):
135:             raise ValueError("VCF sample columns differ")
136:         counts["input"] += 1
137:         pos = int(fields[1])
138:         if fields[0] != "22" or pos <= previous or pos not in position_join:
139:             raise ValueError("VCF/SNP coordinate join or chromosome differs")
140:         previous = pos
141:         if fields[3] not in "ACGT" or len(fields[3]) != 1 or fields[4] not in "ACGT" or len(fields[4]) != 1 or fields[3] == fields[4]:
142:             counts["multiallelic_or_nonnucleotide"] += 1
143:             continue
144:         fmt = fields[8].split(":")
145:         if "GT" not in fmt:
146:             raise ValueError("VCF lacks genotype field")
147:         gt_index = fmt.index("GT")
148:         genotypes = []
149:         for value in fields[9:]:
150:             gt = value.split(":")[gt_index]
151:             if gt not in ("0|0", "0|1", "1|0", "1|1"):
152:                 raise ValueError("unphased/missing/invalid observed genotype")
153:             genotypes.extend((int(gt[0]), int(gt[2])))
154:         if not 0 < sum(genotypes) < len(genotypes):
155:             counts["monomorphic"] += 1
156:             continue
157:         kept.append({"fields": fields, "pos1": pos, "genotypes": genotypes,
158:                      "original_site_id": position_join[pos]["site_id"]})
159:     if not kept:
160:         raise ValueError("no eligible observed SNPs")
161:     counts["eligible"] = len(kept)
162:     return header, kept, counts
```

## `scripts/infer_biological_paths_relate_v1.py:280–340`

```text
280:     """Convert retained VCF coordinates to clipped original physical support."""
281:     inferred = tskit.load(native_file)
282:     mapping = inferred_sample_mapping(inferred, assets)
283:     site_positions = inferred.tables.sites.position
284:     if not np.array_equal(site_positions, site_positions.astype(np.int64)):
285:         raise ValueError("converter SNP positions are not exact integers")
286:     actual = site_positions.astype(np.int64).tolist()
287:     if not actual or any(p not in retained_vcf for p in actual) or len(set(actual)) != len(actual):
288:         raise ValueError("converted SNPs cannot join observed haps")
289:     left, right = float(min(actual) - 1), float(max(actual))
290:     if not 0 <= left < right <= 2000000:
291:         raise ValueError("converted physical coverage invalid")
292:     tables = inferred.dump_tables()
293:     edge = tables.edges
294:     shifted_left = np.maximum(edge.left - 1, left)
295:     shifted_right = np.minimum(edge.right - 1, right)
296:     keep = shifted_right > shifted_left
297:     parent, child = edge.parent[keep], edge.child[keep]
298:     tables.edges.clear()
299:     tables.edges.set_columns(left=shifted_left[keep] - left, right=shifted_right[keep] - left,
300:                              parent=parent, child=child)
301:     tables.sites.position = site_positions - 1 - left
302:     tables.sequence_length = right - left
303:     tables.nodes.population = np.full(inferred.num_nodes, tskit.NULL, dtype=np.int32)
304:     tables.nodes.individual = np.full(inferred.num_nodes, tskit.NULL, dtype=np.int32)
305:     tables.nodes.flags = tables.nodes.flags & tskit.NODE_IS_SAMPLE
306:     for table in (tables.nodes, tables.edges, tables.sites, tables.mutations):
307:         table.drop_metadata()
308:     tables.migrations.clear()
309:     tables.populations.clear()
310:     tables.individuals.clear()
311:     tables.provenances.clear()
312:     tables.metadata_schema = tskit.MetadataSchema(None)
313:     tables.metadata = b""
314:     tables.sort()
315:     clean = tables.tree_sequence()
316:     if inferred_sample_mapping(clean, assets) != mapping or clean.num_nodes != inferred.num_nodes:
317:         raise ValueError("clipping changed native inferred node IDs/order")
318:     clean.dump(destination)
319:     positions = np.arange(50, 2000000, 100, dtype=np.float64)
320:     retained_grid = np.flatnonzero((positions >= left) & (positions < right))
321:     query_rows = [r for r in assets["rows"] if r["role"] == "query"]
322:     result = {
323:         "condition": condition, "native_tree_sha256": sha(native_file), "clean_tree_sha256": sha(destination),
324:         "native_tree": str(native_file), "clean_tree": str(destination),
325:         "coverage_original_bp0": [left, right], "physical_coverage_fraction": (right-left)/2000000,
326:         "local_to_original_offset_bp": left, "sequence_length_local_bp": clean.sequence_length,
327:         "original_parent_interval_bp0": [0, 2000000], "normalization_length_bp": 2000000,
328:         "physical_coordinate_transform": "physical_original = local_tree_position + coverage_left",
329:         "sample_index_to_inferred_tree_node_id": mapping,
330:         "mapping_basis": "actual NODE_IS_SAMPLE rows in converter insertion order; verified VCF diploid header, .sample order, phased .haps columns, and observed ages",
331:         "node_ids_preserved_during_clipping": True,
332:         "converter_VCF_to_original_offset_bp": -1,
333:         "retained_fixed100bp_indexes": retained_grid.tolist(), "fixed_grid_total": 20000,
334:         "fixed_grid_retained": len(retained_grid), "samples": 280,
335:         "sample_age_order_preserved": True, "sites": clean.num_sites, "trees": clean.num_trees,
336:         "query_keys": [[r["sample_id"], r["haplotype"]] for r in query_rows],
337:         "source_class_labels": "observed pure-reference labels only; no target loaded",
338:         "site_join": [{"native_site_id": i, "VCF_position_1based": p,
339:                       "original_position_0based": p-1, "local_position_0based": p-1-left,
340:                       "original_site_id": assets["position_join"][p]["site_id"]}
```

## `.runtime/author_sources/argmix_c2481248eb0d/sim/README.md:1–11`

```text
1: # Forward Snakemake workflow
2: 
3: This directory is the backend for **`argmix build-data`**. The CLI syncs a derived
4: `outputs/_work/forward_config.runtime.yaml` and runs Snakemake here with that
5: `--configfile`. Your project’s `forward_config.yaml` is never rewritten.
6: 
7: Do not invoke `Snakefile` directly. Use:
8: 
9: ```bash
10: pixi run argmix build-data --project <dir>
11: ```
```

## `.runtime/author_sources/argmix_c2481248eb0d/sim/scripts/forward_sim_multievent.py:1–6`

```text
1: #!/usr/bin/env python3
2: """Forward-in-time multi-event admixture simulator from real reference haplotypes.
3: 
4: This script intentionally emits only the configured present population into the
5: final VCF. Intermediate simulated populations are kept internal for forward
6: simulation and optional ground-truth export.
```

## `.runtime/author_sources/argmix_c2481248eb0d/sim/scripts/forward_sim_multievent.py:22–29`

```text
22: def parse_args() -> argparse.Namespace:
23:     p = argparse.ArgumentParser(description="Forward simulate present-day from multi-event admixture")
24:     p.add_argument("--reference-vcf", required=True, help="VCF.gz containing only real reference samples")
25:     p.add_argument("--metadata", required=True, help="Metadata CSV with IID, Final_Reference")
26:     p.add_argument("--events", required=True, help="YAML with ordered admixture events")
27:     p.add_argument("--recomb-map", required=True, help="Map file with position + cumulative cM")
28:     p.add_argument("--sim-id-map", required=True, help="TSV with columns: sim_id, original_iid")
29:     p.add_argument("--n-present-sim", type=int, required=True)
```

## `.runtime/author_sources/argmix_c2481248eb0d/sim/scripts/forward_sim_multievent.py:424–428`

```text
424: def main() -> None:
425:     args = parse_args()
426: 
427:     rng = np.random.default_rng(args.seed + args.rep)
428:     events = load_events(Path(args.events))
```

## `.runtime/author_sources/argmix_c2481248eb0d/sim/scripts/build_sample_manifests.py:14–17`

```text
14: def parse_args() -> argparse.Namespace:
15:     p = argparse.ArgumentParser(description="Build manifest files for forward simulation")
16:     p.add_argument("--metadata", required=True, help="CSV with IID, Final_Reference, Generations")
17:     p.add_argument("--batch-ids", required=True, help="Batch sample IDs file (one IID per line)")
```

## `.runtime/author_sources/argmix_c2481248eb0d/sim/scripts/build_sample_manifests.py:130–137`

```text
130:     args = parse_args()
131: 
132:     metadata = pd.read_csv(args.metadata)
133:     ensure_required_columns(metadata, ["IID", "Final_Reference", "Generations"])
134: 
135:     metadata["IID"] = metadata["IID"].map(normalize_id)
136:     metadata["Final_Reference"] = metadata["Final_Reference"].map(lambda x: str(x).strip())
137:     metadata["Generations"] = metadata["Generations"].astype(float)
```

## `.runtime/author_sources/argmix_c2481248eb0d/sim/scripts/build_sample_manifests.py:341–354`

```text
341:     for iid in arg_reference_samples:
342:         row = iid_to_row[iid]
343:         age_rows.append({"sample_id": iid, "age_generations": float(row.Generations)})
344:         pop_rows.append(
345:             {
346:                 "sample": iid,
347:                 "population": str(row.Final_Reference),
348:                 "group": str(row.Final_Reference),
349:                 "sex": "NA",
350:             }
351:         )
352: 
353:     for sim_id in sim_ids:
354:         age_rows.append({"sample_id": sim_id, "age_generations": 0.0})
```

## `.runtime/tools/relate/relate_v1.2.4_x86_64_static/scripts/EstimatePopulationSize/EstimatePopulationSize.sh:19–29`

```text
19:   echo "--pop_of_interest: Optional: Specify the populations for which you want to estimate the population size as a comma separated string. Default is all populations."
20:   echo "--threshold:       Optional: Used to delete trees with less than specified number of mutations. Use 0 to use all trees. Default: Number of haplotypes."
21:   echo "--years_per_gen :  Optional: Years per generation. Default is 28."
22:   echo "--bins:            Optional: Specify epoch bins. Format: lower, upper, stepsize for function c(0,10^seq(lower, upper, stepsize))."
23:   echo "--num_iter:        Optional: Number of iterations of the algorithm. Default: 10."
24:   echo "--chr:             Optional: File listing chromosome IDs (one per line). Overrides --first_chr, --last_chr. Assumes input files are indexed by chr, e.g., example_chr1.anc, example_chr1.mut, etc. Specify -i example in this case."
25:   echo "--first_chr:       Optional: Index of first chr. Assumes that input files are indexed by chr, e.g. example_chr1.anc, example_chr1.mut, etc. Specify -i example in this case. "
26:   echo "--last_chr:        Optional: Index of last chr."
27:   echo "--norm_mutrate     Optional: Normalise coal rates by mutation rate. Recommended if SNPs are MAF ascertained."
28:   echo "--threads:         Optional. Maximum number of threads."
29:   echo "--seed:            Optional: Random seed for branch lengths estimation"
```

## `.runtime/tools/relate/relate_v1.2.4_x86_64_static/scripts/EstimatePopulationSize/EstimatePopulationSize.sh:46–49`

```text
46: 
47: maxjobs=1
48: years_per_gen=28
49: 
```

## `.runtime/tools/relate/relate_v1.2.4_x86_64_static/scripts/EstimatePopulationSize/EstimatePopulationSize.sh:738–746`

```text
738:     for i in `seq 1 1 ${num_iter}`
739:     do
740: 
741:       if [ ! -z "${norm_mutrate-}" ];
742:       then
743:         args=()
744:         args+=( "--mode Avg" )
745:         args+=( "--dist ${output}.dist" )
746:         args+=( "--years_per_gen ${years_per_gen}" )
```

## `.runtime/tools/relate/relate_v1.2.4_x86_64_static/scripts/EstimatePopulationSize/EstimatePopulationSize.sh:860–893`

```text
860:     args+=( "--years_per_gen ${years_per_gen}" )
861:     args+=( "-i ${output}" )
862:     args+=( "-o ${output}" )
863:     [[ ! -z "${bins-}" ]] && args+=( "--bins $bins" )
864:     ${PATH_TO_RELATE}/bin/RelateMutationRate ${args[@]}
865: 
866: 		if [ -z ${noanc-} ]
867: 		then
868: 
869: 			#estimate mutation rate and coalescent rates for a final time
870: 			${PATH_TO_RELATE}/bin/RelateExtract \
871: 				--mode DivideAncMut \
872: 				--threads $maxjobs \
873: 				--anc ${filename}.anc \
874: 				--mut ${filename}.mut \
875: 				-o ${output}_tmp
876: 
877: 			first_chunk=0
878: 			last_chunk=$(ls ${output}_tmp_chr*.mut.gz | wc -l)
879: 			last_chunk=$((${last_chunk} - 1)) 
880: 
881: 			ReEstimateBranchLengths (){
882: 
883: 				chunk=$1
884: 
885:         args=()
886: 
887:         args+=( "--mode ReEstimateBranchLengths" )
888:         args+=( "--coal ${output}.coal" )
889:         args+=( "--dist ${output}.dist" )
890:         args+=( "-m ${mu}" )
891:         args+=( "-i ${output}_tmp_chr${chunk}" )
892:         args+=( "-o ${output}_tmp_chr${chunk}" )
893:         [[ ! -z "${seed-}" ]] && args+=( "--seed $(($seed + $i))" )
```

## `configs/expanded_full_panel_v1.json:1758–1790`

```text
1758:   "relate": {
1759:     "chunk_memory_gb": 2,
1760:     "final_branch_length_reestimate": true,
1761:     "initial_effective_haploid_size": 30000,
1762:     "initial_threads": 8,
1763:     "installation": {
1764:       "path": ".runtime/tools/relate/installation.json",
1765:       "sha256": "c7cf58b9b366bc7a90e1320b17be781a0ad257c9b6cbe464b41451888e391b94"
1766:     },
1767:     "mutation_rate": 1.25e-08,
1768:     "mutation_rate_normalization": false,
1769:     "native_tool_release": "1.2.4",
1770:     "population_size_iterations": 5,
1771:     "population_size_plotting": false,
1772:     "primary_condition": "post_five_size_iterations_primary",
1773:     "size_threads": 8,
1774:     "size_tree_mutation_threshold": 0.5,
1775:     "source_coordinate_reference": {
1776:       "commit": "b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74",
1777:       "file": "include/file_formats/ConvertToTreeSequence.cpp",
1778:       "sha256": "452298c9a5ad60159d9f86dd402f71798e8a88108f2bb217fcb941d28778bfc4"
1779:     },
1780:     "timeout_seconds_per_step": 7200,
1781:     "tool_sha256": {
1782:       "bin/Relate": "71ce82f469dd99749a3eb1847e0704be8785a26be94eae2cb8e1ed80881c7ce8",
1783:       "bin/RelateCoalescentRate": "a4ba4f2971e6669d191327740bcee990710f0b880a7887f88425c140a9322720",
1784:       "bin/RelateExtract": "58845b350623102dfc7c85e5a5cb17feb268a6e67aa28240db8751dd01ed0430",
1785:       "bin/RelateFileFormats": "cb19cb6e277c27704de506f0e8155f8f53d9c3dc8d0f52615a059caa253332ee",
1786:       "bin/RelateMutationRate": "4e38c9a09051b1f22b47293ecfdea2369b6d5e3547fc03b3cb9159a127c142d2",
1787:       "scripts/EstimatePopulationSize/EstimatePopulationSize.sh": "5fcc0e5f80dd4d67255d56b457fd7ce8d5ccd0aa313fc22b6baf62e32d87c96b",
1788:       "scripts/RelateParallel/RelateParallel.sh": "a4558aaffbcb73899bc5a3d87fbb253c80d06160995e10c40e013a7122644315"
1789:     },
1790:     "years_per_generation": 28
```

## `configs/expanded_full_panel_v1.json:1793–1834`

```text
1793:   "roster": {
1794:     "cohort_order": [
1795:       "present_day",
1796:       "Bronze_Age",
1797:       "BAA",
1798:       "Neo",
1799:       "Yam",
1800:       "WHG",
1801:       "EHG",
1802:       "Ana",
1803:       "CHG"
1804:     ],
1805:     "diploid_counts": {
1806:       "Ana": 25,
1807:       "BAA": 7,
1808:       "Bronze_Age": 81,
1809:       "CHG": 14,
1810:       "EHG": 43,
1811:       "Neo": 151,
1812:       "WHG": 51,
1813:       "Yam": 13,
1814:       "present_day": 91
1815:     },
1816:     "diploids": 476,
1817:     "haplotypes": 952,
1818:     "present_day": "hardcoded91_diploids_age0_author_GBR_rows_ignored",
1819:     "pure_reference_haplotypes": 266,
1820:     "query_haplotypes": 686,
1821:     "sample_identity": [
1822:       "parent_id",
1823:       "sample_id",
1824:       "haplotype"
1825:     ],
1826:     "sample_template_sha256": "19fc09e3c7186db6ce95fabeea4efadb535e668c505106aba8bcfc6fcc269ceb",
1827:     "within_cohort_age_order": "original_age_file_order"
1828:   },
1829:   "runtime": {
1830:     "msprime": "1.3.4",
1831:     "numpy": "2.4.6",
1832:     "python": "3.11",
1833:     "tskit": "1.0.3"
1834:   },
```

## `configs/expanded_full_panel_v1.json:1956–1968`

```text
1956:   "seed_scheme": "SHA256 ASCII arggnn|expanded_full_panel_v1|root=101|split={train,val,outer}|parent={000..}|purpose={purpose};first8 big-endian modulo original simulation/Relate ranges;sizebase+1..5;finalreestimate reuses+5",
1957:   "simulation": {
1958:     "ancestry_model": "hudson",
1959:     "census_time": 350,
1960:     "cleaned_oracle_exported": false,
1961:     "discrete_genome": true,
1962:     "length_bp": 2000000,
1963:     "mutation_model": "JC69",
1964:     "mutation_on_census_stripped_genealogy": true,
1965:     "mutation_rate": 1.25e-08,
1966:     "recombination_rate": 1e-08,
1967:     "record_migrations": false,
1968:     "score_bin_bp": 100
```

## `configs/expanded_full_panel_v1.json:2005–2009`

```text
2005:   "splits": {
2006:     "outer": 8,
2007:     "train": 32,
2008:     "val": 8
2009:   },
```

## AP supplement printed page25,PDF page26

```text
diverged from the true ancestral populations and so act as better representatives that existed
before one or many admixture and split events.


Constructing a path model of European population history

Using the extensive amount of previous work elucidating the genetic history of Europeans,
we put forward a standardised model of European population structure. Figure S7 shows a
schematic of this demographic model that describes the population structure in Europe
during the last 50,000 years. Shortly after the expansion of anatomically modern humans into
Eurasia, there is a population split 15,000 years ago (1500 generations ago) between the
Northern Europeans (NE), who continued moving northwest into Europe, and West Asians
(WA) who stayed more locally in the Levant and South Caucasus area. The WA population
then splits to form the Caucasus hunter gatherers/early Iranians (CHG) and the Anatolians
(Ana) ~24,000 years ago (800 generations ago). Within the NE, ~18,000 years ago (600
generations ago), the Western hunter gatherers (WHG) and Eastern hunter gatherers (EHG)
diverged. At that point, the four separate populations that make up present day European
ancestry are distinct in the model. The timing of divergence between the four distinct groups
appears to correspond to the onset of the Last Glacial Maximum [17, 19], during which time
these populations became isolated from each other in refugia.

The model subsequently describes how admixture between these populations leads to the
modern European gene pool. At the start of the Neolithic around 9,000 years ago, farmers
from Anatolian moved into Europe. These incoming Anatolian farmers admixed with the local
WHG ~7,800 years ago (259 generations ago) to form the Neolithic farmer population [20,
21] and their ancestry reaches Britain by 6,000 years ago [22].

During the late Neolithic, steppe populations, characterised by the Yamnaya (Yam) culture,
appear as a mix of EHG and CHG/Iranian ancestries [17, 19, 23]“-5,300 years ago (177
generations ago). At a similar time, the Bronze Age Anatolian group (BAA) represents a later
Anatolian population formed from an admixture of CHGs with Ana −5,100 years ago (170
generations ago). Around the start of the Bronze Age −4,900 years ago (166 generations
ago), migrants with Yam ancestry moved west into Europe and had a profound impact on the
genetic landscape by admixing with the Neolithic farmers to form the European Bronze Age
population, leading to present day Europeans [16, 20, 23–25].




                                                                                            25

```

## AP supplement printed page27,PDF page28

```text
Therefore, we propose a model of four ancestral streams that lead to present-day
Europeans.

Six different paths that haplotypes can take from any sampled individual are shown in
different colours in Figure S7. Path 1 = red, starts from the present day Europeans, going
back through Neolithic farmers, Anatolian farmers, West Asians to the root. Path 2 = purple,
starts at present day Europeans, going back through the Yamnaya, Caucasus hunter
gatherers, then West Asians to the root. Path 3 = black, starts with present day Europeans,
going back through Neolithic farmers to Western hunter gatherers and then through Northern
Europeans to the root. Path 4 = orange, starts at present day Europeans, going back through
the Yamnaya to Eastern hunter gatherers and then through Northern Europeans to the root.
Paths 5 = blue, starting in the Bronze Age Anatolians and joins path 1 part. Path 6 = cyan,
starts in Bronze Age Anatolians and joins path 2. When paths overlap, lineages from all
overlapping paths can coalesce.

We constructed our model of European population history in msprime [18], a coalescent
simulator. The population sizes and admixture fractions are shown in the schematic in Figure
S7. This model has been submitted to the stdpopsim catalogue (https://github.com/popsim-
consortium/stdpopsim/blob/main/stdpopsim/catalog/HomSap/demographic_models.py) and
is publicly available for users to simulate data from.


Comparison of simulated data to MesoNeo genomes

The MesoNeo dataset is a large collection of both ancient and present day samples. While
the dataset contains samples from all over the world, most are from people that lived in the
last 10,000 years in Eurasia. The dataset is composed of 1,490 genomes that have been
imputed to 3.7 million SNPs once filtered for coverage (>0.1X), low imputation quality, close
relatives and >0.5 imputation INFO score (See [27] for data availability).

To test how well our model represents the true history of Europeans, we compared data
simulated from the model to the MesoNeo genomes. Based on PCA (Figure S26) we subset
the 1,492 genomes from the MesoNeo dataset to those that were tightly clustered in the
groups relevant to the genetic history of Europeans, so that only samples diagnostic of each
population in our model were kept.

For our analysis of the time since admixture (see below), we chose 476 diploid individuals
(952 haploid) including 91 GBR 1000 Genomes samples. During simulation, samples were
                                                                                              27

```

## AP supplement printed page28,PDF page29

```text
taken from populations and times to match each real diploid sample, using their radiocarbon
or context dates. We aimed to create a simulated dataset that closely matched our MesoNeo
subset genetically.

For our time-series selection analysis (Supplementary Note 4), we used an expanded set of
1,015 ancient genomes relevant to modern European genetic history. This included samples
that have a West Eurasian archaeological location and lying on EHG-WHG-CHG-Farmer
clines in Principle Component Analysis and excluded very old West Eurasian samples and
archaics. These were merged with all 503 present-day European 1000 Genomes Project
samples to make up our dataset, totalling 1518 diploid samples.

Upon simulation, data for sample individuals is produced, in the form of both VCF files and
tree sequence files. We simulated chromosomes of length 200 Mbp with a recombination
rate of 1e-8 per bp per generation and neutral mutations dropped onto the branches at a rate
of 1.25e-8 per bp per generation. It is also possible to simulate using a real human
chromosome recombination map in which case the sequence length is given by the file and a
variable recombination rate across the sequence is applied. To manipulate and examine
trees in tree sequence files we use the tskit python API.

We performed Principal Component Analysis on the simulated genotype data to compare the
structure to that of the real data. Both the MesoNeo chromosomes and nine simulated
chromosomes were filtered for minor allele frequency 5%. We used EIGENSOFT smartpca
[28] with the outlier removal option disabled to perform PCA with no projection on both the
imputed MesoNeo samples together with 1000 Genomes GBR samples and the simulated
samples.

In Figure S8 the cluster for each simulated ancient population falls in the same vicinity of
PCA space as the real MesoNeo samples and the variation explained by PC1 and PC2 is
comparable. Overall, the similarity of the PCA suggests that a lot of the underlying structure
that determines how these groups relate to each other is captured by the demographic
model.




                                                                                               28

```

## AP supplement printed page30,PDF page31

```text
The relationships of these ancient groups and present day Europeans to each other and the
split/admixture times are well established and needed no adjustments in the model.
However, to produce simulated PCA and Fst results that matched the real data, the
population sizes shown in Figure S7 needed some tuning. Both the Fst analysis and PCA
final results suggest that, while the model is undoubtedly not entirely correct, it produces
simulated data that looks close enough to the true data, by these measurements, that we
assume the model represents the real population history of Europeans.


A method for estimating local path ancestry

Our method infers the path that segments of a chromosome have taken through a population
history. Each segment is covered by a single marginal tree in a tree sequence. Figure S10
depicts how the path ancestry of focal haplotype can be determined from the tree relating
that haplotype to all other sample haplotypes: We traverse up the tree from the focal
haplotype, jumping to successive parent nodes towards the root i.e parent, grandparent,
great-grandparent etc. If we know the population identity of each of the internal nodes
traversed to along the way, then we know the path. This information is recorded in tree
sequences simulated by msprime and so the path ancestry of haplotypes in simulated tree
sequences is straight-forward to find with tskit
(https://tskit.dev/tskit/docs/stable/introduction.html).

However in RELATE tree sequences, inferred from simulated or real variant data, the
population label of internal nodes is not known. We therefore adapted the concept of
Genealogical Nearest Neighbours (GNNs) [29] to identify the population identity of each
internal node encountered during traversal from a focal admixed haplotype. This involves
recording the proportion of each ancestral group that makes up the sample leaves below
each internal node, not including leaves seen at the previous nodes in traversal. The ordered
collection of all x GNN distributions of all x nodes examined during a tree traversal reflects
the path that the focal haplotype has taken to the root. Figure S10 demonstrates how these
ordered GNNs can be used to determine the path.




                                                                                                 30

```

## AP supplement printed page32,PDF page33

```text
Columns 1–n contain the proportions, between 0 and 1, of leaves belonging to each of n
ancient sampled groups and column n +1 contains the age of the node. Informative nodes
are those that have at least one leaf from the set of ancient sampled groups. If the root of the
tree is reached in less than five informative nodes, then the remaining rows of the matrix are
filled with −15 as ‘padding’. The training labels signify the path extracted from the
corresponding simulated tree sequences (determined simply using tskit).

The method also works when traversing up to the root and finding the path for ancient
sample haplotypes. Therefore, we also train over GNNs extracted from admixed samples,
while samples that are diagnostic of one path (e.g., Figure S7 CHG, WHG, EHG and Ana)
can be assigned by their population identity alone.

The method performs well for a range of demographic scenarios including variable number
of paths and extent of population differentiation, as well as being robust to misspecification in
the underlying model (Supplementary Notes 3.2, 3.3).


Training and testing a neural network on simulated European genomes

We simulated three 200Mbp tree sequences from our model of European population
structure, using different random seeds. RELATE tree sequences were inferred from the
corresponding simulated VCF files. Default parameters for RELATE were used; 1.25e-8
mutation rate and starting population size estimate of haplotypes of 30,000.

We extracted 10,000 training pairs of GNNs and true labels from each of the five admixed
sample populations (GBR, Bronze Age, Yamnaya, Neolithic farmers and Bronze Age
anatolians), 50,000 pairs in total. GNNs were taken from the trees covering evenly spaced
sites across three tree sequences output from RELATE, avoiding most of the correlation
between trees. True path labels were taken from the trees covering the same sites in the
corresponding simulated tree sequences. We trained the classifier to predict the path labels
from the GNNs.

For testing, we generated five more simulated and RELATE inferred pairs of tree sequences
and tested the classifier on 10,000 GNNs from each to obtain a mean accuracy of 93.12%
with a standard deviation of 0.29% across the five tree sequences.

To test the precision of the classifier at identifying each path in each population, we pooled
the testing GNNs from all five sequences and applied the classifier to each admixed
                                                                                              32

```

## ARGMix cached primary paragraph `research/reviews/ARGMix_published_benchmark_audit_20261015_v1/paper_text.txt:9`

Graph transformers can facilitate several graph-related tasks. In the context of local ancestry, one approach might be to perform node classification by applying a transformer to the entire coalescent tree, but this does not scale with standard self-attention (O(n2) in time) [17]. Instead, we turned the task of predicting local ancestry into a subgraph classification problem. This sub-sampling approach is similar to existing graph neural network methods that scale to large graphs, improving scalability [23]. We extracted subgraphs containing specified references informative of ancestry (excluding other extant haplotypes that we may be predicting) and inferred ancestral haplotypes (internal nodes) around the extant haplotype (leaf node) being classified (Figure 1A). Specifically, we selected the nearest reference samples by TMRCA, regardless of ancestry, and a representative set of ancestral haplotypes encoding coalescent events. We considered the four-way demography that created present-day Europe, where the ancestral populations are Anatolian (early farmers), Western, Eastern, and Caucasus hunter-gatherers [24]. In this demographic model of Europe, Anatolian farmers expanded into Europe and eventually admixed with Western hunter-gatherers (WHG), forming an admixed Neolithic farmer population. Later, the Caucasus hunter-gatherers (CHG) admixed with Eastern hunter-gatherers (EHG) just north of the Caucasus, forming the Yamnaya steppe population. Critically, Yamnaya and Neolithic farmer samples are informative of their two respective admixed ancestries and are used as additional references. For clarity, Neolithic farmers here refer to individuals with WHG and Anatolian farmer ancestry.

## ARGMix cached primary paragraph `research/reviews/ARGMix_published_benchmark_audit_20261015_v1/paper_text.txt:29`

For each sample i in a tree, we select the nearest reference haplotypes by time-to-most-recent-common-ancestor (TMRCA), excluding post-Neolithic and present-day European samples and the three immediate ancestral haplotypes. Population labels are tokenized for transformer inputs. This model is based on the graph relative positional encoding introduced by Park et al. 2022 with minor modifications in which we drop encodings related to edges [19]. For self-attention, the linear projections are given by (1) qi=Wqueryxi,ki=Wkeyxi,vi=Wvaluexi.

## ARGMix cached primary paragraph `research/reviews/ARGMix_published_benchmark_audit_20261015_v1/paper_text.txt:41`

We generated training data using the same simulation as Irving-Pease et al. 2024 [11]. We modified the msprime simulations to exclude Bronze Age Anatolians [46], which were not relevant to our dataset. Similarly, we increased the number of simulated samples to reflect the real dataset we constructed. To better mimic imputation error in ancient DNA, we added a genotype error of 0.001 across all samples, including present-day samples [47]. We generated misspecified demographic scenarios by perturbing the original model of present-day Europe, shifting event timings and effective population size. We randomly varied effective population sizes by an increase or decrease of 1–50%. Admixture timings were increased by 1–25% so as not to break the demography relative to sample times, and population divergences were varied by 1–25% in either direction. We generated all of these simulations using the recombination map for chromosome 22. We ran Relate with an effective population size of 30,000 and a mutation rate of 1.25×10−8 on the simulated data and extracted the ground-truth local ancestry using the simulated ancestral recombination graphs.

## ARGMix cached primary paragraph `research/reviews/ARGMix_published_benchmark_audit_20261015_v1/paper_text.txt:43`

We reimplemented AncestralPaths in PyTorch [48], using the proportion of genealogical nearest neighbors (GNN) combined with average sample time [11]. We trained both ARGMix and AncestralPaths on the same 500,000 examples over five replicates. To mitigate the redundancy from local correlation between trees we skipped every ten trees. For validation, we used a separate replicate, skipping every 100 trees and using 100,000 examples. For AncestralPaths, we used the default parameterization that aggregates the GNNs up to five nodes from the extant leaf. For ARGMix, we used 25 of the nearest references by TMRCA plus three inferred ancestral nodes, yielding a 29-node subgraph including the extant leaf being classified. For both models we took the best model by validation loss over all the epochs for further evaluation. To initially evaluate the accuracy, we tested on three additional replicates extracting 100,000 samples from each. We then evaluated five misspecified replicates, also with 100,000 samples and no skipped trees.

## ARGMix cached primary paragraph `research/reviews/ARGMix_published_benchmark_audit_20261015_v1/paper_text.txt:45`

We prepared the large MesoNeo aDNA dataset from Allentoft et al. 2024 [24]. First, we filtered down to the 1492 samples that passed the authors’ quality control. We then restricted to Western Eurasian samples already painted by AncestralPaths. Finally, we filtered for variants with an INFO > 0.8. To incorporate the Iceman we started with the BAM file available from Wang et al. 2023 [20]. Using GLIMPSE [49], we applied the same imputation parameters as Allentoft et al. 2024 [24]. However, instead of using the 1000 Genomes Project phase 3 alone, we used a merged Human Genome Diversity Project and 1000 Genomes Project reference panel for imputation [50–52]. This provided additional present-day European populations, such as Sardinians, who have retained more Neolithic farmer ancestry [53]. After imputation, we merged present-day European populations and the Iceman on the intersection of well-imputed variants in the MesoNeo dataset. This yielded 3,720,746 variants total and 1611 samples. All the simulations were generated assuming this sample count. To address differences in haplotype phase, we rephased the final variant set with Beagle 5.4 [54].

## ARGMix cached primary paragraph `research/reviews/ARGMix_published_benchmark_audit_20261015_v1/paper_text.txt:47`

To construct a reference panel for ARGMix and AncestralPaths, we used the references available in the supplement of Irving-Pease et al. 2024 [11]. We further separated the early farmers and Anatolian farmers to identify those suitable as references for Anatolian ancestry alone. Using 255,816 filtered variants (minor allele frequency exceeding 10% and pruned for linkage-disequilibrium), we ran ADMIXTURE with four clusters on the 1611 samples (Supplementary Figure 2) and required the WHG component to be less than 0.03. The remaining farmers were considered Neolithic farmers [33]. Otherwise we used the same samples as references for the remaining ancestries. Notably, we excluded Bronze Age Anatolians from the simulation used by Irving-Pease et al. 2024 [11]. This final reference set determined simulation parameters, namely sample times and per-population sample sizes (Supplementary Figure 1). In total, we had 22 Anatolians, 13 CHG, 42 EHG, 159 Neolithic Farmers (including the Iceman), 48 WHG, 13 Yamnaya. Additionally, we included 654 post-Neolithic samples and 660 present-day Europeans from HGDP and 1KGP [52]. For all ancestry-specific analyses, we used snputils [55], including PCA and f-statistics. To apply CLUES2 we followed the approach of Vaughn et al. 2024 for performing the ancestry-stratified selection tests [21]. For rs3135388 we partitioned each of the ancient haplotypes based on the local ancestry inferred by ARGMix. If a sample was not included in local ancestry inference but was used as a reference, we also included its haplotypes. Then, we computed the frequency of the allele in present-day Europeans conditioned on local ancestry. We used the default parameters otherwise.

## ARGMix cached supplementary caption lines8–9

Supplementary Figure S1: PCA of ancient DNA samples. Indicated in bold shapes are references used for
local ancestry with ARGMix.
