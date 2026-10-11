# Bounded source excerpts

These are scientific paper/source text, not biological inputs or target arrays. Source SHA256 records are in source_evidence.json.

## ARGformer primary paper: directly inspected paragraphs

### 2.1 BERT model for encoding paths in ancestral recombination graphs

Where τ>0 is a temperature hyperparameter (default τ=0.05). The denominator sums over all in-batch samples except the anchor, providing implicit hard negatives from other classes. Anchors with no same-class partners in the batch (|P(i)|=0) are excluded from the loss. To mitigate class imbalance, we apply inverse-frequency weighting:

### 2.2 ARGformer on simulated data

We first applied ARGformer to simulated data generated using msprime (Kelleher et al. 2016). We simulated 1600 samples from four populations (Admixed = 1000, European = 200, African = 200, East Asian = 200) in the three-population out-of-Africa demography (AmericanAdmixture 4B18) from the stdpopsim catalog (Adrion et al. 2020), we use the default parameters for a single 30 million base pair (bp) chromosome, using a flat recombination rate of 1×10−8 per bp generation and a mutation rate of 1.25e-8 per bp per generation (Browning et al. 2018, Lauterbur et al. 2023). We generated a dataset for pretraining using tskit to extract coalescent events (Wong et al. 2024). To better evaluate ARGformer on inferred ARGs, we inferred the ARG using tsinfer+tsdate on the simulated genotypes, without phasing or genotype error (Kelleher et al. 2019). We split this dataset into training (90%) and validation (10%) for pretraining.

### 3.1 Capturing population structure

We benchmarked local ancestry inference (LAI) in admixed individuals against FLARE (Browning et al. 2023). Because FLARE is a specialized local-ancestry method, we use it here as a task-specific baseline, while ARGformer is intended as a general representation model for multiple downstream analyses on inferred genealogies. Using $10 000$ unique marginal paths sampled from admixed individuals, we considered two inference strategies based solely on ARGformer embeddings. First, performing PCA on embeddings of unadmixed reference individuals and retaining the top 10 principal components. Second, project each path embedding into a 10-dimensional PC space and assign an ancestry label based on its cluster membership (PCA clustering).

## ARGMix availability boundary

April article specifics in this memo are attributed to the preserved prior primary-browser-read record in docs/argformer_control.md. A fresh primary article copy is not available locally. The current pinned author source was directly inspected. No repeated archive search or network request was performed.

## AP supplement

Source: .runtime/author_sources/ancestralpaths_supplement_41586_2023_6705_MOESM1_ESM.txt


Lines 869-889:

```text
869: Europeans to the root. Path 4 = orange, starts at present day Europeans, going back through
870: the Yamnaya to Eastern hunter gatherers and then through Northern Europeans to the root.
871: Paths 5 = blue, starting in the Bronze Age Anatolians and joins path 1 part. Path 6 = cyan,
872: starts in Bronze Age Anatolians and joins path 2. When paths overlap, lineages from all
873: overlapping paths can coalesce.
874: 
875: We constructed our model of European population history in msprime [18], a coalescent
876: simulator. The population sizes and admixture fractions are shown in the schematic in Figure
877: S7. This model has been submitted to the stdpopsim catalogue (https://github.com/popsim-
878: consortium/stdpopsim/blob/main/stdpopsim/catalog/HomSap/demographic_models.py) and
879: is publicly available for users to simulate data from.
880: 
881: 
882: Comparison of simulated data to MesoNeo genomes
883: 
884: The MesoNeo dataset is a large collection of both ancient and present day samples. While
885: the dataset contains samples from all over the world, most are from people that lived in the
886: last 10,000 years in Eurasia. The dataset is composed of 1,490 genomes that have been
887: imputed to 3.7 million SNPs once filtered for coverage (>0.1X), low imputation quality, close
888: relatives and >0.5 imputation INFO score (See [27] for data availability).
889: 
```


Lines 967-994:

```text
967: depicts how the path ancestry of focal haplotype can be determined from the tree relating
968: that haplotype to all other sample haplotypes: We traverse up the tree from the focal
969: haplotype, jumping to successive parent nodes towards the root i.e parent, grandparent,
970: great-grandparent etc. If we know the population identity of each of the internal nodes
971: traversed to along the way, then we know the path. This information is recorded in tree
972: sequences simulated by msprime and so the path ancestry of haplotypes in simulated tree
973: sequences is straight-forward to find with tskit
974: (https://tskit.dev/tskit/docs/stable/introduction.html).
975: 
976: However in RELATE tree sequences, inferred from simulated or real variant data, the
977: population label of internal nodes is not known. We therefore adapted the concept of
978: Genealogical Nearest Neighbours (GNNs) [29] to identify the population identity of each
979: internal node encountered during traversal from a focal admixed haplotype. This involves
980: recording the proportion of each ancestral group that makes up the sample leaves below
981: each internal node, not including leaves seen at the previous nodes in traversal. The ordered
982: collection of all x GNN distributions of all x nodes examined during a tree traversal reflects
983: the path that the focal haplotype has taken to the root. Figure S10 demonstrates how these
984: ordered GNNs can be used to determine the path.
985: 
986: 
987: 
988: 
989:                                                                                                  30
990: 
991: Figure S10. Overview of GNN extraction for a European population. (A) shows an example
992: marginal tree that relates the focal haplotype to all other haplotypes in the dataset. Samples
993: are shown by their population labels. (B) shows the GNN matrix determined from the
994: marginal tree, traversing up four nodes towards the root V1-V4. (C) shows how the path can
```


Lines 1002-1020:

```text
1002: infer RELATE tree sequences from the simulated genotype data. We then extract training
1003: GNNs from the RELATE trees and train a neural network to predict the path given the true
1004: path label extracted from the corresponding simulated tree sequences. By training on GNNs
1005: extracted from RELATE trees, we aim to ‘train in’ any bias from the RELATE inference so
1006: that when we apply the neural network to tree sequences inferred by RELATE from the real
1007: data, these biases are somewhat accounted for.
1008: 
1009: Training data for the network is the set of GNN distributions for the first five informative
1010: nodes traversed towards the root extracted from the RELATE inferred trees. The GNN
1011: distributions are configured as a 5 x n matrix, one row per 5 informative nodes examined.
1012: 
1013:                                                                                                31
1014: 
1015: Columns 1–n contain the proportions, between 0 and 1, of leaves belonging to each of n
1016: ancient sampled groups and column n +1 contains the age of the node. Informative nodes
1017: are those that have at least one leaf from the set of ancient sampled groups. If the root of the
1018: tree is reached in less than five informative nodes, then the remaining rows of the matrix are
1019: filled with −15 as ‘padding’. The training labels signify the path extracted from the
1020: corresponding simulated tree sequences (determined simply using tskit).
```


Lines 1050-1086:

```text
1050: 
1051: To test the precision of the classifier at identifying each path in each population, we pooled
1052: the testing GNNs from all five sequences and applied the classifier to each admixed
1053:                                                                                               32
1054: 
1055: population separately. Figure S11 shows the confusion matrices for each population,
1056: normalised by the sum of the rows to show the precision i.e of the labels assigned a class,
1057: what proportion are true positives. The GBR obtains the lowest accuracy. GBR samples
1058: have the greatest number of generations between sampling time and admixture time. More
1059: generations since the admixture event means more recombination events have broken down
1060: admixture LD, resulting in shorter tracts of ancestry. RELATE uses flanking SNPs to
1061: calculate the distance matrix and allows some relaxation on the mapping of SNPs when
1062: constructing a new tree along the chromosome. These properties make it more difficult to
1063: determine the switch of ancestry at the edges of tracts as there is some inertia in the
1064: changing of tree topologies compared to the true tree sequences. Shorter tracts produce
1065: more edges and therefore less accuracy in the GNN assignment. However, high accuracy is
1066: still maintained for all populations, even the GBR.
1067: 
1068: 
1069: 
1070: 
1071: Figure S11. Confusion matrices per population when testing on simulated data. Values are
1072: normalised by the sum of rows to show the precision values, when testing a classifier trained
1073: on the model of European population structure.
1074: 
1075: Figure S27 shows a classified painted chromosome alongside the true simulated
1076: chromosome. Noise appears as short tracts of correlated trees, within larger chunks of
1077: ancestry covering many sites in admixture LD.
1078: 
1079: 
1080:                                                                                               33
1081: 
1082: Comparison to an advanced LAI tool
1083: 
1084: GNOMix is a recent LAI tool that has been shown to outperform previous methods on whole
1085: genome data [13]. Like other LAI methods, GNOMix views each reference population as a
1086: discrete ancestral population with no awareness of the relationships between reference
```

## AP released census/length

Source: .runtime/author_sources/ancestralpaths_published_zenodo8319452/simulate.py


Lines 72-80:

```text
72: demography.add_admixture(time=T_bronze, derived="present_bronze", ancestral=["Neo", "Yam"], proportions=[0.5,0.5])
73: demography.add_admixture(time=T_neo, derived="Neo", ancestral=["WHG", "Ana"], proportions=[0.25,0.75])
74: demography.add_admixture(time=T_baa, derived="BAA", ancestral=["CHG", "Ana"], proportions=[0.25,0.75])
75: demography.add_admixture(time=T_Yam, derived="Yam", ancestral=["CHG", "EHG"], proportions=[0.5,0.5])
76: demography.add_population_split(time=T_europe, derived=["WHG", "EHG"], ancestral="NE")
77: demography.add_population_split(time=T_near_east, derived=["CHG", "Ana"], ancestral="WA")
78: demography.add_population_split(time=T_basal, derived=["WA", "NE"], ancestral="trunk")
79: 
80: demography.add_census(time=350)
```


Lines 119-124:

```text
119: #Simulate chromosome 3 only
120: tree_sequence_rep = msprime.sim_ancestry(num_replicates=10, recombination_rate=1e-8, demography=demography,samples=samples, sequence_length=2000000)
121: 
122: 
123: for j, tree_sequence in enumerate(tree_sequence_rep):
124: 	ts=msprime.sim_mutations(tree_sequence, rate=1.25e-8)
```

## AP released split/path ontology/model

Source: .runtime/author_sources/ancestralpaths_published_zenodo8319452/training.py


Lines 42-44:

```text
42: for tseq in [3,5,6]:
43: 	ts_rel=tskit.load(str(args.ts)+"_relate_popsize_"+str(tseq)+".trees")
44: 	ts_sim=tskit.load(str(args.ts)+"_"+str(tseq)+".trees")
```


Lines 74-90:

```text
74: 						parent_sim = tree_sim.parent(sample)
75: 						while int(tree_sim.time(parent_sim))!=350:
76: 							parent_sim=tree_sim.parent(parent_sim)
77: 						pop=tree_sim.get_population(parent_sim)
78: 						if samples[samset][0]=="BAA":
79: 							if pop==8:
80: 								path=5
81: 							elif pop==7:
82: 								path=6
83: 						elif pop==3:
84: 							path=4
85: 						elif pop==4:
86: 							path=3
87: 						elif pop==7:
88: 							path=2
89: 						elif pop==8:
90: 							path=1
```


Lines 172-177:

```text
172: lab_train_target=keras.utils.to_categorical(labels-1, 6)
173: model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(5, 9)),tf.keras.layers.Dense(512, activation=tf.nn.relu),tf.keras.layers.Dropout(0.2),tf.keras.layers.Dense(6, activation=tf.nn.softmax)])
174: model.summary()
175: model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
176: model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
177: model.fit(X_train, lab_train_target, epochs=5)
```

## AP released test parents

Source: .runtime/author_sources/ancestralpaths_published_zenodo8319452/testing_bygroup.py


Lines 95-101:

```text
95: for ts_num,tseq in enumerate(range(5,10)):
96: 	print("TREE SEQUENCE", tseq)
97: 	counts=np.zeros((10000,5,(9*int(args.nn))), dtype=float)
98: 	labels=np.zeros((10000,5), dtype=int)
99: 	
100: 	ts_rel=tskit.load(str(args.ts)+"_relate_popsize_"+str(tseq)+".trees")
101: 	ts_sim=tskit.load(str(args.ts)+"_"+str(tseq)+".trees")
```

## ARGMix current architecture

Source: .runtime/author_sources/argmix_c2481248eb0d/templates/forward_blank/argmix.yaml


Lines 45-80:

```text
45: model:
46:   num_refs: 15
47:   hidden_size: 128
48:   gt_output_dim: 16
49:   dim_feedforward_mult: 4
50:   num_heads: 4
51:   num_layers: 4
52:   dropout_prob: 0.25
53:   attn_dropout_prob: 0.1
54:   max_rel_distance: 14
55: 
56: joint_model:
57:   chunk_rows: 1024
58:   chunk_stride: 512
59:   gru:
60:     hidden_size: 128
61:     num_layers: 1
62:     bidirectional: true
63:     dropout: 0.1
64:   train:
65:     batch_size: 16
66:     num_epochs: 20
67:     learning_rate: 2.0e-4
68:     weight_decay: 1.0e-4
69:     label_smoothing: 0.1
70:     early_stop_patience: 2
71:     loss_weighting:
72:       mode: none
73:     sample_seed: 0
74:     reference_selection_augmentation:
75:       train:
76:         enabled: true
77:         drop_top_m_nearest: 1  # take top k+m refs, drop nearest m, keep next k (train only)
78:       val:
79:         enabled: false
80:         drop_top_m_nearest: 0
```

## GST native gate

Source: research/reviews/paper_dataset_evaluation_contract_v1/gst_source_metadata/src__graph_set_transformer__models__gst.py.txt


Lines 207-219:

```text
207:                 self.ln_q(x),
208:                 z_dense,
209:                 node_set_id,
210:                 mask,
211:             )
212:         else:
213:             set_info = z_set[batch]
214: 
215:         if self.use_gating:
216:             gate_input = torch.cat([x, set_info], dim=-1)
217:             g_set = torch.sigmoid(self.gate(gate_input))
218:             x_out = x + graph_info + g_set * set_info
219:         else:
```

## Census Relate-only scientific binding

Source: src/arggnn/census_native_qualification_runner_v1.py


Lines 194-207:

```text
194:       'feature_columns':list(range(7))+list(range(8,48)),'scaler':'identity_no_fitted_scaler',
195:       'original_TRAIN_parent_ids':[p for p in config['parents'] if '_train_' in p],'labels_in_input':False}
196:     if feature!=expected:raise ValueError('Original TRAIN feature/scaler lineage changed')
197:     if native.get('implementation_sha256')!=config['source_sha256']['scripts/infer_biological_paths_relate_v1.py'] or native.get('config_sha256')!=config['source_sha256']['configs/biological_paths_relate_v1.json'] or prepared.get('config_sha256')!=config['source_sha256']['configs/biological_candidate_pilot_v1.json']:raise ValueError('Original producer/config lineage changed')
198:     for path in ('src/arggnn/biological_candidate.py','src/arggnn/reference_encoder.py','src/arggnn/genealogy_baselines.py','src/arggnn/data.py'):
199:         if prepared.get('source_sha256',{}).get(path)!=config['source_sha256'][path]:raise ValueError('Prepared source feature transformation changed')
200:     paired=[];paths=set()
201:     for p in sorted(parents,key=lambda p:p['parent_id']):
202:         bank=next(b for b in banks if b['parent_id']==p['parent_id'])
203:         split='train' if '_train_' in p['parent_id'] else 'val'
204:         if p.get('split')!=split or bank.get('split')!=split or p.get('status')!='inferred_both_conditions' or p.get('truth_or_oracle_read') is not False:raise ValueError('Original development split/provenance changed')
205:         if bank.get('relate_entry')!=p:raise ValueError('Prepared parent observed producer entry changed')
206:         if p['inferred_tree'].get('condition')!='post_five_size_iterations_primary':raise ValueError('Post-five primary tree required')
207:         if len(bank['tiles'])!=8 or {t['tile_id'] for t in bank['tiles']}!=set(range(8)):raise ValueError('All32 original tiles required before selection')
```


Lines 264-266:

```text
264:     post=parent['post_five_iterations']
265:     post_checks={'clean_tree':parent['inferred_tree']['path'],'clean_tree_sha256':parent['inferred_tree']['sha256'],'sample_index_to_inferred_tree_node_id':mapping,'observed_sample_manifest':derived,'coverage_original_bp0':coverage,'local_to_original_offset_bp':offset,'sequence_length_local_bp':coverage[1]-coverage[0],'retained_fixed100bp_indexes':retained.tolist()}
266:     if any(post.get(k)!=v for k,v in post_checks.items()):raise ValueError('Condition-specific native source lineage changed')
```

## Physical bin support

Source: scripts/train_ancestralpaths_biological_v1.py


Lines 235-239:

```text
235:         # Include only bins whose original fixed center is retained. Clip their
236:         # outer physical edges to coverage; do not fill partial bins lacking centers.
237:         edges = parent["edges"][np.r_[retained, retained[-1] + 1]].copy()
238:         edges[0] = max(edges[0], coverage[0])
239:         edges[-1] = min(edges[-1], coverage[1])
```


Lines 494-507:

```text
494: def intersections(edges, intervals):
495:     bins, rows, spans = [], [], []
496:     i = j = 0
497:     while i < len(edges) - 1 and j < len(intervals):
498:         left = max(edges[i], intervals[j, 0])
499:         right = min(edges[i + 1], intervals[j, 1])
500:         if right > left:
501:             bins.append(i); rows.append(j); spans.append(right - left)
502:         if edges[i + 1] <= intervals[j, 1]:
503:             i += 1
504:         else:
505:             j += 1
506:     spans = np.asarray(spans, dtype=np.float64)
507:     if not np.isclose(spans.sum(), edges[-1] - edges[0], atol=1e-8, rtol=0):
```


Lines 545-549:

```text
545: def full_span_score(p4, parent, truth, columns, metrics_cfg):
546:     """Exact integration of constant prediction bins against every true interval."""
547:     p = p4[:, columns].astype(np.float64)
548:     labels = truth.c4[:, columns]
549:     bins, rows, spans = intersections(parent["edges"], truth.intervals)
```


Lines 644-656:

```text
644:     represented = float(parent["edges"][-1] - parent["edges"][0])
645:     coverage = parent["coverage_original"]
646:     return dict(query_count=len(columns), query_bp=total, denominator_query_bp=total,
647:                 original_parent_interval_bp0=[0, 2000000], normalization_length_bp=2000000,
648:                 original_parent_query_bp=2000000 * len(columns),
649:                 coverage_original_bp0=coverage,
650:                 physical_coverage_fraction=float((coverage[1] - coverage[0]) / 2000000),
651:                 represented_bin_support_bp=[parent["edges"][0], parent["edges"][-1]],
652:                 represented_span_bp=represented,
653:                 represented_fraction_original_parent=represented / 2000000,
654:                 covered_span_without_retained_center_bp=float(coverage[1] - coverage[0] - represented),
655:                 retained_original_grid_indexes=parent["retained_indexes"],
656:                 score_scope="conditional on exact retained-center clipped-bin support; uncovered tails excluded",
```

## ARGformer native full path and split

Source: .runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py


Lines 54-80:

```text
54: # For each sample, sweep trees left-to-right and add unique paths
55: sample_iterator = tqdm(
56:     enumerate(selected_sample_nodes), total=len(selected_sample_nodes), desc="Processing Samples", unit="sample"
57: )
58: for idx, sample_id in sample_iterator:
59:     tree_iterator = tqdm(
60:         enumerate(ts.trees()), total=ts.num_trees, desc=f"Processing Trees (Sample {idx+1})", unit="tree", leave=False
61:     )
62:     for tree_idx, tree in tree_iterator:
63:         key = get_path_key(tree, sample_id)
64:         if key not in path_key_to_id:
65:             path_id = next_path_id
66:             path_key_to_id[key] = path_id
67:             next_path_id += 1
68:             leaf_node_id = key[0]
69:             root_node_id = key[-1]
70:             unique_paths_rows[path_id] = {
71:                 "path_id": path_id,
72:                 "path_length": len(key),
73:                 "leaf_node_id": int(leaf_node_id),
74:                 "root_node_id": int(root_node_id),
75:                 "path_nodes": json.dumps(key),
76:             }
77:             # Initialize tree set for this path_key
78:             path_key_to_trees[key] = set()
79:         # Track that this path_key appears in this tree
80:         path_key_to_trees[key].add(tree_idx)
```


Lines 125-168:

```text
125: unique_path_sequences = deduplicate_sequences(unique_path_sequences)
126:     
127: def build_vocab_from_sequences(sequences):
128:     """
129:     Build vocabularies from the observed raw tokens in the sequences.
130:     Sequences contain only node IDs.
131:     Assign contiguous indices to unique node tokens (after reserved special tokens).
132:     """
133:     node_set = set()
134:     for seq in sequences:
135:         for token in seq:
136:             node_set.add(int(token))
137:     # Sort tokens to ensure reproducibility
138:     sorted_nodes = sorted(node_set)
139: 
140:     # Assign node tokens
141:     current_index = SPECIAL_OFFSET
142:     node_id_tokens = {}
143:     for n in sorted_nodes:
144:         key = f"node_{n}"
145:         node_id_tokens[key] = current_index
146:         current_index += 1
147: 
148:     vocab = {
149:         "special": SPECIAL_TOKENS,
150:         "node_id_tokens": node_id_tokens,
151:     }
152:     return vocab
153: 
154: vocab = build_vocab_from_sequences(unique_path_sequences)
155: 
156: def split_unique_sequences(sequences, train_frac=0.9):
157:     """
158:     Split the unique sequences (already deduplicated) into train and validation.
159:     """
160:     random.seed(RANDOM_SEED)
161:     random.shuffle(sequences)
162:     n = len(sequences)
163:     train_end = int(train_frac * n)
164:     train_seq_local = sequences[:train_end]
165:     val_seq_local = sequences[train_end:]
166:     return train_seq_local, val_seq_local
167: 
168: train_seq, val_seq = split_unique_sequences(unique_path_sequences, train_frac=TRAIN_FRAC)
```
