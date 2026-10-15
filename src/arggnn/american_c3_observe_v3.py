"""Current tsinfer+tsdate C3 observation adapter; explicitly distinct from true ARG."""
from .american_c3_common_v1 import ROOT,CLASSES,require,ref,bound,read,cli
from .american_c3_simulate_v1 import sanitize,dump_tree
SOURCE='src/arggnn/american_c3_observe_v3.py'
DEFAULTS=dict(schema='american_c3_inferred_observation_v1',execution_enabled=False,observation_ref=None,
    mutation_rate=1.25e-8,num_threads=8,max_node_map_bytes=2000000000,
    unary_preprocessing_policy='direct_tskit_simplify_all_samples_keep_unary_false_no_site_filter_no_trimming_no_gap_split_v1',max_tables_bytes=2000000000,max_sites=1000000,max_sample_store_bytes=4000000000,
    output_cap_bytes=12500000000,inference_recipe='current tsinfer.infer num_threads=8; explicit direct all-sample unary collapse with node map before unchanged tsdate.date defaults; reconstructed recipe, paper versions/options unresolved')

PREPROCESSING_POLICY='direct_tskit_simplify_all_samples_keep_unary_false_no_site_filter_no_trimming_no_gap_split_v1'


def verify_observed_sites(ts, nodes, g, pos, alleles_ref):
    """Verify all original sites and phased sample columns without allele-code assumptions."""
    import json,numpy as np
    require(ts.num_sites==len(pos),'Every original observed site must remain present')
    with bound(alleles_ref).open() as stream:
        for i,v in enumerate(ts.variants(samples=nodes)):
            original_alleles=json.loads(stream.readline())
            require(v.site.position==pos[i] and (v.genotypes>=0).all(),
                'Observed site/sample completeness drift')
            observed=np.asarray([v.alleles[int(a)] for a in v.genotypes])
            expected=np.asarray([original_alleles[int(a)] for a in g[i]])
            require(np.array_equal(observed,expected),'Observed allele/sample correspondence drift')
        require(not stream.readline(),'Extra original allele rows after inference/preprocessing')


def execute(cfg,w):
    import numpy as np,tsinfer,tsdate
    require(type(cfg.get('num_threads')) is int and cfg['num_threads']==8,
        'Explicit root-declared tsinfer num_threads=8 runtime setting required')
    require(cfg.get('unary_preprocessing_policy')==PREPROCESSING_POLICY
        and type(cfg.get('max_node_map_bytes')) is int and cfg['max_node_map_bytes']>0,
        'Exact root-declared unary-collapse reconstruction policy and node-map cap required')
    m=read(cfg['observation_ref']);source_roster=read(m['roster_ref']);roster=source_roster['samples']
    require(source_roster['parent_id']==m['parent_id'] and source_roster['class_order']==list(CLASSES)
        and [r['sample_index'] for r in roster]==list(range(len(roster))),'Stable original genotype-column/parent identity required')
    require(m['schema']=='american_c3_observed_v1' and m['query_ancestry_truth_in_inputs'] is False
        and m['class_order']==list(CLASSES) and cfg['mutation_rate']==1.25e-8,'Observation-only C3 source required')
    g=np.load(bound(m['genotypes_ref']),mmap_mode='r',allow_pickle=False)
    pos=np.load(bound(m['positions_ref']),allow_pickle=False)
    require(g.dtype==np.int8 and g.shape==(len(pos),len(roster)) and len(pos)<=cfg['max_sites']
        and (np.diff(pos)>0).all() and ((pos>=0)&(pos<m['sequence_length_bp'])).all(),'Complete phased genotype coordinate axes required')
    sample_path=w.prepare('samples/observed.samples',cfg['max_sample_store_bytes'])
    with tsinfer.SampleData(path=str(sample_path),sequence_length=m['sequence_length_bp'],
            max_file_size=cfg['max_sample_store_bytes']) as data:
        for i in range(0,len(roster),2):
            a,b=roster[i:i+2]
            require(a['sample_id']==b['sample_id'] and (a['haplotype'],b['haplotype'])==(0,1),'Preserved diploid order')
            data.add_individual(ploidy=2,time=0)
        with bound(m['alleles_ref']).open() as stream:
            import json
            for i,p in enumerate(pos):
                line=stream.readline();require(line,'Missing allele metadata')
                alleles=json.loads(line);require((g[i]>=0).all() and (g[i]<len(alleles)).all(),'No missing/error genotypes')
                data.add_site(position=float(p),genotypes=g[i],alleles=alleles,ancestral_allele=0)
            require(not stream.readline(),'Extra allele rows')
    # __exit__ finalises and reopens a read handle; close it before reloading.
    data.close()
    w.finish(sample_path,cfg['max_sample_store_bytes'])
    with tsinfer.SampleData.load(str(sample_path)) as samples:
        inferred=tsinfer.infer(samples,num_threads=cfg['num_threads'],record_provenance=True)
    inferred_ref=dump_tree(w,'observed/tsinfer_sanitized.trees',sanitize(inferred),cfg['max_tables_bytes'])
    raw_samples=inferred.samples()
    collapsed,node_map=inferred.simplify(samples=inferred.samples(),keep_unary=False,
        filter_sites=False,filter_individuals=False,filter_populations=False,map_nodes=True)
    require(inferred.num_samples==collapsed.num_samples==len(roster)
        and inferred.sequence_length==collapsed.sequence_length==m['sequence_length_bp']
        and inferred.num_sites==collapsed.num_sites==len(pos),
        'Unary collapse must preserve every original sample, site and coordinate extent')
    require(node_map.dtype==np.int32 and node_map.shape==(inferred.num_nodes,)
        and node_map.nbytes+4096<=cfg['max_node_map_bytes']
        and np.array_equal(node_map[raw_samples],collapsed.samples()),
        'Exact original sample order and bounded old-to-new inferred-node map required')
    verify_observed_sites(collapsed,list(map(int,collapsed.samples())),g,pos,m['alleles_ref'])
    map_ref=w.npy('observed/tsinfer_unary_old_to_new_node_map.npy',node_map)
    preprocessing_ref=w.json('observed/unary_preprocessing.json',dict(
        schema='american_c3_explicit_unary_collapse_v1',policy=PREPROCESSING_POLICY,
        raw_inferred_tree_ref=inferred_ref,old_to_new_node_map_ref=map_ref,
        raw_num_nodes=inferred.num_nodes,collapsed_num_nodes=collapsed.num_nodes,
        raw_sample_nodes=list(map(int,raw_samples)),
        collapsed_sample_nodes=list(map(int,collapsed.samples())),
        num_sites=collapsed.num_sites,sequence_length_bp=collapsed.sequence_length,
        all_observed_sites_and_genotypes_verified=True,all_sample_identities_preserved=True,
        filter_sites=False,filter_individuals=False,filter_populations=False,keep_unary=False,
        trimming_performed=False,gap_splitting_performed=False,allow_unary_bypass=False,
        tsdate_options_unchanged=True))
    del node_map,raw_samples,inferred
    dated=tsdate.date(collapsed,mutation_rate=cfg['mutation_rate'],record_provenance=True)
    del collapsed
    clean=sanitize(dated)
    require(clean.num_samples==len(roster) and clean.sequence_length==m['sequence_length_bp'],'No trimming/relabeling/subsampling')
    actual_nodes=list(map(int,clean.samples()))
    require(len(actual_nodes)==len(roster),'Exact inferred sample identity map')
    new_rows=[dict(r,observed_node_id=actual_nodes[i]) for i,r in enumerate(roster)]
    tree_ref=dump_tree(w,'observed/tsinfer_tsdate_sanitized.trees',clean,cfg['max_tables_bytes'])
    roster_ref=w.json('observed/roster.json',dict(parent_id=m['parent_id'],class_order=list(CLASSES),samples=new_rows,
        sample_index_to_inferred_node=actual_nodes))
    verify_observed_sites(clean,actual_nodes,g,pos,m['alleles_ref'])
    output=dict(m,tree_ref=tree_ref,roster_ref=roster_ref,
        genealogy_kind='inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG',
        source_observation_ref=cfg['observation_ref'],inference_recipe=cfg['inference_recipe'],
        unary_preprocessing_policy=PREPROCESSING_POLICY,unary_preprocessing_ref=preprocessing_ref,
        inference_runtime_settings=dict(tsinfer_num_threads=cfg['num_threads'],
            tsinfer_engine='installed default C_ENGINE',sample_store_read_api='tsinfer.SampleData.load'),
        sites_and_sample_identities_verified=True,migrations_and_internal_population_labels_removed=True)
    obs_ref=w.json('observed/manifest.json',output)
    return w.json('completed.json',dict(status='completed_current_C3_inference',observation_ref=obs_ref,
        tsinfer_observation_ref=inferred_ref,unary_preprocessing_ref=preprocessing_ref,
        unary_preprocessing_policy=PREPROCESSING_POLICY,tsinfer_num_threads=cfg['num_threads'],author_Table1_exact_reproduction=False))


if __name__=='__main__':cli(SOURCE,DEFAULTS,execute,dependencies=('src/arggnn/american_c3_simulate_v1.py',))
