"""Current tsinfer+tsdate C3 observation adapter; explicitly distinct from true ARG."""
from .american_c3_common_v1 import ROOT,CLASSES,require,ref,bound,read,cli
from .american_c3_simulate_v1 import sanitize,dump_tree
SOURCE='src/arggnn/american_c3_observe_v2.py'
DEFAULTS=dict(schema='american_c3_inferred_observation_v1',execution_enabled=False,observation_ref=None,
    mutation_rate=1.25e-8,num_threads=8,max_tables_bytes=2000000000,max_sites=1000000,max_sample_store_bytes=4000000000,
    output_cap_bytes=12500000000,inference_recipe='current tsinfer.infer and tsdate.date defaults except explicit num_threads=8 reconstruction runtime setting; paper versions/options unresolved')


def execute(cfg,w):
    import numpy as np,tsinfer,tsdate
    require(type(cfg.get('num_threads')) is int and cfg['num_threads']==8,
        'Explicit root-declared tsinfer num_threads=8 runtime setting required')
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
    dated=tsdate.date(inferred,mutation_rate=cfg['mutation_rate'],record_provenance=True)
    clean=sanitize(dated)
    require(clean.num_samples==len(roster) and clean.sequence_length==m['sequence_length_bp'],'No trimming/relabeling/subsampling')
    actual_nodes=list(map(int,clean.samples()))
    require(len(actual_nodes)==len(roster),'Exact inferred sample identity map')
    new_rows=[dict(r,observed_node_id=actual_nodes[i]) for i,r in enumerate(roster)]
    tree_ref=dump_tree(w,'observed/tsinfer_tsdate_sanitized.trees',clean,cfg['max_tables_bytes'])
    roster_ref=w.json('observed/roster.json',dict(parent_id=m['parent_id'],class_order=list(CLASSES),samples=new_rows,
        sample_index_to_inferred_node=actual_nodes))
    # Independently verify all input sites and phased sample-column genotypes.
    require(clean.num_sites==len(pos),'Inferred observation must retain every declared genotype site')
    with bound(m['alleles_ref']).open() as stream:
        for i,v in enumerate(clean.variants(samples=actual_nodes)):
            original_alleles=json.loads(stream.readline())
            require(v.site.position==pos[i] and (v.genotypes>=0).all(),'Observed inferred site/sample completeness drift')
            observed=np.asarray([v.alleles[int(a)] for a in v.genotypes])
            expected=np.asarray([original_alleles[int(a)] for a in g[i]])
            require(np.array_equal(observed,expected),'Observed inferred allele/sample correspondence drift')
        require(not stream.readline(),'Extra original allele rows after inference')
    output=dict(m,tree_ref=tree_ref,roster_ref=roster_ref,
        genealogy_kind='inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG',
        source_observation_ref=cfg['observation_ref'],inference_recipe=cfg['inference_recipe'],
        inference_runtime_settings=dict(tsinfer_num_threads=cfg['num_threads'],
            tsinfer_engine='installed default C_ENGINE',sample_store_read_api='tsinfer.SampleData.load'),
        sites_and_sample_identities_verified=True,migrations_and_internal_population_labels_removed=True)
    obs_ref=w.json('observed/manifest.json',output)
    return w.json('completed.json',dict(status='completed_current_C3_inference',observation_ref=obs_ref,
        tsinfer_observation_ref=inferred_ref,tsinfer_num_threads=cfg['num_threads'],author_Table1_exact_reproduction=False))


if __name__=='__main__':cli(SOURCE,DEFAULTS,execute,dependencies=('src/arggnn/american_c3_simulate_v1.py',))
