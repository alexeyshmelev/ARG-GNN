"""American C3 reconstruction: genotype observations and separately traced pulse truth."""
from pathlib import Path
from .american_c3_common_v1 import ROOT,CLASSES,require,ref,cli
SOURCE='src/arggnn/american_c3_simulate_v1.py'
DEFAULTS=dict(schema='american_c3_simulation_reconstruction_v1',execution_enabled=False,
    model='AmericanAdmixture_4B18',species='HomSap',sequence_length_bp=30000000,
    mutation_rate=1.25e-8,recombination_rate=1e-8,admixture_generation=12.0,
    sample_unit='EXPLICIT_REQUIRED_haploid_or_diploid',sample_counts=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),
    interpretation='registered reconstruction; original Table1 sample unit/seed/label rule unresolved',
    class_order=list(CLASSES),parent_id='american_c3_train_000',split='train',ancestry_seed=21001,mutation_seed=1021001,
    max_sites=1000000,max_tables_bytes=2000000000,max_truth_intervals=100000,
    output_cap_bytes=6500000000)


def resolve_samples(cfg,populations):
    unit=cfg['sample_unit'];require(unit in ('haploid','diploid'),'Explicit sample_unit required; no silent paper ploidy assumption')
    names={getattr(p,'name',getattr(p,'id',None)):i for i,p in enumerate(populations)}
    require(set(cfg['sample_counts'])=={'AFR','EUR','ASIA','ADMIX'} and set(cfg['sample_counts'])<=set(names),
        'Resolve named catalog populations; do not assume numeric catalog order')
    factor=1 if unit=='haploid' else 2
    counts={p:factor*n for p,n in cfg['sample_counts'].items()}
    require(all(type(n) is int and n>0 and n%2==0 for n in counts.values()),'Even positive haploid counts required for explicit diploid pairing')
    return names,counts


def storage_bound(cfg,counts):
    h=sum(counts.values());q=counts['ADMIX'];s=cfg['max_sites'];i=cfg['max_truth_intervals']
    # Two kastore tree files, genotype int8, positions, bounded allele JSON, external truth NPZ.
    return 2*(cfg['max_tables_bytes']+16777216)+s*h+s*8+s*128+i*(q+16)+16777216+8388608


def sanitize(ts):
    """Retain genotype/topology/time/sample identities; strip ancestry-bearing annotations."""
    import numpy as np,tskit
    t=ts.dump_tables()
    t.nodes.population=np.full(t.nodes.num_rows,-1,dtype=np.int32)
    t.nodes.individual=np.full(t.nodes.num_rows,-1,dtype=np.int32)
    t.migrations.clear();t.populations.clear();t.individuals.clear();t.provenances.clear()
    t.metadata_schema=tskit.MetadataSchema(None);t.metadata=b''
    for table in (t.nodes,t.edges,t.sites,t.mutations):
        table.metadata_schema=tskit.MetadataSchema(None);table.packset_metadata([b'']*table.num_rows)
    out=t.tree_sequence()
    require(out.num_migrations==out.num_populations==out.num_individuals==0
        and (out.tables.nodes.population==-1).all() and (out.tables.nodes.individual==-1).all(),'No internal population/migration annotations')
    return out


def trace_pulse(ts,query_nodes,*,pulse,admix_population,destination_classes,max_intervals):
    """Classify each query lineage at the actual recorded admixture pulse; no fallback."""
    import numpy as np
    rows=[m for m in ts.migrations() if m.source==admix_population and m.time==pulse]
    require(rows and all(m.dest in destination_classes for m in rows),'Recorded ADMIX pulse migrations to all declared sources required')
    starts={};ends={}
    for n,m in enumerate(rows):starts.setdefault(m.left,[]).append(n);ends.setdefault(m.right,[]).append(n)
    cuts=sorted(set(ts.breakpoints())|set(starts)|set(ends));active=set();intervals=[];labels=[]
    query_lookup={int(u):j for j,u in enumerate(query_nodes)}
    for left,right in zip(cuts[:-1],cuts[1:]):
        active.difference_update(ends.get(left,[]));active.update(starts.get(left,[]))
        tree=ts.at(left);row=np.full(len(query_nodes),255,dtype=np.uint8)
        for n in sorted(active):
            m=rows[n];cls=destination_classes[m.dest]
            for sample in tree.samples(m.node):
                j=query_lookup.get(int(sample))
                if j is not None:
                    require(row[j] in (255,cls),'Ambiguous pulse source; no majority label substitution')
                    row[j]=cls
        require((row<3).all(),'Every query/basepair must trace to one pulse destination')
        if labels and np.array_equal(labels[-1],row):intervals[-1][1]=right
        else:
            require(len(labels)<max_intervals,'Truth interval count cap exceeded')
            intervals.append([left,right]);labels.append(row)
    return np.asarray(intervals,dtype=np.float64),np.asarray(labels,dtype=np.uint8)


def dump_tree(writer,name,ts,max_tables):
    require(ts.tables.nbytes<=max_tables,'Tree table byte cap exceeded')
    planned=ts.tables.nbytes+16777216;p=writer.prepare(name,planned);ts.dump(p)
    return writer.finish(p,planned)


def execute(cfg,w):
    import msprime,stdpopsim,numpy as np,json
    require(cfg['schema']==DEFAULTS['schema'] and cfg['class_order']==list(CLASSES)
        and cfg['mutation_rate']==1.25e-8 and cfg['recombination_rate']==1e-8
        and cfg['split'] in ('train','val','test') and type(cfg['ancestry_seed']) is int
        and type(cfg['mutation_seed']) is int and cfg['mutation_seed']!=cfg['ancestry_seed']
        and type(cfg['parent_id']) is str and cfg['parent_id'] and cfg['sequence_length_bp']>0,'Registered C3 rate/split/independent seed contract required')
    model=stdpopsim.get_species(cfg['species']).get_demographic_model(cfg['model'])
    names,counts=resolve_samples(cfg,model.populations)
    require(storage_bound(cfg,counts)<=cfg['output_cap_bytes'],'Declared production count bound exceeds admitted cap')
    demography=model.model
    demo_names={p.name:i for i,p in enumerate(demography.populations)}
    require(demo_names==names,'Named demographic/catalog population correspondence required')
    def population_id(value):
        return demo_names[value] if isinstance(value,str) else value
    pulses=[e for e in demography.events if population_id(getattr(e,'source',None))==names['ADMIX']
        and e.time==cfg['admixture_generation']]
    require(pulses,'Catalog model must contain declared admixture pulse')
    samples=[msprime.SampleSet(num_samples=counts[p],population=names[p],time=0,ploidy=1) for p in ('AFR','EUR','ASIA','ADMIX')]
    ts=msprime.sim_ancestry(samples=samples,demography=demography,sequence_length=cfg['sequence_length_bp'],
        recombination_rate=cfg['recombination_rate'],random_seed=cfg['ancestry_seed'],record_migrations=True,discrete_genome=True)
    ts=msprime.sim_mutations(ts,rate=cfg['mutation_rate'],random_seed=cfg['mutation_seed'],model=msprime.JC69(),discrete_genome=True)
    require(0<ts.num_sites<=cfg['max_sites'] and ts.tables.nbytes<=cfg['max_tables_bytes'],'Full simulation count cap exceeded; no truncation')
    raw=list(map(int,ts.samples()));roster=[];offset=0
    source_classes={'EUR':0,'ASIA':1,'AFR':2}
    for pop in ('AFR','EUR','ASIA','ADMIX'):
        for j in range(counts[pop]):
            roster.append(dict(sample_index=offset+j,observed_node_id=raw[offset+j],sample_id=f'{pop}_{j//2:05d}',
                haplotype=j%2,age_generations=0,role='query' if pop=='ADMIX' else 'pure_source_reference',
                observed_reference_class=None if pop=='ADMIX' else source_classes[pop]))
        offset+=counts[pop]
    require(offset==ts.num_samples,'Exact declared sample count required')
    expected_populations=[names[p] for p in ('AFR','EUR','ASIA','ADMIX') for _ in range(counts[p])]
    require([ts.node(u).population for u in raw]==expected_populations,'Declared sample-major population correspondence required')
    query=np.asarray([r['observed_node_id'] for r in roster if r['role']=='query'],dtype=np.int64)
    intervals,classes=trace_pulse(ts,query,pulse=cfg['admixture_generation'],admix_population=names['ADMIX'],
        destination_classes={names[p]:c for p,c in source_classes.items()},max_intervals=cfg['max_truth_intervals'])
    true_ref=dump_tree(w,'truth/simulated_true_with_migrations.trees',ts,cfg['max_tables_bytes'])
    truth_ref=w.npz('truth/c3_pulse_ancestry.npz',intervals_bp=intervals,
        query_sample_indices=np.asarray([r['sample_index'] for r in roster if r['role']=='query'],dtype=np.int64),
        source_classes_0based=classes,parent_id=np.asarray(cfg['parent_id']),class_order=np.asarray(CLASSES))
    observed=sanitize(ts)
    tree_ref=dump_tree(w,'observed/simulated_true_topology_sanitized.trees',observed,cfg['max_tables_bytes'])
    positions_ref=w.npy('observed/positions_bp0.npy',observed.tables.sites.position)
    h=observed.num_samples;s=observed.num_sites
    gp=w.prepare('observed/genotypes_int8.npy',s*h+4096)
    genotype=np.lib.format.open_memmap(gp,mode='w+',dtype=np.int8,shape=(s,h))
    ap=w.prepare('observed/site_alleles.jsonl',s*128+4096)
    with ap.open('x') as stream:
        for i,v in enumerate(observed.variants()):
            require(v.genotypes.min()>=0 and v.genotypes.max()<len(v.alleles),'Complete phased genotypes without missing/error')
            genotype[i]=v.genotypes
            line=json.dumps(list(v.alleles),separators=(',',':'))+'\n'
            require(len(line.encode())<=128,'Allele serialization bound exceeded');stream.write(line)
    genotype.flush();del genotype
    genotype_ref=w.finish(gp,s*h+4096);alleles_ref=w.finish(ap,s*128+4096)
    roster_ref=w.json('observed/roster.json',dict(parent_id=cfg['parent_id'],class_order=list(CLASSES),samples=roster))
    manifest=dict(schema='american_c3_observed_v1',parent_id=cfg['parent_id'],split=cfg['split'],class_order=list(CLASSES),
        sequence_length_bp=ts.sequence_length,sample_unit=cfg['sample_unit'],counts_haploid=counts,
        genealogy_kind='simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG',
        tree_ref=tree_ref,genotypes_ref=genotype_ref,positions_ref=positions_ref,alleles_ref=alleles_ref,roster_ref=roster_ref,
        query_ancestry_truth_in_inputs=False,migrations_and_internal_population_labels_removed=True,
        interpretation=cfg['interpretation'],mutation_model='JC69/discrete_genome; paper setting not explicitly released')
    obs_ref=w.json('observed/manifest.json',manifest)
    return w.json('completed.json',dict(status='completed_C3_simulation_reconstruction',observation_ref=obs_ref,
        private_true_tree_ref=true_ref,private_truth_ref=truth_ref,class_order=list(CLASSES),parent_id=cfg['parent_id'],
        sample_unit=cfg['sample_unit'],storage_count_bound_bytes=storage_bound(cfg,counts),
        actual_tables_bytes=ts.tables.nbytes,sites=ts.num_sites,truth_intervals=len(intervals),
        author_Table1_exact_reproduction=False))


if __name__=='__main__':cli(SOURCE,DEFAULTS,execute)
