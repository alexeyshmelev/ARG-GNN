#!/usr/bin/env python3
"""Prospective1611 CPU generation. Numeric commands are deferred, never implicit.

One shared observed realization is exported; full genealogy/census truth is private.
Original author repositories and reviewed adapter source remain frozen.
"""
from __future__ import annotations
import argparse
import bisect
import collections
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
ADAPTER=ROOT/'research/reviews/ARGMix_prospective1611_native_adapters_20261016_v1'
PLAN=ROOT/'research/reviews/ARGMix_published_panel_fair_comparison_source_plan_20261016_v1/prospective_panel_config.proposed.inactive.json'
SEEDS=ROOT/'research/reviews/ARGMix_published_panel_next_source_20261015_v1/published_panel_plan.v1.json'
OFFICIAL_ARCHIVE_SHA='80f22d9e6cb0e497074ed1bc277e765fa9d8e22f21b2f66c3b10286520f6b68f'
OFFICIAL_MEMBER='genetic_map_GRCh37_chr22.txt'
OFFICIAL_SOURCE_SHA='fe688fb1722c0f2aaeb691d813ad5d759fd033e345369e50694f5cb04b12d8a8'
MAP_ALGORITHM='linear_cumulative_interpolation_clip_rebase_v1'
RATE_SEMANTICS='piecewise_constant_between_breakpoints_from_cumulative_cM_slope'
COHORT_POP={'NF':'Neo','PostNeolithic':'present_bronze','Present':'present_bronze'}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def integer(v,name,minimum=0):
    if isinstance(v,bool) or not isinstance(v,int) or v<minimum:raise ValueError(name+' invalid integer')
    return v


def digest(v):
    if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):raise ValueError('invalid SHA256')
    return v


def verify_sources():
    doc=read_json(HERE/'source_pins.v1.json')
    for r in doc['sources']:
        p=ROOT/r['path']
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('source changed: '+r['path'])
    return len(doc['sources'])


def adapter_modules():
    verify_sources()
    sys.path.insert(0,str(ADAPTER))
    modules=[importlib.import_module(name) for name in ('panel_adapter','truth_adapter')]
    if any(Path(m.__file__).resolve()!=ADAPTER/(name+'.py') for m,name in zip(modules,('panel_adapter','truth_adapter'))):raise ValueError('loaded adapter module differs from frozen source')
    return tuple(modules)


def scientific_inputs(parent_id):
    plan=read_json(PLAN)
    registry=read_json(SEEDS)
    candidates=[r for r in registry['parents'] if r['parent_id']==parent_id]
    if len(candidates)!=1:raise ValueError('unregistered parent')
    parent=candidates[0]
    if parent['split'] not in ('train','val','iid'):
        raise ValueError('OOD generation requires separately frozen realized demographic dictionaries; not implemented by default')
    expected={'mutation_rate':1.25e-8,'error_rate':0.001}
    if any(plan['mutations_and_observation'][k]!=v for k,v in expected.items()):raise ValueError('pinned mutation/error condition changed')
    panel,truth=adapter_modules()
    roster=read_json(ADAPTER/'prospective_roster.unbound.v1.json')
    panel.Panel(roster,require_bound=False)
    demography=plan['demography']
    truth.validate_cut_demography(demography)
    if set(demography['populations_initial_size'])!={'present_bronze','Yam','Neo','EHG','WHG','NE','CHG','Ana','WA','trunk'}:
        raise ValueError('unexpected retained demography/BAA branch')
    return plan,parent,roster,demography


def sample_specs(roster):
    """One explicit diploid SampleSet per named synthetic diploid, roster order."""
    by_dip=collections.OrderedDict()
    for r in roster['rows']:by_dip.setdefault(r['diploid_id'],[]).append(r)
    result=[]
    for did,rows in by_dip.items():
        if len(rows)!=2 or {r['phase'] for r in rows}!={0,1}:raise ValueError('unpaired roster')
        r=rows[0]
        result.append({'diploid_id':did,'population':COHORT_POP.get(r['cohort'],r['cohort']),
                       'time':r['age_generations'],'ploidy':2,'num_samples':1,
                       'haplotype_ids':[x['haplotype_id'] for x in sorted(rows,key=lambda x:x['phase'])]})
    return result


def build_native_demography(msprime,doc):
    dem=msprime.Demography()
    # Preserve original population order with only the unused BAA entry omitted.
    for name in ('present_bronze','Yam','Neo','EHG','WHG','NE','CHG','Ana','WA','trunk'):
        dem.add_population(name=name,initial_size=doc['populations_initial_size'][name])
    for e in doc['admixtures']:dem.add_admixture(time=e['time'],derived=e['derived'],ancestral=e['ancestral'],proportions=e['proportions'])
    for e in doc['splits']:dem.add_population_split(time=e['time'],derived=e['derived'],ancestral=e['ancestral'])
    dem.add_census(time=doc['census_time']);dem.sort_events()
    return dem


def read_map_rows(path,columns,header):
    lines=[s.split() for s in Path(path).read_text().splitlines() if s.strip()]
    if not lines or lines[0]!=header:raise ValueError('map header differs from pinned receipt')
    if not {'position','rate','cumulative_cM'}<=set(columns) or set(columns)-{'chromosome','position','rate','cumulative_cM'}:raise ValueError('explicit map columns required')
    idx=[integer(columns[k],k) for k in ('position','rate','cumulative_cM')]
    if len(set(idx))!=3:raise ValueError('map columns must differ')
    out=[]
    for fields in lines[1:]:
        if max(idx)>=len(fields):raise ValueError('short map row')
        if 'chromosome' in columns:
            c=integer(columns['chromosome'],'chromosome column')
            if c>=len(fields) or fields[c]!='chr22':raise ValueError('source map row is not chromosome22')
        x,rate,cm=(float(fields[i]) for i in idx)
        if not all(math.isfinite(v) and v>=0 for v in (x,rate,cm)) or x!=int(x):raise ValueError('invalid map coordinate/rate')
        out.append((int(x),rate,cm))
    if len(out)<2 or any(b[0]<=a[0] or b[2]<a[2] for a,b in zip(out,out[1:])):raise ValueError('nonmonotone map')
    return out


def clip_rebase(source_rows,start,end):
    integer(start,'interval start');integer(end,'interval end',1)
    if not source_rows[0][0]<=start<end<=source_rows[-1][0]:raise ValueError('interval outside map support; extrapolation forbidden')
    xp=[r[0] for r in source_rows]
    def cm_at(x):
        i=bisect.bisect_right(xp,x)-1
        if source_rows[i][0]==x:return source_rows[i][2]
        a,b=source_rows[i:i+2]
        return a[2]+(b[2]-a[2])*(x-a[0])/(b[0]-a[0])
    baseline=cm_at(start)
    coords=[start]+[x for x in xp if start<x<end]+[end]
    values=[cm_at(x)-baseline for x in coords]
    out=[]
    for i,(x,cm) in enumerate(zip(coords,values)):
        rate=0.0 if i==len(coords)-1 else (values[i+1]-cm)*1e6/(coords[i+1]-x)
        out.append((x-start,rate,cm))
    return out


def verify_map_document(doc,*,expected_archive_sha=OFFICIAL_ARCHIVE_SHA):
    """The production entrypoint always uses the official archive SHA constant."""
    if doc.get('schema')!='prospective_chr22_GRCh37_map_derivation_receipt_v1' or doc.get('complete') is not True or doc.get('root_approved') is not True:
        raise ValueError('complete root-approved map derivation receipt required')
    if doc.get('chromosome')!='22' or doc.get('genome_build')!='GRCh37':raise ValueError('wrong chromosome/build')
    if not isinstance(doc.get('source_url'),str) or not doc['source_url'].startswith('https://'):raise ValueError('explicit authenticated source URL required')
    if doc.get('archive_sha256')!=expected_archive_sha or sha(doc['archive_path'])!=expected_archive_sha:raise ValueError('official map archive hash mismatch')
    digest(doc['source_map_sha256']);digest(doc['rebased_map_sha256'])
    if expected_archive_sha==OFFICIAL_ARCHIVE_SHA and (doc['source_member_name']!=OFFICIAL_MEMBER or doc['source_map_sha256']!=OFFICIAL_SOURCE_SHA):raise ValueError('wrong authenticated chr22 archive member')
    if sha(doc['source_map_path'])!=doc['source_map_sha256']:raise ValueError('source map bytes mismatch')
    with tarfile.open(doc['archive_path'],'r:*') as archive:
        member=archive.getmember(doc['source_member_name'])
        if not member.isfile():raise ValueError('map archive member must be a regular file')
        h=hashlib.sha256()
        with archive.extractfile(member) as stream:
            for b in iter(lambda:stream.read(1048576),b''):h.update(b)
        if h.hexdigest()!=doc['source_map_sha256']:raise ValueError('source map is not exact authenticated archive member')
    if doc.get('derivation_algorithm')!=MAP_ALGORITHM or doc.get('simulation_rate_semantics')!=RATE_SEMANTICS:
        raise ValueError('explicit cumulative-map derivation semantics required')
    if doc.get('physical_coordinate_transform')!='rebased_bp=source_bp-interval_start_bp; no source coordinate shift':
        raise ValueError('coordinate transform must be explicit')
    start,end=doc['source_interval_bp']
    source_columns={'chromosome':0,'position':1,'rate':2,'cumulative_cM':3}
    source_header=['Chromosome','Position(bp)','Rate(cM/Mb)','Map(cM)']
    rebased_columns={'position':0,'rate':1,'cumulative_cM':2}
    if doc.get('source_column_indices')!=source_columns or doc.get('source_header')!=source_header or doc.get('rebased_column_indices')!=rebased_columns:raise ValueError('raw four-column/native three-column layouts must be pinned separately')
    expected=clip_rebase(read_map_rows(doc['source_map_path'],source_columns,source_header),start,end)
    if doc.get('length_bp')!=end-start:raise ValueError('wrong map length')
    if sha(doc['rebased_map_path'])!=doc['rebased_map_sha256']:raise ValueError('rebased map bytes mismatch')
    actual=read_map_rows(doc['rebased_map_path'],rebased_columns,['position','COMBINED_rate(cM/Mb)','Genetic_Map(cM)'])
    if len(actual)!=len(expected) or any(a[0]!=b[0] or any(not math.isclose(a[k],b[k],rel_tol=1e-12,abs_tol=1e-12) for k in (1,2)) for a,b in zip(actual,expected)):
        raise ValueError('rebased map does not match exact clipped source cumulative curve')
    return actual


def rate_map_arrays(rows):
    positions=[r[0] for r in rows]
    rates=[(b[2]-a[2])/100.0/(b[0]-a[0]) for a,b in zip(rows,rows[1:])]
    if any(not math.isfinite(r) or r<0 for r in rates):raise ValueError('invalid recombination slope')
    return positions,rates


def validate_request(doc):
    verify_sources()
    required={'schema','root_approved','parent_id','science_plan_sha256','generator_source_sha256',
              'map_derivation_receipt','map_derivation_receipt_sha256','runtime_binding','runtime_binding_sha256',
              'resource_binding','resource_binding_sha256','observed_output_dir','private_truth_output_dir','private_receipt_dir'}
    if set(doc)!=required or doc['schema']!='prospective1611_generation_request_v1' or doc['root_approved'] is not True:raise ValueError('explicit root-approved generation request required')
    if doc['science_plan_sha256']!=sha(PLAN) or doc['generator_source_sha256']!=sha(HERE/'generation.py'):raise ValueError('generation/science source identity mismatch')
    scientific_inputs(doc['parent_id'])
    for key in ('map_derivation_receipt','runtime_binding','resource_binding'):
        if sha(doc[key])!=doc[key+'_sha256']:raise ValueError(key+' hash mismatch')
    runtime=read_json(doc['runtime_binding'])
    if runtime.get('schema')!='prospective1611_cpu_runtime_binding_v1' or runtime.get('root_qualified') is not True or runtime.get('cpu_only') is not True:raise ValueError('root-qualified CPU runtime binding required')
    if set(runtime.get('modules',{}))!={'msprime','numpy','tskit'}:raise ValueError('explicit msprime/numpy/tskit versions and files required')
    digest(runtime.get('python_executable_sha256'));digest(runtime.get('qualification_receipt_sha256'))
    for r in runtime['modules'].values():
        digest(r.get('sha256'))
        if not isinstance(r.get('version'),str) or not r['version'] or not isinstance(r.get('file'),str):raise ValueError('runtime version/path missing')
    resource=read_json(doc['resource_binding'])
    if resource.get('schema')!='prospective1611_generation_resource_binding_v1' or resource.get('root_approved') is not True or resource.get('enforced_by_root_wrapper') is not True:raise ValueError('root-enforced resource binding required')
    if resource.get('cpu_threads')!=1:raise ValueError('this source specifies one CPU thread per parent')
    digest(resource.get('wrapper_source_sha256'));digest(resource.get('admission_receipt_sha256'))
    for key in ('rss_bytes','wall_seconds','observed_output_bytes','private_truth_bytes','private_receipt_bytes'):integer(resource.get(key),key,1)
    paths=[Path(doc[k]).resolve() for k in ('observed_output_dir','private_truth_output_dir','private_receipt_dir')]
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[i+1:]):raise ValueError('observed/truth/receipt directories must be separate, nonnested fresh paths')
    if any(p.exists() for p in paths):raise ValueError('output paths must be fresh')
    return runtime,resource


def bind_generated_samples(ts,roster,population_names):
    """Allocate synthetic IDs from actual individuals/populations/ages, never offsets.

    Haplotypes are exchangeable within each generated diploid. Phase is the actual
    ts.samples encounter order within that individual, recorded explicitly once.
    """
    individuals=collections.OrderedDict()
    for nid in map(int,ts.samples()):
        n=ts.node(nid)
        if int(n.individual)<0:raise ValueError('simulated sample lacks diploid individual identity')
        individuals.setdefault(int(n.individual),[]).append(nid)
    queues=collections.defaultdict(collections.deque)
    for iid,nids in individuals.items():
        if len(nids)!=2:raise ValueError('simulator did not produce diploid pairs')
        values={(population_names[int(ts.node(n).population)],float(ts.node(n).time)) for n in nids}
        if len(values)!=1:raise ValueError('simulated mate population/age differs')
        queues[next(iter(values))].append((iid,nids))
    mapping=[];allocation=[]
    for spec in sample_specs(roster):
        key=(spec['population'],float(spec['time']))
        if not queues[key]:raise ValueError('generated sample population/age counts differ from roster')
        iid,nids=queues[key].popleft()
        for phase,(hid,nid) in enumerate(zip(spec['haplotype_ids'],nids)):
            mapping.append({'haplotype_id':hid,'simulation_node_id':nid})
            allocation.append({'haplotype_id':hid,'diploid_id':spec['diploid_id'],'phase':phase,'simulation_node_id':nid,'simulation_individual_id':iid})
    if any(queues.values()) or len(mapping)!=len(roster['rows']):raise ValueError('extra/missing generated individuals')
    return {'schema':'simulation_haplotype_node_mapping_v1','rows':mapping},allocation


def qualify_loaded_runtime(binding,modules):
    if sha(Path(sys.executable).resolve())!=binding.get('python_executable_sha256'):raise ValueError('Python executable differs from qualified runtime')
    for name,module in modules.items():
        r=binding['modules'][name]
        if module.__version__!=r.get('version') or Path(module.__file__).resolve()!=Path(r['file']).resolve() or sha(module.__file__)!=r.get('sha256'):
            raise ValueError('loaded '+name+' source/version differs from qualified CPU runtime')


def attach_sample_metadata(ts,allocation):
    """Keep node IDs and census nodes unchanged; never simplify genealogy."""
    import tskit
    tables=ts.dump_tables()
    node_meta={r['simulation_node_id']:{k:v for k,v in r.items() if k not in ('simulation_node_id','simulation_individual_id')} for r in allocation}
    individual_meta={r['simulation_individual_id']:{'diploid_id':r['diploid_id']} for r in allocation}
    old_nodes=list(tables.nodes);tables.nodes.clear();tables.nodes.metadata_schema=tskit.MetadataSchema.permissive_json()
    for i,n in enumerate(old_nodes):
        data=n.asdict();data['metadata']=node_meta.get(i,{})
        tables.nodes.add_row(**data)
    old_individuals=list(tables.individuals);tables.individuals.clear();tables.individuals.metadata_schema=tskit.MetadataSchema.permissive_json()
    for i,ind in enumerate(old_individuals):
        data=ind.asdict();data['metadata']=individual_meta.get(i,{})
        tables.individuals.add_row(**data)
    result=tables.tree_sequence()
    if list(map(int,result.samples()))!=list(map(int,ts.samples())):raise ValueError('metadata attachment changed sample IDs')
    return result


def apply_haploid_error(alleles,draws):
    """Fixed prospective Bernoulli(.001), with one PCG64 uniform draw per allele."""
    if len(alleles)!=len(draws) or not alleles:raise ValueError('allele/draw row mismatch')
    if any(a not in (0,1) for a in alleles):raise ValueError('binary alleles required')
    if any(not math.isfinite(float(u)) or not 0<=u<1 for u in draws):raise ValueError('uniform draws outside [0,1)')
    flips=[u<0.001 for u in draws]
    return [int(a)^int(f) for a,f in zip(alleles,flips)],flips


class OutputBudget:
    def __init__(self,limit):self.limit=limit;self.bytes=0
    def write(self,stream,text):
        self.bytes+=len(text.encode())
        if self.bytes>self.limit:raise ValueError('observed output byte cap exceeded; incomplete directory is unusable')
        stream.write(text)


def export_observed(ts,roster,mapping,out,start,length,error_seed,np,budget):
    """PCG64 draws once per true mutation site, stable roster-haplotype order.

    Flip every haploid allele independently; then retain only observed segregating
    binary sites. REF=A represents binary state0, ALT=C state1. No true state or
    per-site error flags are emitted to the public observed representation.
    """
    node_by_hap={r['haplotype_id']:r['simulation_node_id'] for r in mapping['rows']}
    ordered=[node_by_hap[r['haplotype_id']] for r in roster['rows']]
    specs=sample_specs(roster);rng=np.random.Generator(np.random.PCG64(error_seed))
    bit_digest=hashlib.sha256();sites=kept=flips=0
    paths={name:out/name for name in ('observed.vcf','observed.haps','observed.sample','variants.tsv','haplotypes.tsv')}
    with paths['observed.vcf'].open('x') as vcf,paths['observed.haps'].open('x') as haps,paths['observed.sample'].open('x') as sample,paths['variants.tsv'].open('x') as variants,paths['haplotypes.tsv'].open('x') as hapfile:
        budget.write(vcf,'##fileformat=VCFv4.2\n##source=prospective1611_shared_observed_generation_v1\n##contig=<ID=22,length='+str(length)+'>\n##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n')
        budget.write(vcf,'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'+'\t'.join(s['diploid_id'] for s in specs)+'\n')
        budget.write(sample,'ID_1 ID_2 missing\n0 0 0\n')
        for s in specs:budget.write(sample,s['diploid_id']+' '+s['diploid_id']+' 0\n')
        budget.write(hapfile,'haplotype_column_zero_based\thaplotype_id\tdiploid_id\tphase\tcohort\tage_generations\n')
        for i,r in enumerate(roster['rows']):budget.write(hapfile,'\t'.join(map(str,(i,r['haplotype_id'],r['diploid_id'],r['phase'],r['cohort'],r['age_generations'])))+'\n')
        budget.write(variants,'variant_id\tposition_bp_zero_based\tvcf_position_one_based\tsource_position_bp\tREF\tALT\tobserved_alt_haplotype_count\n')
        previous=-1
        for variant in ts.variants(samples=ordered,isolated_as_missing=False):
            pos=float(variant.site.position)
            if pos!=int(pos) or not previous<pos<length or pos<0:raise ValueError('noninteger/duplicate/out-of-range mutation position')
            previous=pos;sites+=1
            if any(a is not None and a not in ('0','1') for a in variant.alleles):raise ValueError('nonbinary mutation state')
            g=np.asarray(variant.genotypes)
            if g.shape!=(len(ordered),) or np.any(g<0):raise ValueError('missing/incorrect haplotype genotype row')
            states=np.asarray([int(a) if a is not None else -1 for a in variant.alleles],dtype=np.int8)
            alleles=states[g]
            observed,flip=apply_haploid_error(alleles.tolist(),rng.random(len(ordered)).tolist())
            bit_digest.update(str(int(pos)).encode()+b':'+np.packbits(flip,bitorder='little').tobytes()+b'\n')
            flips+=sum(flip)
            total=sum(observed)
            if total in (0,len(ordered)):continue
            kept+=1;vid='var%09d'%kept;position=int(pos)+1
            cells=[str(int(x)) for x in observed]
            phase=['|'.join(cells[i:i+2]) for i in range(0,len(cells),2)]
            budget.write(vcf,'22\t'+str(position)+'\t'+vid+'\tA\tC\t.\tPASS\t.\tGT\t'+'\t'.join(phase)+'\n')
            budget.write(haps,'22 '+vid+' '+str(position)+' A C '+' '.join(cells)+'\n')
            budget.write(variants,'\t'.join(map(str,(vid,int(pos),position,start+int(pos),'A','C',total)))+'\n')
    if not kept:raise ValueError('no observed segregating binary sites; no resimulation or interval fallback permitted')
    return {'simulated_mutation_sites':sites,'observed_sites':kept,'posterror_invariant_sites_dropped':sites-kept,
            'all_site_haploid_error_flips':flips,'all_site_error_bitstream_sha256':bit_digest.hexdigest(),
            'observed_bytes_streamed':budget.bytes,'files':paths}


def file_manifest(paths):
    return {k:{'path':str(Path(p).resolve()),'bytes':Path(p).stat().st_size,'sha256':sha(p)} for k,p in paths.items()}


def generate(request_path):
    request=read_json(request_path);runtime,resource=validate_request(request)
    plan,parent,roster,demography=scientific_inputs(request['parent_id'])
    map_doc=read_json(request['map_derivation_receipt']);map_rows=verify_map_document(map_doc)
    # The sole numerical generation path; no model/GPU import occurs.
    import msprime
    import numpy as np
    import tskit
    qualify_loaded_runtime(runtime,{'msprime':msprime,'numpy':np,'tskit':tskit})
    out=Path(request['observed_output_dir']).resolve();private=Path(request['private_truth_output_dir']).resolve();receipts=Path(request['private_receipt_dir']).resolve()
    for p in (out,private,receipts):p.mkdir(parents=True,exist_ok=False)
    seeds=parent['new_seed_values'];specs=sample_specs(roster)
    dem=build_native_demography(msprime,demography)
    position,rate=rate_map_arrays(map_rows)
    samples=[msprime.SampleSet(num_samples=s['num_samples'],population=s['population'],time=s['time'],ploidy=s['ploidy']) for s in specs]
    ancestry=msprime.sim_ancestry(samples=samples,demography=dem,recombination_rate=msprime.RateMap(position=position,rate=rate),
                    model=msprime.StandardCoalescent(),ploidy=2,discrete_genome=True,random_seed=seeds['ancestry'])
    if float(ancestry.sequence_length)!=map_doc['length_bp']:raise ValueError('ancestry sequence length differs from frozen interval')
    pop_names={p.id:p.metadata['name'] for p in ancestry.populations()}
    mapping,allocation=bind_generated_samples(ancestry,roster,pop_names)
    ancestry=attach_sample_metadata(ancestry,allocation)
    # Binary finite-sites transitions flip 0<->1; state_independent=False is explicit.
    truth=msprime.sim_mutations(ancestry,rate=1.25e-8,model=msprime.BinaryMutationModel(state_independent=False),
                    discrete_genome=True,random_seed=seeds['mutation'])
    del ancestry
    if not any(float(n.time)==350 for n in truth.nodes()):raise ValueError('generation lost exact census350 nodes')
    _,truth_adapter=adapter_modules()
    truth_adapter.simulation_sample_map(mapping,importlib.import_module('panel_adapter').Panel(roster,require_bound=False))
    write_json(out/'roster.unbound.json',roster)
    write_json(private/'simulation_node_mapping.json',mapping)
    write_json(private/'sample_allocation.json',{'schema':'prospective1611_simulation_sample_allocation_v1','rule':'synthetic diploids allocated within actual population/date; phase is actual ts.samples encounter order per individual','rows':allocation})
    write_json(private/'population_mapping.json',{'schema':'simulation_population_mapping_v1','populations':{str(k):v for k,v in pop_names.items()}})
    write_json(private/'demography.json',demography)
    truth.dump(str(private/'simulation.trees'))
    if sum(p.stat().st_size for p in private.iterdir() if p.is_file())>resource['private_truth_bytes']:raise ValueError('private truth byte cap exceeded; root quota must also enforce during native dump')
    export=export_observed(truth,roster,mapping,out,map_doc['source_interval_bp'][0],map_doc['length_bp'],seeds['genotype_error'],np,OutputBudget(resource['observed_output_bytes']))
    observed_paths=export.pop('files');observed_paths['roster']=out/'roster.unbound.json'
    observed={'schema':'prospective1611_shared_observed_generation_v1','complete':True,'parent_id':parent['parent_id'],'split':parent['split'],
              'diploids':len(specs),'haplotypes':len(roster['rows']),'observed_sites':export['observed_sites'],
              'files':file_manifest(observed_paths),'map_derivation_receipt_sha256':sha(request['map_derivation_receipt']),
              'map_binding':{'schema':'prospective_chr22_GRCh37_map_binding_v1','chromosome':'22','genome_build':'GRCh37',
                  'source_url':map_doc['source_url'],'source_sha256':map_doc['source_map_sha256'],'source_interval_bp':map_doc['source_interval_bp'],
                  'rebased_map_path':map_doc['rebased_map_path'],'rebased_map_sha256':map_doc['rebased_map_sha256'],'length_bp':map_doc['length_bp']},
              'coordinate_system':'rebased0-based [0,L); VCF/haps POS=rebased position+1; source position=rebased+interval start',
              'phase':'explicit diploid_id and phase mapping in haplotypes.tsv; one common realization for every method',
              'contains_genealogy_or_founder_truth':False,'generator_source_sha256':sha(HERE/'generation.py'),'science_plan_sha256':sha(PLAN),
              'error_model':'independent Bernoulli(0.001) haploid 0/1 flip at every simulated mutation site in roster order; PCG64(seed)',
              'published_table_reproduction':False}
    write_json(out/'map_binding.json',observed['map_binding'])
    observed['files']['map_binding']=file_manifest({'map':out/'map_binding.json'})['map']
    write_json(out/'observed_generation_receipt.json',observed)
    if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>resource['observed_output_bytes']:raise ValueError('total observed output byte cap exceeded')
    private_paths={p.name:p for p in private.iterdir() if p.is_file()}
    generation={'schema':'prospective1611_private_generation_receipt_v1','complete':True,'parent_id':parent['parent_id'],'split':parent['split'],
                'request':{'path':str(Path(request_path).resolve()),'sha256':sha(request_path)},'registered_seeds':seeds,
                'scientific_plan_sha256':sha(PLAN),'source_pins_sha256':sha(HERE/'source_pins.v1.json'),'generator_source_sha256':sha(HERE/'generation.py'),
                'observed_receipt':{'path':str(out/'observed_generation_receipt.json'),'sha256':sha(out/'observed_generation_receipt.json')},
                'private_truth_files':file_manifest(private_paths),'sample_specs_sha256':canonical(specs),'demography_semantic_sha256':canonical(demography),
                'map_derivation_receipt':{'path':str(Path(request['map_derivation_receipt']).resolve()),'sha256':sha(request['map_derivation_receipt'])},
                'runtime_binding_sha256':sha(request['runtime_binding']),'resource_binding_sha256':sha(request['resource_binding']),
                'mutation_model':'msprime.BinaryMutationModel(state_independent=False), rate1.25e-8, discrete_genome=True',
                'ancestry_model':'msprime.StandardCoalescent(), diploid SampleSets, discrete genome; no simplification',
                'private_realization_diagnostics':export,'truth_usage':'external census350 supervision/scoring only; never inferred-tree input',
                'no_BAA_scope':'unused noncontributing terminal population/event omitted; same retained marginal law, not identical seeded realization',
                'paper_reproduction':False}
    write_json(receipts/'generation_receipt.json',generation)
    if sum(p.stat().st_size for p in receipts.iterdir() if p.is_file())>resource['private_receipt_bytes']:raise ValueError('receipt byte cap exceeded')
    print(json.dumps({'complete':True,'parent_id':parent['parent_id'],'observed_receipt_sha256':sha(out/'observed_generation_receipt.json'),'private_generation_receipt_sha256':sha(receipts/'generation_receipt.json')},sort_keys=True))


def verify_manifest(manifest):
    for r in manifest.values():
        p=Path(r['path'])
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('artifact hash/bytes mismatch: '+str(p))


def validate_inference_coordinates(inference,observed):
    if inference.get('sequence_length_bp')!=observed['map_binding']['length_bp']:raise ValueError('inferred bank coordinate span differs from generated observations')
    transform=inference.get('coordinate_transform',{})
    if transform.get('export_position_encoding')!='POS=rebased_bp+1' or transform.get('normalized_tree_coordinates')!='rebased_bp_zero_based_half_open':raise ValueError('explicit export offset and normalized inferred-tree coordinates required')
    digest(transform.get('map_relate_transform_receipt_sha256'))
    return transform


def external_truth(args):
    """Later stage: requires actual observed inferred bank/targets, not true trees as features."""
    panel,truth_adapter=adapter_modules()
    generation=read_json(args.generation_receipt)
    if generation.get('schema')!='prospective1611_private_generation_receipt_v1' or not generation.get('complete'):raise ValueError('complete private generation receipt required')
    if generation['generator_source_sha256']!=sha(HERE/'generation.py') or generation['scientific_plan_sha256']!=sha(PLAN):raise ValueError('wrong generation source/condition')
    verify_manifest(generation['private_truth_files'])
    request_ref=generation['request']
    if sha(request_ref['path'])!=request_ref['sha256']:raise ValueError('generation request hash mismatch')
    request=read_json(request_ref['path'])
    if request.get('root_approved') is not True or request.get('parent_id')!=generation['parent_id'] or request.get('generator_source_sha256')!=sha(HERE/'generation.py') or request.get('science_plan_sha256')!=sha(PLAN):raise ValueError('generation request identity/condition differs')
    _,registered,_,_=scientific_inputs(generation['parent_id'])
    if generation.get('registered_seeds')!=registered['new_seed_values']:raise ValueError('generation seed registry differs')
    if sha(request['runtime_binding'])!=generation['runtime_binding_sha256']:raise ValueError('generation runtime binding changed')
    runtime=read_json(request['runtime_binding'])
    obs_ref=generation['observed_receipt']
    if sha(obs_ref['path'])!=obs_ref['sha256']:raise ValueError('observed receipt hash mismatch')
    observed=read_json(obs_ref['path']);verify_manifest(observed['files'])
    inference=read_json(args.inference_receipt)
    if inference.get('schema')!='prospective1611_inferred_bank_receipt_v1' or not inference.get('complete') or inference.get('parent_id')!=generation['parent_id'] or inference.get('observed_generation_receipt_sha256')!=obs_ref['sha256']:
        raise ValueError('inferred-bank receipt must bind exact shared observed realization')
    if set(inference.get('files',{}))!={'observed_trees','roster','observed_node_mapping'}:raise ValueError('exact inferred tree/roster/observed-ID mapping receipt required')
    verify_manifest(inference['files'])
    validate_inference_coordinates(inference,observed)
    if inference.get('true_genealogy_used_as_input') is not False or inference.get('map_derivation_receipt_sha256')!=observed['map_derivation_receipt_sha256']:raise ValueError('inference must declare observed-only input and exact common map')
    if inference['files']['observed_trees']['sha256']==generation['private_truth_files']['simulation.trees']['sha256']:raise ValueError('true simulation genealogy cannot be the inferred observed bank')
    meta=read_json(args.observed_metadata);targets=panel.read_jsonl(args.targets)
    roster=read_json(args.roster);bound=panel.Panel(roster)
    expected_bound=panel.bind_roster(read_json(observed['files']['roster']['path']),read_json(inference['files']['observed_node_mapping']['path']))
    if canonical(expected_bound)!=canonical(roster):raise ValueError('bound roster does not match receipted observed-node conversion')
    if [{k:v for k,v in r.items() if k!='node_id'} for r in roster['rows']]!=[{k:v for k,v in r.items() if k!='node_id'} for r in read_json(observed['files']['roster']['path'])['rows']]:raise ValueError('inference roster differs from exact generated roster')
    if inference['files']['observed_trees']['sha256']!=meta.get('observed_trees_sha256') or inference['files']['roster']['sha256']!=sha(args.roster):raise ValueError('inferred-bank features/roster mismatch')
    panel.validate_targets(targets,bound,parent_id=generation['parent_id'])
    if meta.get('schema')!='prospective1611_native_features_v1' or not meta.get('complete') or meta.get('parent_id')!=generation['parent_id'] or meta.get('targets_sha256')!=sha(args.targets) or meta.get('roster_sha256')!=sha(args.roster):raise ValueError('common target/observation metadata mismatch')
    files=generation['private_truth_files']
    import tskit
    qualify_loaded_runtime(runtime,{'tskit':tskit})
    ts=tskit.load(files['simulation.trees']['path'])
    demography=read_json(files['demography.json']['path']);truth_adapter.validate_cut_demography(demography)
    if max(r['right'] for r in targets)>ts.sequence_length:raise ValueError('targets exceed shared simulation coordinates')
    labels=truth_adapter.construct_common_labels(ts,read_json(files['simulation_node_mapping.json']['path']),read_json(files['population_mapping.json']['path']),targets,bound)
    out=Path(args.output_dir).resolve()
    observed_dir=Path(obs_ref['path']).resolve().parent
    if out==observed_dir or observed_dir in out.parents:raise ValueError('external founder labels must remain outside public observed directory')
    out.mkdir(parents=True,exist_ok=False)
    label_path=out/'labels.jsonl'
    with label_path.open('x') as f:
        for r in labels:f.write(json.dumps(r,separators=(',',':'))+'\n')
    contract={'schema':'prospective1611_external_supervision_contract_v1','complete':True,'parent_id':generation['parent_id'],'target_rows':len(targets),
              'definition':truth_adapter.DEFINITION,'truth_definition_sha256':canonical(truth_adapter.DEFINITION),'targets_sha256':sha(args.targets),
              'roster_sha256':sha(args.roster),'labels_sha256':sha(label_path),'observed_trees_sha256':meta['observed_trees_sha256'],
              'truth_input_sha256':{'simulation_trees':files['simulation.trees']['sha256'],'simulation_sample_mapping':files['simulation_node_mapping.json']['sha256'],
                  'population_mapping':files['population_mapping.json']['sha256'],'demography':files['demography.json']['sha256']},
              'demography':demography,'demography_semantic_sha256':canonical(demography),
              'constructor_source':{'path':str((ADAPTER/'truth_adapter.py').relative_to(ROOT)),'sha256':sha(ADAPTER/'truth_adapter.py')},
              'generation_receipt_sha256':sha(args.generation_receipt),'inference_receipt_sha256':sha(args.inference_receipt),
              'producer_stage':'separate_external_supervision_only','AP_positive_six_head_target_indices':[0,1,2,3],
              'AP_no_positive_target_indices_native_BAA_paths':[4,5],'AP_method_label':'AncestralPaths-derived C4 adapter','paper_reproduction':False}
    truth_adapter.validate_supervision_contract(contract,parent_id=generation['parent_id'],targets_path=args.targets,roster_path=args.roster,labels_path=label_path,observed_metadata=meta)
    write_json(out/'labels.jsonl.contract.json',contract)
    print(json.dumps({'complete':True,'parent_id':generation['parent_id'],'target_rows':len(labels),'labels_sha256':sha(label_path),'contract_sha256':sha(out/'labels.jsonl.contract.json')},sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('validate-request');s.add_argument('--request',required=True)
    g=sub.add_parser('generate');g.add_argument('--request',required=True)
    t=sub.add_parser('external-truth')
    for name in ('generation-receipt','inference-receipt','observed-metadata','targets','roster','output-dir'):t.add_argument('--'+name,required=True)
    args=p.parse_args()
    if args.command=='generate':generate(args.request)
    elif args.command=='external-truth':external_truth(args)
    else:
        validate_request(read_json(args.request));print(json.dumps({'metadata_request_valid':True,'numerical_paths_executed':False}))

if __name__=='__main__':main()
