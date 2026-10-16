"""Prospective ancient-panel inputs for untouched current-author ARGMix backend.
Source only. Root owns numerical execution, resource admission, native build and fit.
The old100k plan is retained as sensitivity. It is not the primary native fit.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import zipfile

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=Path(__file__).resolve().parent
AUTHOR=ROOT/'.runtime/author_sources/argmix_c2481248eb0d'
CONTRACT=HERE/'method_contract.proposed.v1.json'
GENERATOR_REF={'path':'research/reviews/ARGMix_prospective1611_generator_source_20261016_v1/prospective1611_iid_v1.py','sha256':'7e6dc35934252f3d404da3b9c612e4580877e752de41e5f88fe2a58418633e0a'}
CLASSES=['Ana','CHG','WHG','EHG']
COMMIT='c2481248eb0de5fda659d68f59c51e0e27becd20'

def need(ok,message):
    if not ok:raise ValueError(message)

def path(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and ROOT in p.parents,'Canonical repository path required')
    return p

def sha(value):
    h=hashlib.sha256()
    with path(value).open('rb') as f:
        for block in iter(lambda:f.read(4*2**20),b''):h.update(block)
    return h.hexdigest()

def ref(value):
    p=path(value)
    return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)

def checked(record):
    p=path(record['path'])
    need(p.is_file() and not p.is_symlink() and sha(p)==record['sha256'],'Pinned artifact changed: '+str(p))
    return p

def metadata(value):
    p=path(value);need(p.suffix=='.json' and p.stat().st_size<=8*2**20,'Bounded JSON metadata')
    return json.loads(p.read_text())

def contract():
    c=metadata(CONTRACT)
    need(c['schema']=='ARGMix_prospective1611_primary_native_contract_v1','Exact primary contract')
    for r in c['fixed_refs'].values():checked(r)
    return c

def fresh(value,guard):
    p=path(value);need(callable(guard) and p.parent.is_dir() and not p.exists(),'Fresh scope and actual root guard')
    guard(16384);p.mkdir();return p

def save_json(value,obj,guard):
    data=(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    need(len(data)<=8*2**20,'Bounded JSON artifact');guard(len(data))
    with path(value).open('xb') as f:f.write(data)
    guard(0);return ref(value)

def write_bytes(value,data,guard):
    guard(len(data))
    with path(value).open('xb') as f:f.write(data)
    guard(0);return ref(value)

def load_generator():
    p=checked(GENERATOR_REF)
    spec=importlib.util.spec_from_file_location('_prospective1611_primary_generator',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def materialize_author_copy(output,*,guard):
    c=contract();inv=metadata(checked(c['fixed_refs']['author_inventory']))
    output=fresh(output,guard);guard(sum(r['bytes'] for r in inv['files'].values())+2**20)
    for relative,r in inv['files'].items():
        source=AUTHOR/relative;need(sha(source)==r['sha256'],'Pinned author source drift')
        dest=output/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(source.read_bytes())
    build=output/'owned_build';build.mkdir()
    for n in ['cargo_home','target','tmp']:(build/n).mkdir()
    value=dict(schema='ARGMix_prospective1611_primary_untouched_copy_v1',author_commit=COMMIT,
        source_copy=str(output.relative_to(ROOT)),author_inventory_ref=c['fixed_refs']['author_inventory'],
        modified_author_files=[],native_python=c['native_python'],
        build_argv=['cargo','build','--manifest-path',str(output/'src/rust/ancpath/Cargo.toml'),
                    '--locked','--release','--no-default-features','--bin','build_training'],
        build_env=dict(CARGO_HOME=str(build/'cargo_home'),CARGO_TARGET_DIR=str(build/'target'),
            TMPDIR=str(build/'tmp'),PYO3_PYTHON=str(ROOT/c['native_python']),CUDA_VISIBLE_DEVICES=''),
        expected_binary=str(build/'target/release/build_training'),
        build_output_ceiling_bytes=8*2**30,build_and_runtime_qualification_pending=True)
    save_json(output/'untouched_copy.json',value,guard);guard(0);return value

def _copy(record):
    info=metadata(checked(record));c=contract()
    need(info['schema']=='ARGMix_prospective1611_primary_untouched_copy_v1'
        and info['author_inventory_ref']==c['fixed_refs']['author_inventory']
        and info['modified_author_files']==[],'Untouched pinned native copy')
    source=path(info['source_copy']);inv=metadata(checked(info['author_inventory_ref']))
    for name,r in inv['files'].items():
        need(sha(source/name)==r['sha256'],'Untouched author copy changed')
    return source,info

def _target_rows(support):
    with checked(support['targets_ref']).open() as f:
        header=next(f).rstrip().split('\t')
        rows=[dict(zip(header,line.rstrip().split('\t'))) for line in f]
    need(len(rows)==100000 and [int(r['target_row']) for r in rows]==list(range(100000)),
         'Original frozen100k sensitivity/evaluation order')
    return rows

def prepare_observed_parent(generator_config,parent_id,parent_output,inferred_output,support_output,output,*,guard,resource_check):
    """Observed-only native panel and common-support/coverage. No census or labels."""
    c=contract();need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only observed bridge')
    need(callable(resource_check),'Actual root supervisor callback')
    g=load_generator();cfg,parent,obs,observation,rows=g.observed_assets(generator_config,parent_id,parent_output)
    bridge=metadata(path(inferred_output)/'completed.json');support=metadata(path(support_output)/'completed.json')
    need(bridge['parent_id']==support['parent_id']==parent_id and
        bridge['config_ref']==support['config_ref']==ref(generator_config),'Exact same prospective parent')
    need(support['inferred_bridge_ref']==ref(path(inferred_output)/'completed.json') and
        support['truth_or_labels_read'] is False,'Original frozen observed-only targets')
    commands=metadata(checked(bridge['native_commands_ref']))
    need(commands['parent_id']==parent_id and commands['observation_manifest_ref']==ref(obs/'observation_manifest.json'),
         'Same observed Relate inputs')
    native_inputs=commands['input_refs']
    import numpy as np
    import tskit
    native=tskit.load(checked(bridge['native_tree_ref']))
    joined=metadata(checked(bridge['roster_ref']))
    need(len(rows)==3222 and [r['sample_index'] for r in rows]==list(range(3222)),'Exact explicit haplotype ordering')
    need(list(map(int,native.samples()))==list(range(3222))==[r['inferred_tree_node_id'] for r in joined],
         'Native NPZ resolver requires inferred sample nodes=2*i,2*i+1. Do not remap silently')
    need(all(all(j[k]==v for k,v in r.items()) for j,r in zip(joined,rows)),'Stable identity/role join')
    sample_path=checked(native_inputs['input.sample'])
    names=[line.split()[0] for line in sample_path.read_text().splitlines()[2:] if line.strip()]
    need(names==[r['sample_id'] for r in rows[::2]] and len(names)==1611,'Native sample_ids and diploid order')
    positions=np.fromfile(checked(observation['observed_files']['positions.u4']),dtype='<u4').astype(np.int64)
    need(len(positions)==observation['sites'] and np.all(np.diff(positions)>0),'All original observed SNPs')
    native_positions=positions+1
    need(np.array_equal(native.tables.sites.position,native_positions),'Raw native position=p0+1 identity')
    starts=[];ends=[];eligible=0
    for tree in native.trees():
        lo,hi=float(tree.interval.left),float(tree.interval.right)
        starts.append(lo);ends.append(hi)
        if 0<tree.index<native.num_trees-1:
            left=np.searchsorted(native_positions,math.floor(lo),side='left')
            right=np.searchsorted(native_positions,math.ceil(hi),side='left')
            eligible+=int(right>left)
    targets=_target_rows(support)
    query_ids=[r['sample_index'] for r in rows if r['query_eligible']]
    reference_ids=[r['sample_index'] for r in rows if r['reference_eligible']]
    need(len(query_ids)==2972 and len(reference_ids)==594,'Native1611 donor/query panel counts')
    native_midpoints=np.empty(100000,dtype=np.float64)
    for i,t in enumerate(targets):
        focal=rows[int(t['sample_index'])]
        need(focal['query_eligible'] and
            [t['sample_id'],int(t['haplotype']),t['cohort']]==[focal['sample_id'],focal['haplotype'],focal['observed_cohort']],
            'Exact shared target focal identity')
        native_midpoints[i]=float(t['rebased_midpoint_bp'])+1.
    indexes=np.searchsorted(np.asarray(starts),native_midpoints,side='right')-1
    need(np.all(indexes>=0) and np.all(indexes<native.num_trees) and
        np.all(native_midpoints<np.asarray(ends)[indexes]),'Exact shared midpoint in native interval')
    supported=(indexes>0)&(indexes<native.num_trees-1)
    output=fresh(output,guard)
    reference_text=''.join(f'{r["sample_index"]} {r["observed_cohort"]} {r["age_generations"]}\n' for r in rows)
    references_ref=write_bytes(output/'references.tsv',reference_text.encode(),guard)
    mask_ref=write_bytes(output/'native_supported.u1',supported.astype(np.uint8).tobytes(),guard)
    native_index_ref=write_bytes(output/'native_target_tree_index.i8',indexes.astype('<i8').tobytes(),guard)
    # Same numeric native position/cM values. Only header names are standardized.
    native_map=checked(native_inputs['input.map']);map_lines=native_map.read_text().splitlines()
    map_body='\n'.join(map_lines[1:])+'\n'
    map_ref=write_bytes(output/'native_header.map',
        ('position COMBINED_rate(cM/Mb) Genetic_Map(cM)\n'+map_body).encode(),guard)
    chunks_per_query=0 if eligible==0 else 1+max(0,math.ceil((eligible-1024)/512))
    padded_rows=len(query_ids)*chunks_per_query*1024
    true_rows=len(query_ids)*eligible
    by_cohort={}
    for t,ok in zip(targets,supported):
        entry=by_cohort.setdefault(t['cohort'],dict(original=0,supported=0))
        entry['original']+=1;entry['supported']+=int(ok)
    value=dict(schema='ARGMix_prospective1611_primary_observed_parent_v1',parent_id=parent_id,split=parent['split'],
        source_ref=ref(__file__),contract_ref=ref(CONTRACT),generator_config_ref=ref(generator_config),
        observed_manifest_ref=ref(obs/'observation_manifest.json'),inferred_bridge_ref=ref(path(inferred_output)/'completed.json'),
        native_commands_ref=bridge['native_commands_ref'],original_shared_support_ref=ref(path(support_output)/'completed.json'),
        native_tree_ref=bridge['native_tree_ref'],native_sample_ref=native_inputs['input.sample'],
        native_haps_ref=native_inputs['input.haps'],native_map_ref=map_ref,references_ref=references_ref,
        native_target_index_ref=native_index_ref,native_supported_ref=mask_ref,
        sample_manifest_ref=bridge['roster_ref'],query_count=2972,reference_count=594,variants=len(positions),
        native_tree_count=int(native.num_trees),native_observed_snp_eligible_tree_count=eligible,
        coordinate='raw Relate-converted tree and haps: original rebased0 position+1',
        native_edge_policy='exclude first and last raw native trees',
        rank='unchanged FastSubgraphEngine uncapped absolute quantized integer-time difference, native HashMap-vector ties',
        native_reference_order='unmodified native runtime HashMap-derived vector, not overwritten with roster order',
        common_support=dict(original_targets=100000,supported_targets=int(supported.sum()),
            excluded_by_native_edge_support=int((~supported).sum()),coverage_by_cohort=by_cohort,
            eligibility_uses_only_observed_intervals=True,truth_or_labels_read=False,
            all_methods_use_identical_supported_target_mask=True),
        capacity=dict(condition='exact if every observed SNP truth class is valid for every query, verified by truth bridge before native build',
            full_supervised_rows=true_rows,native_chunks=len(query_ids)*chunks_per_query,
            padded_chunk_rows=padded_rows,archive_array_payload_bytes=657*padded_rows,
            archive_ceiling_bytes=657*padded_rows+2**20,
            raw_truth_NPZ_label_payload_bytes=2*len(positions)*3222,
            builder_row_payload_plus_x64_struct_bytes_estimate=728*true_rows,
            builder_dense_array_payload_bytes=657*padded_rows,
            train_loader_loads_full_TRAIN_and_VAL_archives=True,
            source_count_is_not_root_capacity_admission=True),
        primary_full_supervision_differs_from_100k_budget=True,
        original100k_plan_replaced=False,truth_or_labels_read=False,original_paper_reproduction=False)
    save_json(output/'observed_plan.json',value,guard)
    resource_check('after_primary_ARGMix_observed_parent_plan');return value

def build_truth_npz(plan_record,parent_output,output,*,guard,resource_check):
    """Format-only full observed-SNP census labels for native TRAIN/VAL resolver."""
    c=contract();need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only truth-format bridge')
    need(callable(resource_check),'Actual root supervisor callback')
    plan=metadata(checked(plan_record))
    need(plan['schema']=='ARGMix_prospective1611_primary_observed_parent_v1' and
        plan['source_ref']==ref(__file__) and plan['contract_ref']==ref(CONTRACT) and
        plan['split'] in ('train','val'),'Full primary labels permitted TRAIN/VAL only, no IID metric access')
    g=load_generator();cfg,parent,obs,observation,rows=g.observed_assets(
        checked(plan['generator_config_ref']),plan['parent_id'],parent_output)
    need(ref(obs/'observation_manifest.json')==plan['observed_manifest_ref'],'Exact observed SNP source')
    truth_manifest_path=path(parent_output)/'truth/truth_manifest.json'
    truth_meta=metadata(truth_manifest_path)
    need(truth_meta['parent_id']==plan['parent_id'] and truth_meta['config_ref']==plan['generator_config_ref']
        and truth_meta['class_order']==CLASSES,'Exact census truth schema and class order')
    import numpy as np
    import tskit
    import msprime
    truth=tskit.load(checked(truth_meta['ancestry_file']))
    sample_nodes=np.asarray(truth_meta['sample_index_to_raw_node_id'],dtype=np.int64)
    need(np.array_equal(sample_nodes,truth.samples()) and len(sample_nodes)==3222,'Raw sample identity, never ages as IDs')
    positions=np.fromfile(checked(observation['observed_files']['positions.u4']),dtype='<u4').astype(np.int64)
    need(len(positions)==plan['variants'] and np.all(np.diff(positions)>0),'Full exact original observed sites')
    names=np.asarray([r['sample_id'] for r in rows[::2]],dtype=str)
    need(len(names)==1611 and all(rows[2*i]['sample_id']==rows[2*i+1]['sample_id']
        and rows[2*i]['haplotype']==0 and rows[2*i+1]['haplotype']==1 for i in range(1611)),
        'Exact diploid h0/h1 interleaving')
    flags=truth.tables.nodes.flags;times=truth.tables.nodes.time;pops=truth.tables.nodes.population
    census=(flags&msprime.NODE_IS_CEN_EVENT)!=0
    population_to_class={int(k):CLASSES.index(v) for k,v in truth_meta['population_id_to_name'].items() if v in CLASSES}
    output=fresh(output,guard);estimate=2*len(positions)*3222+8*len(positions)+2**20
    guard(estimate);npz_path=output/'native_truth.npz'
    written_rows=0
    with zipfile.ZipFile(npz_path,mode='x',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        with archive.open('truth__hap_labels.npy','w',force_zip64=True) as stream:
            np.lib.format.write_array_header_1_0(stream,dict(
                descr=np.lib.format.dtype_to_descr(np.dtype('<i2')),fortran_order=False,shape=(len(positions),3222)))
            for tree in truth.trees():
                lo=np.searchsorted(positions,tree.interval.left,side='left')
                hi=np.searchsorted(positions,tree.interval.right,side='left')
                if hi==lo:continue
                resource_check('before_native_full_SNP_truth_block')
                need(lo==written_rows,'All observed SNPs assigned once, right-continuous census truth')
                nodes=sample_nodes.copy()
                unresolved=~census[nodes]
                while np.any(unresolved):
                    nodes[unresolved]=tree.parent_array[nodes[unresolved]]
                    need(np.all(nodes>=0),'Every observed haplotype has census ancestor')
                    unresolved=~census[nodes]
                need(np.all(times[nodes]==truth_meta['census_time']),'Exact census350 label semantics')
                classes=np.asarray([population_to_class.get(int(p),-1) for p in pops[nodes]],dtype='<i2')
                need(np.all((classes>=0)&(classes<4)),'All full native SNP/query labels valid, no truth-based row pruning')
                row=classes.tobytes();batch=max(1,(2**20)//len(row))
                for start in range(lo,hi,batch):
                    stream.write(row*min(batch,hi-start))
                written_rows=int(hi);guard(0)
        for name,array in [('variant_pos',positions+1),('truth__sample_ids',names),('ancestry_names',np.asarray(CLASSES,dtype=str))]:
            with archive.open(name+'.npy','w',force_zip64=True) as stream:
                np.lib.format.write_array(stream,array,allow_pickle=False)
    need(written_rows==len(positions) and npz_path.stat().st_size<=estimate,'Full label coverage and counted NPZ ceiling')
    guard(0)
    value=dict(schema='ARGMix_prospective1611_primary_native_truth_format_v1',parent_id=plan['parent_id'],split=plan['split'],
        source_ref=ref(__file__),contract_ref=ref(CONTRACT),observed_plan_ref=plan_record,
        truth_manifest_ref=ref(truth_manifest_path),truth_npz_ref=ref(npz_path),
        truth_dataset='truth',class_order=CLASSES,sample_ids=list(map(str,names)),
        variant_coordinate='all original observed mutation positions rebased0+1, identical to native .haps',
        hap_labels_shape=[len(positions),3222],hap_label_dtype='little-endian int16',
        all_classes_valid=True,native_label_rule_unchanged='first_snp, no purity filtering',
        labels_are_supervision_only=True,model_inputs_unchanged=True,genotype_states_modified=False,
        IID_labels_read=False,source_count_is_not_root_capacity_admission=True)
    save_json(output/'truth_format.json',value,guard);return value

def _merge(base,overlay):
    result=dict(base)
    for key,value in overlay.items():
        result[key]=_merge(result[key],value) if isinstance(result.get(key),dict) and isinstance(value,dict) else value
    return result

def prepare_native_backend(train_plan_records,val_plan_records,truth_records,copy_record,output,*,guard):
    """Native configs/argv only. No native build, archive export, import or fit."""
    c=contract();source,copy=_copy(copy_record)
    plans=[metadata(checked(r)) for r in train_plan_records+val_plan_records]
    need([p['parent_id'] for p in plans]==['published_recon_train_%03d'%i for i in range(5)]+['published_recon_val_000'],
         'Primary fiveTRAIN and oneVAL, unchanged parent data')
    need(len(truth_records)==6,'Full native SNP supervision for six exact TRAIN/VAL parents')
    truths=[metadata(checked(r)) for r in truth_records]
    for plan,truth,record in zip(plans,truths,train_plan_records+val_plan_records):
        need(truth['observed_plan_ref']==record and truth['all_classes_valid'] and truth['class_order']==CLASSES,
            'Validated full SNP labels, no selection by truth')
    need(len({p['native_map_ref']['sha256'] for p in plans})==1,'Same numeric native-coordinate map across parents')
    import yaml
    template=yaml.safe_load((source/'templates/forward_blank/argmix.yaml').read_text())
    tree=ast.parse((source/'src/argmix_project.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.AnnAssign) and getattr(n.target,'id',None)=='DEFAULT_ADVANCED')
    defaults=ast.literal_eval(node.value)
    resolved=_merge(defaults,template)
    output=fresh(output,guard)
    pairs=[]
    for plan,truth in zip(plans,truths):
        pairs.append(dict(rel=str(checked(plan['native_tree_ref'])),truth_npz=str(checked(truth['truth_npz_ref'])),
            relate_sample=str(checked(plan['native_sample_ref'])),relate_haps=str(checked(plan['native_haps_ref'])),
            references=str(checked(plan['references_ref']))))
    # YAML accepts JSON, preserving exact structured values without shell interpolation.
    train_pairs=write_bytes(output/'train_pairs.yaml',(json.dumps(pairs[:5],indent=2)+'\n').encode(),guard)
    val_pairs=write_bytes(output/'val_pairs.yaml',(json.dumps(pairs[5:],indent=2)+'\n').encode(),guard)
    all_rows=metadata(checked(plans[0]['sample_manifest_ref']))
    ref_pops=sorted({r['observed_cohort'] for r in all_rows if r['reference_eligible']})
    query_pops=sorted({r['observed_cohort'] for r in all_rows if r['query_eligible']})
    cfg={k:resolved[k] for k in ('model','joint_model','processing','inference','testing','wandb','relate','runtime')}
    cfg['model']=dict(cfg['model'],num_classes=4,log_scale_tmrca=True)
    cfg['joint_model']=dict(cfg['joint_model'],base_encoder='tmrca_transformer')
    cfg['processing']=dict(cfg['processing'],backend_name='fast')
    cfg['populations']=dict(reference=ref_pops,predict=query_pops,class_labels={i:v for i,v in enumerate(CLASSES)})
    cfg['supervision']=dict(label_source='npz_truth',truth_dataset='truth',truth_label_mode='first_snp',
        truth_min_label_purity=None,truth_ancestry_rename={},truth_collect_diagnostics=False)
    cfg['paths']=dict(train_pairs=str(checked(train_pairs)),val_pairs=str(checked(val_pairs)),
        references=str(checked(plans[0]['references_ref'])),save_dir=str(output/'checkpoints'),
        wandb_logs_dir=str(output/'wandb'))
    cfg['relate']=dict(cfg['relate'],recomb_map=str(checked(plans[0]['native_map_ref'])))
    cfg['wandb']=dict(cfg['wandb'],enabled=False)
    need(cfg['joint_model']['train']['num_epochs']==20 and cfg['processing']['exclude_edge_trees'] is True,
         'Preserve actual author template20 epochs and native edge policy')
    need(cfg['joint_model']['train']['max_train_subgraphs'] is None and
        cfg['joint_model']['train']['max_val_subgraphs'] is None,'No native subgraph/query cap')
    backend_ref=write_bytes(output/'backend_config.yaml',(json.dumps(cfg,indent=2)+'\n').encode(),guard)
    train_archive=output/'train.pt';val_archive=output/'val.pt'
    binary=path(copy['expected_binary'])
    train_bytes=sum(p['capacity']['archive_ceiling_bytes'] for p in plans[:5])
    val_bytes=plans[5]['capacity']['archive_ceiling_bytes']
    value=dict(schema='ARGMix_prospective1611_primary_native_job_plan_v1',source_ref=ref(__file__),contract_ref=ref(CONTRACT),
        untouched_copy_ref=copy_record,observed_plan_refs=train_plan_records+val_plan_records,truth_format_refs=truth_records,
        backend_config_ref=backend_ref,train_pairs_ref=train_pairs,val_pairs_ref=val_pairs,
        build_training_argv=[[str(binary),'--config',str(checked(backend_ref)),'--split',split,'--output',str(dest)]
            for split,dest in [('train',train_archive),('val',val_archive)]],
        fit_argv=[str(ROOT/c['native_python']),str(HERE/'native_train_launcher_v1.py'),
            str(source/'src/train_joint_model.py'),'--config',str(checked(backend_ref)),
            '--train-pt',str(train_archive),'--val-pt',str(val_archive),
            '--checkpoint-output',str(output/'trained_checkpoint.path')],
        fit_env=dict(CUDA_VISIBLE_DEVICES='0',TMPDIR=str(path(copy['source_copy'])/'owned_build/tmp')),
        native_hardcoded_fit_seed=50,template_max_epochs=20,native_patience=2,
        CPU_preparation_env=dict(CUDA_VISIBLE_DEVICES=''),
        capacity=dict(TRAIN_archive_ceiling_bytes=train_bytes,VAL_archive_ceiling_bytes=val_bytes,
            all_archives_ceiling_bytes=train_bytes+val_bytes,
            CPU_train_tensor_payload_bytes=sum(p['capacity']['archive_array_payload_bytes'] for p in plans),
            note='Additional process/library/allocator/truth-array overhead must be admitted by root. No claim these arrays fit current capacity.'),
        training_supervision='all native eligible observed-SNP rows, first_snp, default TRAIN nearest-drop1 and VAL0',
        shared100k_budget_sensitivity_not_replaced=True,
        primary_label='current-author unchanged backend on prospective ancient panel',
        outer_forward_CLI_and_original_paper_reproduction=False,
        all_builds_archives_and_fits_pending=True)
    save_json(output/'native_job_plan.json',value,guard);return value
