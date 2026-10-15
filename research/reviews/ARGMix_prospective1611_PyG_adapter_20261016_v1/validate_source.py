#!/usr/bin/env python3
"""Compile source and run only AST-extracted pure metadata helpers on literals.

No wrapper import, numerical module, model, genotype, tree or live framework call.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt');args=p.parse_args()
    scripts=sorted(HERE.glob('*.py'))
    for path in scripts:compile(path.read_text(),str(path),'exec')
    pins=json.loads((HERE/'source_pins.v1.json').read_text())
    for r in pins['sources']:
        path=ROOT/r['path'];assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'],r['path']
    tree=ast.parse((HERE/'pyg_adapter.py').read_text())
    names={'require','digest','canonical','class_roles','target_indices','dense_native_forecast','compare_fit_inputs'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={'hashlib':hashlib,'json':json}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'LABELS','HAPS','QUERIES','PURE_COUNTS'}:ns[n.targets[0].id]=ast.literal_eval(n.value)
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pure metadata helpers>','exec'),ns)
    checks=[]
    def reject(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('invalid metadata accepted')
    roster=json.loads((ROOT/'research/reviews/ARGMix_prospective1611_native_adapters_20261016_v1/prospective_roster.unbound.v1.json').read_text())
    refs,ages,queries=ns['class_roles'](roster)
    assert len(refs)==3222 and len(queries)==2972 and [refs.count(c) for c in range(4)]==[44,26,96,84]
    checks.append('3222 observed identities,2972 queries and pure founder counts44/26/96/84')
    mixed=[i for i,r in enumerate(roster['rows']) if r['cohort'] in ('NF','Yam')]
    assert all(refs[i]==-1 and i in queries for i in mixed)
    assert next(r['age_generations'] for r in roster['rows'] if r['haplotype_id']=='NF_0158_h0')==185
    checks.append('mixed NF/Yam remain unknown; separately named older-NF proxy retains age185')
    nf=next(r['haplotype_id'] for r in roster['rows'] if r['cohort']=='NF')
    yam=next(r['haplotype_id'] for r in roster['rows'] if r['cohort']=='Yam')
    targets=[{'tree_index':1,'left':10.,'right':20.,'haplotype_id':yam},{'tree_index':0,'left':0.,'right':10.,'haplotype_id':nf}]
    result=ns['target_indices'](targets,roster,[(0.,10.),(10.,20.)])
    slots={roster['rows'][c]['haplotype_id']:q for q,c in enumerate(queries)}
    assert result==[(1,slots[yam]),(0,slots[nf])]
    checks.append('target order joins by explicit tree/haplotype, without grid interpolation')
    bad=copy.deepcopy(targets);bad[0]['right']=21.
    reject(lambda:ns['target_indices'](bad,roster,[(0.,10.),(10.,20.)]))
    reject(lambda:ns['target_indices']([targets[0],targets[0]],roster,[(0.,10.),(10.,20.)]))
    bad=copy.deepcopy(targets);bad[0]['haplotype_id']='Ana_0000_h0'
    reject(lambda:ns['target_indices'](bad,roster,[(0.,10.),(10.,20.)]))
    checks.append('conflicting intervals, duplicate pairs and founder-as-query rejected')
    for n,c in [(1,1),(1024,1),(1025,2),(1536,2),(1537,3)]:
        f=ns['dense_native_forecast'](n,7)
        assert f['native_chunks_1024_stride512']==7*c and f['final_dense_chunk_tensor_bytes']==7*c*672768
        assert f['converted_context_tensor_bytes_at_least']==n*7*640
    reject(lambda:ns['dense_native_forecast'](0,7))
    checks.append('native dense1024/stride512 integer count and657-byte/row tensor accounting')
    sig={'parent_id':'literal','targets_sha256':'a'*64,'roster_sha256':'b'*64,'observed_trees_sha256':'c'*64,'labels_sha256':'d'*64,'supervision_contract_sha256':'e'*64,'truth_definition_sha256':'f'*64}
    common={'train_shared_signatures':[sig],'val_shared_signatures':[sig],'logical_train_targets':500000,'logical_val_targets':100000}
    native={**common,'schema':'prospective1611_native_fit_summary_v1','method':'argmix'}
    pyg=copy.deepcopy({**common,'schema':'prospective1611_pyg_fit_summary_v1','method':'pyg'})
    docs={'native':native,'pyg':pyg};ns.update(read=docs.__getitem__,verify_sources=lambda:len(pins['sources']))
    assert ns['compare_fit_inputs']('native','pyg')['same_supervised_data_verified']
    pyg['train_shared_signatures'][0]['labels_sha256']='0'*64
    reject(lambda:ns['compare_fit_inputs']('native','pyg'))
    checks.append('new native/PyG common-data comparer requires identical external supervision')
    top=[]
    for n in tree.body:
        if isinstance(n,ast.Import):top.extend(x.name for x in n.names)
        if isinstance(n,ast.ImportFrom):top.append(n.module)
    assert not {'torch','numpy','tskit','torch_geometric','tensorflow','ancpath'}&set(top)
    forbidden={'torch','numpy','tskit','torch_geometric','tensorflow','ancpath'}&set(sys.modules)
    assert not forbidden
    checks.append('wrapper has no top-level numerical imports; validator loaded no numerical libraries')
    config=json.loads((HERE/'finite_fit_config.v1.json').read_text())
    assert config['execution_authorized'] is False and config['model']=={'hidden_dim':32,'expected_haplotypes':3222,'site_chunk_size':256,'checkpoint_sites':True}
    for command in ('prepare','fit','predict'):
        r=json.loads((HERE/(command+'_request.template.inactive.json')).read_text())
        assert r['root_approved'] is False and r['wrapper_source_sha256']==sha(HERE/'pyg_adapter.py') and r['source_pins_sha256']==sha(HERE/'source_pins.v1.json')
    checks.append('finite fit and exact source-bound request templates remain inactive')
    receipt={'schema':'prospective1611_pyg_source_validation_v1','passed':True,'source_syntax_compiles':len(scripts),'frozen_source_refs_verified':len(pins['sources']),
             'pure_metadata_checks':len(checks),'checks':checks,'source_file_sha256':{p.name:sha(p) for p in scripts},
             'wrapper_or_numerical_modules_imported':False,'models_real_genotypes_trees_or_training_executed':False,
             'actual_map_payload_read':False,'map_derivation_receipt_metadata_only':True,'jobs_admissions_guards_installs_Git_or_probes':False}
    if args.receipt:
        with Path(args.receipt).open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(receipt,indent=2,sort_keys=True))
if __name__=='__main__':main()
