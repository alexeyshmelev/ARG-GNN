"""Independent bounded synthetic CPU review; no scientific payloads or models."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,importlib.util,traceback
from pathlib import Path
from dataclasses import fields
import numpy as np
import torch
sys.path.insert(0,str(Path('src').resolve()))
from arggnn import american_c3_gnn_observations_isolated_v2 as original
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1,branch_concordance_support_v1

CANDIDATE=Path('research/reviews/C3_CPU_prepared_features_candidate_20261017_v1')
OWN=Path('research/reviews/C3_CPU_prepared_features_independent_20261017_v3')
EXPECTED={'HANDOFF.v3.json':'923bf0c72355cb535c771e7e6b9fa550c41fe42dc1c81b7916d7f66c7f0679db','MANIFEST.v3.json':'e610aaf8326bd5e1f8f324fb8ce8d229723806b27b3716071dda54b20bfdea7d'}
R={'schema':'C3_prepared_CPU_independent_fixtures_v3','checks':[],'counterexamples':[],'max_abs_error':0.,'fixture_input_bytes':0,'CPU_only':True,'model_forwards':0,'scientific_data_reads':0,'truth_reads':0,'guard_imports':0,'operational_actions':0}

def check(name,condition,**detail):
 R['checks'].append(dict(name=name,pass_=bool(condition),**detail))
 if not condition: raise AssertionError(name)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rejected(name,fn):
 try:fn()
 except (ValueError,TypeError,RuntimeError) as e:
  check(name,True,error_type=type(e).__name__,message=str(e));return
 check(name,False)
def lc(cap=2**20,entries=12):return c.ImmutableTreeLayoutCacheV1(max_bytes=cap,max_entries=entries)
def bc(layouts=None,cap=2**20,entries=12):return c.ImmutableBlockFeatureCacheV1(layout_cache=layouts or lc(),max_tensor_bytes=cap,max_entries=entries)
def ownarray(x,dtype):
 a=np.asarray(x,dtype=dtype);R['fixture_input_bytes']+=a.nbytes;return a

def make_tree(style,ages,seed=0):
 h=len(ages);p=[-1]*h;clock=list(ages);active=list(range(h))
 if style=='isolated':pass
 elif style=='star':
  root=len(p);p.append(-1);clock.append(max(clock)+1000)
  for a in active:p[a]=root
 elif style=='unary':
  root=len(p);p.append(-1);clock.append(max(clock)+1000)
  for a in active:p[a]=root
  upper=len(p);p.append(-1);clock.append(max(clock)+1000);p[root]=upper
 else:
  while len(active)>1:
   size=min(3 if style=='nonbinary' else 2,len(active));take=active[:size];active=active[size:]
   root=len(p);p.append(-1);clock.append(max(clock[a] for a in take)+1000)
   for a in take:p[a]=root
   active.append(root)
 p=np.asarray(p,dtype=np.int64);clock=np.asarray(clock,dtype=np.float32)
 perm=np.random.default_rng(seed).permutation(len(p));inv=np.argsort(perm)
 return NativeObservedTreeV1(ownarray(np.where(p[perm]<0,-1,inv[np.maximum(p[perm],0)]),np.int64),ownarray(clock[perm],np.float32),ownarray(inv[:h],np.int64),ownarray(np.arange(h),np.int64))
def bank_equal(name,a,b):
 count=0;maxerr=0.
 for f in fields(a):
  av,bv=getattr(a,f.name),getattr(b,f.name);aa=av if isinstance(av,tuple) else (av,);bb=bv if isinstance(bv,tuple) else (bv,)
  check(name+':arity:'+f.name,len(aa)==len(bb))
  for i,(x,y) in enumerate(zip(aa,bb)):
   eq=x.shape==y.shape and x.dtype==y.dtype and x.device.type==y.device.type=='cpu'
   if x.is_floating_point():
    eq=eq and x.dtype==torch.float32 and torch.allclose(x,y,rtol=2e-6,atol=2e-6)
    err=float((x-y).abs().max()) if x.numel() else 0.;maxerr=max(maxerr,err)
   else:eq=eq and torch.equal(x,y)
   check(name+':field:'+f.name+':'+str(i),eq)
  count+=1
 R['max_abs_error']=max(R['max_abs_error'],maxerr);check(name+':all_fields',count==21,max_abs_error=maxerr)

def fixture(style='binary',sites=25,seed=7,age_split=350.,tree_count=1):
 ages=ownarray([0,0,349,350,351,700],np.float32);h=len(ages)
 trees=[make_tree(s,ages,seed+i) for i,s in enumerate(([style]*tree_count))]
 k=ownarray(1+np.arange(sites)%4,np.int64);rng=np.random.default_rng(seed)
 g=np.empty((sites,h),dtype=np.int8)
 for j in range(sites):g[j]=rng.integers(-1,int(k[j]),h)
 if sites>0:g[0]=-1
 if sites>1:g[1]=0
 if sites>3:g[3]=np.arange(h)%4
 R['fixture_input_bytes']+=g.nbytes
 intervals=ownarray([[i*10,(i+1)*10] for i in range(tree_count)],np.float32)
 x=ownarray(np.linspace(0,tree_count*10-.25,sites,dtype=np.float32) if sites else [],np.float32)
 ref=ownarray([-1,0,1,2,-1,0],np.int64)
 return trees,intervals,g,k,x,ref,ages,dict(num_classes=3,age_split=age_split,query_columns=ownarray([4,0],np.int64))
def encode_args(data):return data[:7]
def options(data):return data[7]
def keyset(trees,prefix='t'):return [(prefix,i) for i in range(len(trees))]

def run():
 for name,sha in EXPECTED.items():check('pin:'+name,digest(CANDIDATE/name)==sha)
 manifest=json.loads((CANDIDATE/'MANIFEST.v3.json').read_text())
 refs=[manifest[k] for k in ['component_ref','contract_ref','test_ref','result_ref','delta_ref','diff_ref','preserved_v2_manifest_ref','v2_independent_BLOCKED_review_ref']]+manifest['source_dependencies']+manifest['exact_reused_fixture_refs']
 for ref in refs:check('auth:'+ref['path'],digest(Path(ref['path']))==ref['sha256'])
 global c
 spec=importlib.util.spec_from_file_location('independent_pinned_cpu_features',CANDIDATE/'prepared_cpu_features_v3.py');c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
 torch.set_num_threads(1)
 original_globals=dict(original.encode_block.__globals__)
 original_code=original.encode_block.__code__
 for style,sites,split,count in [('binary',25,350.,1),('nonbinary',32,None,2),('unary',17,350.,1),('isolated',19,None,2),('binary',0,350.,1)]:
  data=fixture(style,sites,age_split=split,tree_count=count)
  a,ar=original.encode_block(*encode_args(data),**options(data))
  b,br=c.encode_block_prepared_v1(*encode_args(data),tree_keys=keyset(data[0]),layout_cache=lc(),site_tile=3,**options(data))
  bank_equal('equivalence:'+style+':'+str(sites),a,b);check('receipt:'+style+':'+str(sites),ar==br)
 # Exact original code object is passed to FunctionType, while original module globals stay intact.
 captured=[];actual_fn=c.types.FunctionType
 def capture(code,namespace,*args):
  captured.append((code,namespace));return actual_fn(code,namespace,*args)
 c.types=type('PrivateTypes',(),{'FunctionType':staticmethod(capture)})
 data=fixture();c.encode_block_prepared_v1(*encode_args(data),tree_keys=keyset(data[0]),layout_cache=lc(),**options(data))
 c.types=__import__('types')
 check('encoder_exact_code_object',len(captured)==1 and captured[0][0] is original_code)
 differences={k for k in original_globals if captured[0][1].get(k) is not original_globals[k]}
 check('encoder_only_two_local_bindings',differences=={'_layout_with_isolated_samples','branch_concordance_support_v1'})
 check('original_globals_unpatched',all(original.encode_block.__globals__.get(k) is v for k,v in original_globals.items()))
 # Independent identity/sample-order join in direct branch helper.
 tree=make_tree('unary',np.zeros(6,dtype=np.float32),19);tree=NativeObservedTreeV1(tree.parent,tree.node_time,tree.sample_node,ownarray([42,11,91,5,37,60],np.int64))
 ids=ownarray([60,5,11,42,91,37],np.int64);g=ownarray([[0,1,2,3,-1,0],[1,1,-1,0,0,0],[-1]*6],np.int8);k=ownarray([4,2,1],np.int64)
 a=branch_concordance_support_v1(tree,g,k,ids);b=c.branch_support_batched_v1(tree,g,k,ids,layout_cache=lc(),tree_key=('ids',),site_tile=2)
 check('nontrivial_sample_identity_permutation',np.allclose(a,b,rtol=2e-6,atol=2e-6))
 # Direction-independent reuse is simulated by retrieving chronological blocks in opposite scan order.
 data=fixture();cache=bc(entries=3);keys=keyset(data[0]);call=lambda cache=cache,**extra:cache.get_or_prepare(('observation','plan',0),*encode_args(data),tree_keys=keys,**(options(data)|extra))
 a,receipt=call();b,receipt2=call();check('same_bank_identity_on_reverse_retrieval',a is b and cache.builds==1 and cache.hits==1)
 check('owned_bank_input_alias',a.reference_class.data_ptr()!=torch.as_tensor(data[5]).data_ptr())
 changed=list(data);changed[2]=data[2].copy();changed[2][0,0]=0
 rejected('block_input_drift',lambda:cache.get_or_prepare(('observation','plan',0),*changed[:7],tree_keys=keys,**options(data)))
 rejected('block_option_drift',lambda:call(age_split=None))
 a.genotype.numpy()[0,0]=0;rejected('bank_numpy_mutation',lambda:call())
 layouts=lc();record=layouts.get(data[0][0],('topology',));prop=record['propagation'][0][0];prop[0]=0 if int(prop[0]) else 1
 rejected('propagation_mutation',lambda:layouts.get(data[0][0],('topology',)))
 layouts=lc();record=layouts.get(data[0][0],('topology',));arr=record['layout'][0];arr.flags.writeable=True;arr[0]=0
 rejected('layout_writeable_override_mutation',lambda:layouts.get(data[0][0],('topology',)))
 layouts=lc();layouts.get(data[0][0],('topology',));modified=NativeObservedTreeV1(data[0][0].parent,data[0][0].node_time.copy(),data[0][0].sample_node,data[0][0].sample_id);modified.node_time[0]+=1
 rejected('tree_time_input_drift',lambda:layouts.get(modified,('topology',)))
 rejected('layout_prospective_cap',lambda:lc(1).get(data[0][0],('cap',)))
 rejected('bank_prospective_cap',lambda:bc(cap=1).get_or_prepare(('cap',),*encode_args(data),tree_keys=keys,**options(data)))
 rejected('branch_working_cap',lambda:c.branch_support_batched_v1(data[0][0],data[2],data[3],np.arange(6),layout_cache=lc(),tree_key=('cap',),max_working_bytes=1))
 # LRU counts and independently computed retained byte totals.
 layouts=lc(entries=1)
 for i in range(3):layouts.get(make_tree('star',np.zeros(6,dtype=np.float32),i),('lru',i))
 check('layout_lru_entry_and_bytes',len(layouts.values)==1 and layouts.bytes==sum(r['bytes'] for r in layouts.values.values()) and layouts.bytes<=layouts.max_bytes)
 cache=bc(entries=1)
 for i in range(3):cache.get_or_prepare(('lru',i),*encode_args(data),tree_keys=keys,**options(data))
 check('bank_lru_entry_and_bytes',len(cache.values)==1 and cache.bytes==sum(r['bytes'] for r in cache.values.values()) and cache.bytes<=cache.max_tensor_bytes)
 # Repeated multiallelic sites are within the documented 4096-site block limit.
 tree=make_tree('star',np.zeros(3,dtype=np.float32),41);g=ownarray(np.tile(np.asarray([0,1,2],dtype=np.int8),(4096,1)),np.int8);k=ownarray(np.full(4096,3),np.int64);ids=ownarray(np.arange(3),np.int64)
 a=branch_concordance_support_v1(tree,g,k,ids)
 for tile in (1,32):
  b=c.branch_support_batched_v1(tree,g,k,ids,layout_cache=lc(),tree_key=('repeat',),site_tile=tile)
  err=float(np.abs(a-b).max());passed=bool(np.allclose(a,b,rtol=2e-6,atol=2e-6))
  R['checks'].append(dict(name='4096_repeated_triallelic_tile_'+str(tile),pass_=passed,max_abs_error=err))
  if not passed:R['counterexamples'].append(dict(name='4096_repeated_triallelic_tile_'+str(tile),tree_parent=tree.parent.tolist(),sample_node=tree.sample_node.tolist(),sites=4096,genotype_row=[0,1,2],allele_count=3,site_tile=tile,original=a.tolist(),candidate=b.tolist(),max_abs_error=err,rtol=2e-6,atol=2e-6))
 # Realistic haplotype count and only three SNPs, using synthetic inputs.
 ages=ownarray(np.zeros(1600,dtype=np.float32),np.float32);tree=make_tree('binary',ages,75)
 rng=np.random.default_rng(75);g=ownarray(np.stack([rng.integers(-1,k,1600) for k in (2,3,4)]),np.int8)
 k=ownarray([2,3,4],np.int64);interval=ownarray([[0,10]],np.float32);x=ownarray([0,4,9],np.float32)
 ref=ownarray(np.arange(1600)%3,np.int64);ref[-1]=-1
 data=([tree],interval,g,k,x,ref,ages,dict(num_classes=3,age_split=350.))
 a,ar=original.encode_block(*data[:7],**data[7]);b,br=c.encode_block_prepared_v1(*data[:7],tree_keys=[('realistic1600',)],layout_cache=lc(),**data[7])
 bank_equal('1600haplotypes_3199nodes_3sites',a,b);check('1600haplotypes_receipt',ar==br)
 check('fixture_inputs_below_16MiB' ,R['fixture_input_bytes']<16*2**20,total_bytes=R['fixture_input_bytes'])
 R['status']='BLOCKED' if R['counterexamples'] else 'SOURCE_PASS'

try:run()
except Exception:R['fatal_error']=traceback.format_exc();R['status']='HARNESS_OR_COMPONENT_FAILURE';print(R['fatal_error'])
(OWN/'fixture_results.v3.json').write_text(json.dumps(R,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':R.get('status'),'checks':len(R['checks']),'counterexamples':R['counterexamples'],'fixture_input_bytes':R['fixture_input_bytes'],'fatal_error':R.get('fatal_error')},indent=2))
