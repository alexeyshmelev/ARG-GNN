"""Bounded CPU native-format smoke for the readable staged Relate driver."""
from pathlib import Path
import json
import numpy as np
from arggnn.benchmarking.paper_argmix import paper_argmix_generator as g
from arggnn.benchmarking.paper_argmix.paper_argmix_relate import run_parallel
from arggnn.benchmarking.paper_argmix.paper_argmix_features import ScopeGuard

base=g.ROOT/'research/benchmark_rebuild_20261024/argmix/smoke_native_v1'
g.need(not base.exists(),'Fresh native smoke scope required')
base.mkdir()
work=base/'work'; work.mkdir()
config=g.metadata(g.CONFIG)
config['relate_execution']['workers']=2
config['bounds']['native_timeout_seconds']=120
config['bounds']['native_relate_bytes']=512*1024**2
config_path=base/'smoke_recipe.json'
config_path.write_text(json.dumps(config,indent=2)+'\n')
rows=['ID_1 ID_2 missing','0 0 0']+[f'sample{i} sample{i} 0' for i in range(8)]
(work/'input.sample').write_text('\n'.join(rows)+'\n')
(work/'sample_ages.txt').write_text('0\n'*16)
(work/'input.map').write_text('pos COMBINED_rate Genetic_Map\n1 1 0\n20001 0 0.02\n')
rng=np.random.Generator(np.random.PCG64(825))
rows=[]
for index in range(120):
 genotype=rng.binomial(1,.25,size=16)
 if genotype.sum()==0: genotype[index%16]=1
 if genotype.sum()==16: genotype[index%16]=0
 rows.append(f'22 site{index} {100+index*100} A C '+ ' '.join(map(str,genotype))+'\n')
(work/'input.haps').write_text(''.join(rows))
installation=g.metadata(config['input_refs']['relate_installation']['path'])
commands=dict(parent_id='synthetic_native_smoke_not_a_benchmark_parent',config_ref=g.file_ref(config_path),
 input_refs={p.name:g.file_ref(p) for p in work.iterdir()},cwd=str(work),native_trees=str(work/'inferred.trees'),
 argv=[[str(g.verify_ref(installation['binaries']['Relate'])),'--mode','All','-m','1.25e-8','-N','30000','--memory','1','--seed','825',
        '--haps',str(work/'input.haps'),'--sample',str(work/'input.sample'),'--sample_ages',str(work/'sample_ages.txt'),'--map',str(work/'input.map'),'-o','inferred'],
       [str(g.verify_ref(installation['binaries']['RelateFileFormats'])),'--mode','ConvertToTreeSequence','-i','inferred','-o','inferred']])
(work/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
guard=ScopeGuard(base,512*1024**2,150)
result=run_parallel(config_path,work,base/'run',guard=guard)
import tskit
ts=tskit.load(work/'inferred.trees')
g.need(ts.num_samples==16 and ts.num_sites==120,'Native tiny tree conversion differs')
receipt=dict(status='pass',scope='Synthetic120-site16-haplotype CPU format smoke only',
             terminal_ref=g.file_ref(base/'run/terminal.json'),trees=ts.num_trees,sites=ts.num_sites,samples=ts.num_samples,
             driver_ref=g.file_ref(g.ROOT/'src/arggnn/benchmarking/paper_argmix/paper_argmix_relate.py'),
             cuda_allocated=False,full1611_fit_claim=False)
(base/'completed.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
