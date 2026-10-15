#!/usr/bin/env python3
"""Independent tiny metadata checks; no numerical/native modules or real payloads."""
import copy
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
AUTHOR=HERE.parent/'ARGMix_prospective1611_native_adapters_20261016_v1'
sys.path.insert(0,str(AUTHOR))
from panel_adapter import Panel,AGES,project_ap_probabilities,compare_fit_summaries
from truth_adapter import construct_common_labels,census_founder_label,simulation_sample_map

def must_reject(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('invalid metadata accepted')

rows=[]
cohorts=('NF','Yam','Ana','Present')
for ci,cohort in enumerate(cohorts):
    for phase in (0,1):
        rows.append({'haplotype_id':cohort+'_h'+str(phase),'diploid_id':cohort,
                     'phase':phase,'cohort':cohort,'age_generations':AGES[cohort],
                     'node_id':100+31*ci+7*phase})
panel=Panel({'schema':'prospective1611_roster_v1','rows':rows},full=False)
map_rows=[{'haplotype_id':r['haplotype_id'],'simulation_node_id':50000+13*i}
          for i,r in enumerate(reversed(rows))]
sim_mapping={'schema':'simulation_haplotype_node_mapping_v1','rows':map_rows}
sim={r['haplotype_id']:r['simulation_node_id'] for r in map_rows}
names={1:'Yam',2:'Neo',3:'EHG',4:'WHG',7:'CHG',8:'Ana',9:'present_bronze'}
cohort_pop={'NF':2,'Yam':1,'Ana':8,'Present':9}
pop_mapping={'schema':'simulation_population_mapping_v1','populations':{str(k):v for k,v in names.items()}}
node_rows={sim[r['haplotype_id']]:r for r in rows}
parent={nid:70000+i for i,nid in enumerate(node_rows)}
ancestor_pop={parent[nid]:(7 if r['cohort']=='NF' else 3 if r['cohort']=='Yam' else 8)
              for nid,r in node_rows.items()}
class O:
    def __init__(self,**kw): self.__dict__.update(kw)
class Tree:
    interval=O(left=5.,right=10.)
    def parent(self,n): return parent.get(n,-1)
    def time(self,n): return float(node_rows[n]['age_generations']) if n in node_rows else 350.
    def population(self,n): return ancestor_pop[n]
class TS:
    sequence_length=10.
    def samples(self): return list(reversed(list(node_rows)))
    def node(self,n): return O(time=node_rows[n]['age_generations'],population=cohort_pop[node_rows[n]['cohort']])
    def population(self,p): return O(metadata={'name':names[p]})
    def at(self,position):
        assert position==5., 'truth lookup must use exact target midpoint'
        return Tree()
targets=[{'row_id':'independent_'+cohort,'parent_id':'metadata_parent','haplotype_id':cohort+'_h0',
          'tree_index':0,'left':0.,'right':10.,'position':5.} for cohort in ('NF','Yam')]
results=[]
labels=construct_common_labels(TS(),sim_mapping,pop_mapping,targets,panel)
assert labels==[{'row_id':'independent_NF','ancestry':'CHG'},
                {'row_id':'independent_Yam','ancestry':'EHG'}]
results.append('exact right-continuous midpoint and distinct simulation IDs yield CHG for NF and EHG for Yam')
class NoCensus(Tree):
    def time(self,n): return super().time(n)+1 if n not in node_rows else super().time(n)
must_reject(lambda:census_founder_label(NoCensus(),sim['NF_h0'],names))
results.append('missing exact census350 node rejected')
bad_names=dict(names);bad_names[7]='Neo'
must_reject(lambda:census_founder_label(Tree(),sim['NF_h0'],bad_names))
results.append('mixed NF population cannot substitute for founder ancestry')
bad_mapping=copy.deepcopy(sim_mapping);bad_mapping['rows'][1]['simulation_node_id']=bad_mapping['rows'][0]['simulation_node_id']
must_reject(lambda:simulation_sample_map(bad_mapping,panel))
results.append('duplicate simulation node mapping rejected')
donor_ids={r['haplotype_id'] for r in panel.donor_rows('NF_h0')}
assert not donor_ids&{'NF_h0','NF_h1'} and {'Yam_h0','Yam_h1'}<=donor_ids
results.append('both focal mates excluded while other mixed-cohort donors retained')
p=[.26,.28,.11,.10,.15,.10]
projected=project_ap_probabilities(p)
assert max(range(6),key=p.__getitem__)==1 and max(range(4),key=projected.__getitem__)==0
results.append('pooled AP founder decoding can differ from native six-way argmax')
sig={'parent_id':'metadata_parent','targets_sha256':'a'*64,'roster_sha256':'b'*64,
     'observed_trees_sha256':'c'*64,'labels_sha256':'d'*64,
     'supervision_contract_sha256':'e'*64,'truth_definition_sha256':'f'*64}
a={'schema':'prospective1611_native_fit_summary_v1','method':'argmix',
   'train_shared_signatures':[sig],'val_shared_signatures':[sig],
   'logical_train_targets':1,'logical_val_targets':1,'fit_config_sha256':'1'*64}
b=copy.deepcopy(a);b['method']='ap'
assert compare_fit_summaries(a,b)['same_supervised_data_verified']
b['train_shared_signatures'][0]['labels_sha256']='2'*64
must_reject(lambda:compare_fit_summaries(a,b))
results.append('paired fit rejects a different label file on identical target IDs')
forbidden={'numpy','torch','tensorflow','tskit','ancpath'} & set(sys.modules)
assert not forbidden, forbidden
print(json.dumps({'schema':'independent_native_adapter_tiny_metadata_checks_v1','passed':True,
                  'checks':results,'check_count':len(results),'native_numeric_paths_executed':False,
                  'forbidden_numeric_modules_loaded':sorted(forbidden)},indent=2,sort_keys=True))
