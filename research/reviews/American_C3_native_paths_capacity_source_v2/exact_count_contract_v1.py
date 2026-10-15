"""Specific inactive exact-count contract. Reuses the frozen v1 root guard/supervisor helpers."""
from pathlib import Path
import ast
from research.reviews.American_C3_native_paths_root_v1 import native_paths_contract_v1 as old
ROOT=old.ROOT
REL='research/reviews/American_C3_native_paths_capacity_source_v2'
SOURCE=REL+'/exact_count_contract_v1.py'
DRIVER=REL+'/root_exact_count_driver_v1.py'
WRAPPER=REL+'/exact_count_consumer_v1.py'
COUNT_SOURCE=REL+'/exact_path_count_worker_v1.py'
require,ref,read,save,local=old.require,old.ref,old.read,old.save,old.local
previous=old.previous
GUARD_REF,POLICY_REF,REPAIR_REF=old.GUARD_REF,old.POLICY_REF,old.REPAIR_REF
MEMO_REF,V4_REF,RUNTIME_REF=old.MEMO_REF,old.V4_REF,old.RUNTIME_REF
FROZEN_PATHS_REF,COMMON_REF,AUTHOR_REF=old.PATHS_REF,old.COMMON_REF,old.AUTHOR_REF
COUNTS_REF,OBSERVATION_REF=old.COUNTS_REF,old.OBSERVATION_REF
CORE,BASE=old.CORE,old.BASE
COUNT_REF=ref(COUNT_SOURCE)
COUNT_CLOSURE=[COMMON_REF,COUNT_REF,FROZEN_PATHS_REF,AUTHOR_REF]
OUTPUT=REL+'/actual_counts_v1'
CAP=100000000
LOG_CAP=16777216
METADATA_ALLOWANCE=1048576
COUNT_JSON_CAP=1048576
COMPLETED_JSON_CAP=65536
DEADLINE_SECONDS=7200
RSS_CAP=32000000000
COMPLETED_STATUS='completed_native_C3_exact_path_count'
original_guard_v1=old.original_guard_v1
validate_counts_v1=old.validate_counts_v1

def configuration_v1(active=False):
    require(ref(FROZEN_PATHS_REF['path'])==FROZEN_PATHS_REF,'Frozen native paths source required')
    tree=ast.parse((ROOT/COUNT_SOURCE).read_text())
    node=next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='DEFAULTS' for t in n.targets))
    cfg={kw.arg:ast.literal_eval(kw.value) for kw in node.keywords}
    cfg.update(execution_enabled=active,observation_ref=OBSERVATION_REF,root_counts_ref=COUNTS_REF)
    require(cfg['output_cap_bytes']==CAP and cfg['count_json_cap_bytes']==COUNT_JSON_CAP
        and cfg['completed_json_cap_bytes']==COMPLETED_JSON_CAP
        and COUNT_JSON_CAP+COMPLETED_JSON_CAP+LOG_CAP+METADATA_ALLOWANCE<CAP,
        'Exact <=100MB count/log/metadata scope')
    return cfg,COUNT_JSON_CAP+COMPLETED_JSON_CAP

def closure_v1():
    return [ref(p) for p in (SOURCE,DRIVER,WRAPPER,old.SOURCE,old.DRIVER,
        old.PATTERN_REF['path'],MEMO_REF['path'],V4_REF['path'],GUARD_REF['path'],
        *tuple(r['path'] for r in COUNT_CLOSURE))]
