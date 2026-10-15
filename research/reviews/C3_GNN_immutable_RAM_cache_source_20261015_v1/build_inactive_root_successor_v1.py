"""Build a staged additive root route; no installation, guard or execution."""
import pathlib,json,hashlib,ast,copy,difflib
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN")
P=ROOT/"research/reviews/C3_GNN_immutable_RAM_cache_source_20261015_v1"
TREE=P/"staged_tree"
OLD="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2"
NEW="research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3"
OLD_MODULE="american_c3_gnn_prefetch_isolated980_exec_v2"
NEW_MODULE="american_c3_gnn_prefetch_ramcache_isolated980_exec_v3"
OLD_FAMILY="C3_GNN_prefetch_isolated980";NEW_FAMILY="C3_GNN_prefetch_ramcache_isolated980"
files=[];diffs=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def stage(old,target,text,kind):
 ast.parse(text,filename=target) if target.endswith(".py") else json.loads(text)
 p=TREE/target;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(text)
 files.append(dict(original_ref=ref(ROOT/old) if old else None,staged_ref=ref(p),intended_ref=dict(path=target,sha256=sha(p)),change_kind=kind))
 before=(ROOT/old).read_text() if old else ""
 diffs.extend(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile=old or "/dev/null",tofile=target))
def set_assign(s,name,value):
 n=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
 seg=ast.get_source_segment(s,n);assert s.count(seg)==1;return s.replace(seg,name+"="+repr(value),1)
def intended(target):return dict(path=target,sha256=sha(TREE/target))
stage(None,NEW+"/ram_cache_provider_v1.py",(P/"ram_cache_provider_v1.staged.py").read_text(),"new_RAM_provider_original_science_and_worker_lifecycle")
caps=dict(per_parent_logical_bytes=2*2**30,aggregate_logical_bytes=40*2**30,
 main_RSS_cap_bytes=88*2**30,main_noncache_planning_allowance_bytes=48*2**30,
 main_block_duplicate_planning_allowance_bytes=2**30,aggregate_RSS_cap_bytes=120*2**30,
 root_RSS_cap_bytes=128*2**30,worker_count=2,worker_RSS_cap_bytes=8*2**30)
s=(ROOT/OLD/"tracking_factory_v1.py").read_text().replace(OLD,NEW)
s=s.replace("def __init__(self,*,operational_caps,resource_check):","def __init__(self,*,operational_caps,immutable_cache_caps,resource_check):")
s=s.replace("from .persistent_provider_v1 import PersistentObservationProviderFactoryV1","from .ram_cache_provider_v1 import ImmutableRAMObservationProviderFactoryV1")
s=s.replace("self.base=PersistentObservationProviderFactoryV1(operational_caps=operational_caps,resource_check=resource_check)","self.base=ImmutableRAMObservationProviderFactoryV1(operational_caps=operational_caps,immutable_cache_caps=immutable_cache_caps,resource_check=resource_check)")
stage(OLD+"/tracking_factory_v1.py",NEW+"/tracking_factory_v1.py",s,"delegate_to_RAM_factory_only")
s=(ROOT/OLD/"worker_entry_v1.py").read_text().replace('REL="'+OLD+'"','REL="'+NEW+'"')
s=s.replace(OLD.replace("/",".")+".tracking_factory_v1",NEW.replace("/",".")+".tracking_factory_v1")
s=s.replace('operational_caps=cfg["observation_prefetch"],resource_check=fast','operational_caps=cfg["observation_prefetch"],immutable_cache_caps=cfg["immutable_feature_cache"],resource_check=fast')
s=s.replace('resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=caps["max_cpu_rss_bytes"]','resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=(cfg["immutable_feature_cache"]["main_RSS_cap_bytes"] if stage=="train" else caps["max_cpu_rss_bytes"])')
s=s.replace('"started_C3_GNN_prefetch_isolated980"','"started_C3_GNN_prefetch_ramcache_isolated980"')
stage(OLD+"/worker_entry_v1.py",NEW+"/worker_entry_v1.py",s,"fresh_entry_RAM_factory_and_88GiB_main_RSS")
s=(ROOT/OLD/"root_driver_v1.py").read_text().replace('REL="'+OLD+'"','REL="'+NEW+'"')
s=s.replace(OLD_FAMILY+"_transitive_closure_v1",NEW_FAMILY+"_transitive_closure_v1")
s=s.replace("source_only_"+OLD_FAMILY,"source_only_"+NEW_FAMILY)
stage(OLD+"/root_driver_v1.py",NEW+"/root_driver_v1.py",s,"fresh_root_namespace_only")
s=(ROOT/OLD/"canonical_dispatch.staged.py").read_text().replace(OLD,NEW).replace(OLD_MODULE,NEW_MODULE)
stage(OLD+"/canonical_dispatch.staged.py",NEW+"/canonical_dispatch.staged.py",s,"fresh_canonical_dispatch")
stage("src/arggnn/"+OLD_MODULE+".py","src/arggnn/"+NEW_MODULE+".py",s,"fresh_additive_canonical_dispatch")
s=(ROOT/OLD/"root_contract_v1.py").read_text().replace('REL="'+OLD+'"','REL="'+NEW+'"').replace(OLD_MODULE,NEW_MODULE)
for tail in ["operational_caps_v1","operational_config_v1","request_v1"]:s=s.replace(OLD_FAMILY+"_"+tail,NEW_FAMILY+"_"+tail)
s=s.replace('STATUSES={p:"completed_C3_GNN_prefetch_isolated980_"+p for p in PHASES}','STATUSES={p:"completed_C3_GNN_prefetch_ramcache_isolated980_"+p for p in PHASES}')
s=s.replace('"experiments/results/C3_GNN_prefetch_isolated980_"+stage+"_v1"','"experiments/results/C3_GNN_prefetch_ramcache_isolated980_"+stage+"_v1"')
s=s.replace("c3gnnprefetchisolated980","c3gnnprefetchramcacheisolated980")
oldpins=next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PREFETCH_SOURCE_PINS" for t in n.targets))
oldpins[NEW+"/ram_cache_provider_v1.py"]=intended(NEW+"/ram_cache_provider_v1.py")["sha256"]
s=set_assign(s,"PREFETCH_SOURCE_PINS",oldpins)
needle='REL+"/metadata_contract_v1.py",REL+"/tracking_factory_v1.py",REL+"/persistent_provider_v1.py",REL+"/persistent_workers_v1.py",'
replacement=repr(OLD+"/metadata_contract_v1.py")+',REL+"/tracking_factory_v1.py",'+repr(OLD+"/persistent_provider_v1.py")+','+repr(OLD+"/persistent_workers_v1.py")+','
assert s.count(needle)==1;s=s.replace(needle,replacement)
extra="RAM_CACHE_CAPS="+repr(caps)+"\nPRIOR_ISOLATED_REL="+repr(OLD)+"\nPRIOR_ISOLATED_DRIVER_REF="+repr(ref(ROOT/OLD/"root_driver_v1.py"))+"\n"
assert s.count("def source_paths():\n")==1;s=s.replace("def source_paths():\n",extra+"def source_paths():\n")
s=s.replace('*ORIGINAL_METADATA_IMPORT_SOURCE_PATHS,PRIOR_PREFETCH_DRIVER_REF["path"],','*ORIGINAL_METADATA_IMPORT_SOURCE_PATHS,PRIOR_PREFETCH_DRIVER_REF["path"],PRIOR_ISOLATED_DRIVER_REF["path"],')
a=s.index(' require(evidence["route"] in (');b=s.index(' folder=rel+"/admitted_"+stage+"_v1"',a)
s=s[:a]+''' require(evidence["route"] in ("current980_zero_worker","prefetch_current980","prefetch_isolated980","prefetch_ramcache_isolated980"),"Known immutable plan route")
 routes={"current980_zero_worker":(LEGACY_REL,"C3_GNN_current980_",LEGACY_DRIVER_REF),
  "prefetch_current980":(PRIOR_PREFETCH_REL,"C3_GNN_prefetch_current980_",PRIOR_PREFETCH_DRIVER_REF),
  "prefetch_isolated980":(PRIOR_ISOLATED_REL,"C3_GNN_prefetch_isolated980_",PRIOR_ISOLATED_DRIVER_REF),
  "prefetch_ramcache_isolated980":(REL,"C3_GNN_prefetch_ramcache_isolated980_",ref(DRIVER))}
 rel,prefix,driver=routes[evidence["route"]]
'''+s[b:]
a=s.index('  cache=cfg["CPU_observation_cache_bytes"]');b=s.index('\n else:require(caps.get("observation_prefetch")',a)
s=s[:a]+'''  cache=cfg["CPU_observation_cache_bytes"]
  require(caps.get("immutable_feature_cache")==RAM_CACHE_CAPS and cache==RAM_CACHE_CAPS["per_parent_logical_bytes"]
   and p["num_workers"]==2 and p["worker_rss_cap_bytes"]==8*2**30 and p["max_block_bytes"]==128*2**20
   and p["aggregate_rss_cap_bytes"]==120*2**30 and caps["max_cpu_rss_bytes"]==128*2**30,"Exact RAM/cache/worker and unchanged root RSS caps")
  require(RAM_CACHE_CAPS["aggregate_logical_bytes"]+RAM_CACHE_CAPS["main_noncache_planning_allowance_bytes"]==RAM_CACHE_CAPS["main_RSS_cap_bytes"]
   and RAM_CACHE_CAPS["main_block_duplicate_planning_allowance_bytes"]<=RAM_CACHE_CAPS["main_noncache_planning_allowance_bytes"]
   and RAM_CACHE_CAPS["main_RSS_cap_bytes"]+p["num_workers"]*p["worker_rss_cap_bytes"]<=p["aggregate_rss_cap_bytes"]-16*2**30,"Complete88+16=104GiB RSS bound and16GiB aggregate headroom")
  total=0
  for record in req["bindings"]["parent_plan_refs"].values():
   plan=read(record,cap=33554432)
   require(p["max_block_bytes"]>=plan["feature_tensor_byte_planning_bound_per_block"],"Block cap covers every original actual plan")
   require(plan["haplotypes"]==1600 and plan["focal_records"]==10000 and plan["class_order"]==["EUR","EAS","AFR"],"Exact RAM tensor-bound cardinalities")
   n=sum(b["nodes"] for b in plan["blocks"]);t=plan["trees"];sites=plan["sites"];blocks=len(plan["blocks"])
   bound=100*n+8*t*1600+1730*sites+24*t+27200*blocks+240000
   require(bound<=cache,"Every complete original parent fits its no-eviction RAM cap");total+=bound
  require(total<=RAM_CACHE_CAPS["aggregate_logical_bytes"],"All20 complete original feature banks fit aggregate RAM")
'''+s[b:]
s=s.replace('else:require(caps.get("observation_prefetch") is None,"Prefetch belongs only to TRAIN; VAL route unchanged")','else:require(caps.get("observation_prefetch") is None and caps.get("immutable_feature_cache") is None,"RAM/prefetch belongs only to TRAIN; VAL route unchanged")')
s=s.replace('observation_prefetch=copy.deepcopy(caps.get("observation_prefetch")),','observation_prefetch=copy.deepcopy(caps.get("observation_prefetch")),\n  immutable_feature_cache=copy.deepcopy(caps.get("immutable_feature_cache")),')
stage(OLD+"/root_contract_v1.py",NEW+"/root_contract_v1.py",s,"fresh_pins_RAM_caps_forecast_and_exact_prior_plan_routes")
cfg=json.loads((ROOT/OLD/"training_config.inactive.json").read_text());cfg["CPU_observation_cache_bytes"]=2*2**30
stage(OLD+"/training_config.inactive.json",NEW+"/training_config.inactive.json",json.dumps(cfg,indent=2,sort_keys=True)+"\n","cache_capacity_only_science_unchanged")
c=json.loads((ROOT/OLD/"caps_train.root_active_v1.json").read_text());c.update(active=False,schema=NEW_FAMILY+"_operational_caps_v1",immutable_feature_cache=caps)
stage(OLD+"/caps_train.root_active_v1.json",NEW+"/caps_train.proposed.inactive.json",json.dumps(c,indent=2,sort_keys=True)+"\n","inactive_same_GPU40GiB_storage2GiB_RSS128GiB_two8GiB_workers_plus_RAM_caps")
req=copy.deepcopy(json.loads((ROOT/OLD/"request_train.root_active_v1.json").read_text()));req.pop("caps_ref",None)
req.update(active=False,schema=NEW_FAMILY+"_request_v1",caps_template_ref=intended(NEW+"/caps_train.proposed.inactive.json"),
 scientific_training_config_ref=intended(NEW+"/training_config.inactive.json"))
stage(OLD+"/request_train.root_active_v1.json",NEW+"/request_train.proposed.inactive.json",json.dumps(req,indent=2,sort_keys=True)+"\n","inactive_fresh_namespace_exact20_bindings_and_original_science_refs")
(P/"operational_successor.diff").write_text("".join(diffs))
manifest=dict(schema="C3_immutable_RAM_cache_inactive_install_manifest_v1",canonical_install_performed=False,files=files,
 future_root_packet=NEW,future_output="experiments/results/"+NEW_FAMILY+"_train_v1",
 future_admission_path=NEW+"/admitted_train_v1",future_module="arggnn."+NEW_MODULE,
 original_live_route_unchanged=OLD,current_live_session=50244)
(P/"staged_source_install_manifest.v1.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(staged_files=len(files),new_root_contract=intended(NEW+"/root_contract_v1.py"),new_provider=intended(NEW+"/ram_cache_provider_v1.py")),indent=2))
