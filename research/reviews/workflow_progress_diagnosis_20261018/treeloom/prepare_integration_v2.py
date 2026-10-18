"""Private original-runner integration for exact prepare-once compact RAM reuse."""
from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from . import candidate_v1 as candidate
from . import resident_pool_v2 as pool_core


def compiled_complete(preparation_function=None):
    from research.reviews.TreeLoom_preparation_reuse_source_20261017_v1 import optimized_preparation_v1 as original
    def transform(node):
        starts=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Tuple) and [getattr(e,"id",None) for e in t.elts]==
                        ["arrays","episode_metadata"] for t in n.targets)]
        ends=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=="episodes" for t in n.targets)]
        candidate.need(len(starts)==len(ends)==1 and ends[0]==starts[0]+1,
                       "One exact original episode builder/loader pair required")
        node.body[starts[0]:ends[0]+1]=ast.parse(
            "episodes,episode_metadata=_build_episodes(bank,max_payload_bytes=max_episode_payload_bytes,"
            "isolated_hold_indices=isolated_holds)").body
    call=candidate.compile_function(preparation_function or original.prepare_from_loaded_observation_v1,transform,
        encode_block=candidate.encode_block,_build_episodes=candidate.build_episodes,
        prepare_cut_plan_v1=candidate.prepare_cut)
    return call


def prepare_complete(state,**kwargs):
    return compiled_complete()(state,**kwargs)


def install_runtime_preparation(runtime):
    # Clone the actual private namespace, preserving fresh IDs and private bundle type.
    compiled=compiled_complete(runtime.preparation.prepare_from_loaded_observation_v1)
    runtime.preparation=SimpleNamespace(prepare_from_loaded_observation_v1=compiled)
    return runtime


def build_spec(**kwargs):
    from arggnn.benchmarking import treeloom_training as original
    kwargs.setdefault("observation_cache_bytes",pool_core.COMPACT_POOL_BYTES)
    kwargs.setdefault("preparation_headroom_bytes",pool_core.WORKING_BYTES)
    kwargs.setdefault("training_headroom_bytes",pool_core.WORKING_BYTES)
    kwargs.setdefault("max_rss_bytes",pool_core.MAX_RSS_BYTES)
    return original.build_spec(**kwargs)


def review_source_refs():
    from arggnn.benchmarking import contracts as c
    return [c.file_ref(Path(__file__).parent/name) for name in
        ("indices_v1.py","candidate_v1.py","cache_v1.py","resident_pool_v2.py","prepare_integration_v2.py","storage_bound_v2.json")]


def run_training(spec,*,resource_check):
    """Only callable in a separately admitted ROOT GPU worker; never launches one."""
    from arggnn.benchmarking import treeloom_training as original
    from arggnn.benchmarking import treeloom_fresh as fresh
    from arggnn.benchmarking import contracts as c
    registry=original._validate_spec(spec)
    bounds=json.loads((Path(__file__).parent/"storage_bound_v2.json").read_text())
    parents=tuple(registry.ids("train"))
    parent_bounds={r["parent_id"]:r["complete_compact_cache_upper_bound_bytes"]+160000
                   for r in bounds["parents"]}
    candidate.need(set(parent_bounds)==set(parents),"All20 metadata-only original full parent bounds required")
    inputs={p:{k:copy.deepcopy(spec["train_inputs"][p][k]) for k in fresh.INPUT_REFS} for p in parents}
    external={p:{"bank.genotype":c.read_ref(inputs[p]["observation_ref"])["genotypes_ref"]} for p in parents}
    compact=pool_core.CompactParentPool(parent_bounds=parent_bounds,input_refs=inputs,
        external_arrays=external,max_cache_bytes=spec["observation_cache_bytes"],
        max_rss_bytes=spec["memory_envelope"]["max_rss_bytes"],
        preparation_headroom_bytes=spec["memory_envelope"]["preparation_headroom_bytes"],
        rss_bytes=original._current_rss_bytes,resource_check=resource_check,event=lambda *args:None)
    def runtime_factory(actual_registry):
        return install_runtime_preparation(fresh.FreshTreeLoom(actual_registry))
    proxy=SimpleNamespace(**{k:v for k,v in vars(fresh).items() if not k.startswith("__")})
    proxy.FreshTreeLoom=runtime_factory
    def load(parent,cache,prepare,**kwargs):
        # Bind the original runner's bounded progress writer without changing it.
        compact.event=kwargs["event"]
        candidate.need(cache is compact,"Private exact compact pool required")
        return compact.load(parent,prepare)
    def no_evict(cache,*,memory_envelope,stage,rss_bytes,resource_check,event,keep_parent=None):
        candidate.need(cache is compact,"No expanded-bank LRU or compact eviction")
        reserve=memory_envelope[stage+"_headroom_bytes"]
        if stage=="training":
            # The declared128GiB working set includes the already restored parent.
            # Reserve only its remaining truth/autograd workspace beyond current RSS.
            expanded=compact.active_expanded_bytes[keep_parent]
            candidate.need(0<=expanded<reserve,"One complete expanded parent must fit active working reserve")
            reserve-=expanded
        compact.admit(reserve,"compact_pool_"+stage+"_headroom")
    def train_loop(**kwargs):
        compact.require_all_prepared()
        result=original.train_loop(**kwargs)
        candidate.need(sum(compact.restorations.values())==600
                       and set(compact.restorations.values())=={30},
                       "All20 compact parents must restore once per epoch for all600 updates")
        return result
    def transform(node):
        matches=0
        for at,item in enumerate(node.body):
            if isinstance(item,ast.Expr) and isinstance(item.value,ast.Call) and \
                    isinstance(item.value.func,ast.Name) and item.value.func.id=="write_json" and \
                    "training_completed.json" in ast.unparse(item):
                node.body.insert(at,ast.parse("result['compact_pool_receipt']=_compact_receipt()").body[0])
                matches+=1;break
        candidate.need(matches==1,"One final original training receipt write required")
    call=candidate.compile_function(original.run_training,transform,fresh=proxy,
        OrderedDict=lambda:compact,_load_cached_parent=load,_admit_parent_headroom=no_evict,
        train_loop=train_loop,_compact_receipt=compact.receipt)
    return call(spec,resource_check=resource_check)
