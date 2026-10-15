"""H050 private metadata boundary: scientific inputs are not runtime bindings.

Only the exact warm-start checkpoint moves behind a bound JSON proxy. Original
producer/context body, scientific refs, payload hash checks and guard APIs remain.
"""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H050_shared_admission_source_metadata_closure_fix_20261016_v1"
SOURCE=HERE+"/context_metadata_fix_v1.py"
ADMISSION_SOURCE=HERE+"/root_h050_shared_admission_fix_v1.py"
COMMON_REF={'path': 'research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/bridge_common_v1.py', 'sha256': '5ec43d65bde777d18c071be2cdf4ee5c14ef2784bfe54ac27582f187ee93552b'}
CONTEXT_REF=dict(path="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/context_factory_v1.py",
 sha256="58bf94db37f8fb63afeca64352fd362ddf580d7117ef794481dfe4395f087fa2")
WARM_REF=dict(path="experiments/results/H045_GPU5_episode_train_seed101_v1/own_state.pass_30.pt",
 sha256="11dc132ac1958c452fa1d2651bd261258aac6c036ebbdbe7175f235148905331")
ALLOWED=frozenset({".py",".sh",".json"})


def require(ok,message):
    if not ok:raise ValueError(message)


def ref(name):
    p=ROOT/name
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink(),"Canonical bound metadata source")
    return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def load_exact(record,label):
    require(ref(record["path"])==record,"Exact unchanged metadata/context source")
    sp=importlib.util.spec_from_file_location(label,ROOT/record["path"]);m=importlib.util.module_from_spec(sp)
    sys.modules[label]=m;sp.loader.exec_module(m);return m


def source_metadata_refs_v1(rows,cfg):
    scientific=[r for r in rows if Path(r["path"]).suffix not in ALLOWED]
    require(scientific==[WARM_REF] and cfg["warmstart_checkpoint_ref"]==WARM_REF,
        "Only the single exact original scientific warm-start payload is indirect")
    proxy_ref=cfg["scientific_input_closure_ref"]
    require(ref(proxy_ref["path"])==proxy_ref,"Exact canonical JSON proxy bytes before decode")
    proxy=json.loads((ROOT/proxy_ref["path"]).read_text())
    require(proxy.get("schema")=="H050_scientific_input_transitive_closure_v1"
        and proxy.get("hypothesis_id")=="H050" and proxy.get("scientific_input_refs")==[WARM_REF]
        and proxy.get("metadata_adapter_ref")==ref(SOURCE)
        and proxy.get("admission_successor_ref")==ref(ADMISSION_SOURCE)
        and proxy.get("original_common_ref")==COMMON_REF and proxy.get("original_context_ref")==CONTEXT_REF,
        "Bound JSON proxy preserves exact scientific SHA and original metadata source proof")
    # Original producer's reader still authenticates WARM_REF bytes before load.
    # Every other exact source/runtime/authority ref remains directly bound.
    result=[r for r in rows if Path(r["path"]).suffix in ALLOWED]
    for r in [proxy_ref,ref(SOURCE),ref(ADMISSION_SOURCE),COMMON_REF,CONTEXT_REF]:
        if r not in result:result.append(r)
    require(all(Path(r["path"]).suffix in ALLOWED for r in result) and len(result)<=64,"Original metadata-only64binding rule")
    return result


def private_common_v1():
    c=load_exact(COMMON_REF,"_H050_private_source_metadata_common_fix_v1")
    c.CONTEXT=SOURCE  # The producer actually loads this declared context source.
    original=c.direct_required_refs_v1
    def required(cfg,spec):return c.unique(source_metadata_refs_v1(original(cfg,spec),cfg))
    c.direct_required_refs_v1=required
    return c


def build_context_v1(record,spec):
    c=private_common_v1()
    # The original context function body is compiled unchanged into a private
    # metadata namespace. No guard, Torch method, model or training code changes.
    context=load_exact(CONTEXT_REF,"_H050_original_context_body_for_metadata_fix_v1")
    text=(ROOT/CONTEXT_REF["path"]).read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="build_context_v1")
    namespace=dict(context.__dict__)
    namespace.update(validate_spec_v1=c.validate_spec_v1,CONTEXT=SOURCE,SOURCE=SOURCE)
    exec(compile(ast.Module(body=[node],type_ignores=[]),"<unchanged_H050_original_context_body>","exec"),namespace)
    return namespace["build_context_v1"](record,spec)
