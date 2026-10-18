"""Fresh per-ARG native ARGformer and FLARE numerical stages.

No launch, admission, GPU lease, numerical import or data access occurs on import.
ROOT supplies an admitted supervisor and a bounded native runner. Author science
is unchanged; all inference/focal joins precede any native call. Only new corpus
and embedding scratch is declared disposable after the final prediction seal AND
successful outer wait. This module never deletes it or any historical artifact.
"""
from __future__ import annotations
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from . import american_focals as focals
from . import american_observation as observation
from . import american_preparation as generation

ROOT = generation.ROOT
AUTHOR = ".runtime/author_sources/argformer_f7545be7bca0"
NATIVE_PYTHON = ".runtime/conda/envs/argformer_native_v1/bin/python"
GPU0_UUID = "GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
WORKER = "arggnn.benchmarking.american_competitor_worker"
SCHEMA = "american_fresh_native_competitor_stage_v1"
AUTHOR_MANIFEST = dict(path="research/reviews/American_C3_native_consumer_source_v1/author_source_manifest.json",
                       sha256="bed570db438f2c4981ebf606460dc4ff72a97b148ba371e3873ab0141376736a")
BASE_CONFIG = dict(path="research/reviews/C3_bert_base_configuration_root_v1/config.json",
                   sha256="7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20")
PIN_HASHES = {
 "src/arggnn/american_c3_paths_v1.py":"3b32f16e0a81160852e5b93fe9033abf9bc2eeb56c3b15ca552d73f4aacf374c",
 "src/arggnn/american_c3_common_v1.py":"615d3892f1df7e9de39e57991eab883b2eca1af9d13e424661541aac4b2d2abf",
 "src/arggnn/american_c3_native_concat_v1.py":"a9de71c71baed134f829619b23ca11b910eefb63be92af24d8993e232b70c7bb",
 "src/arggnn/american_c3_evaluate_v1.py":"1ada0ab40fe9f198b89a1953bb7c0688e68f1dd44b34dc118600ac25eefc70d3",
 "src/arggnn/american_c3_flare_common_v1.py":"7dd0b7fc49ace60a6c19a8e7f57b3c9ad7dbeb145810e558ea141668677c511a",
 "src/arggnn/american_c3_flare_convert_v1.py":"6e55e295226a183548a23258e7230855842bb876b4d2a90875d8f2ad53181ff5",
 "src/arggnn/american_c3_flare_run_v2.py":"50c48a25834f5b4246f1a28deb2ff564e05e4db2308203578448d966d5273b21",
 "src/arggnn/american_c3_flare_evaluate_v1.py":"ce7567d0f71054cfd96c9a44727e20501ad6972e4af7340de7ccc231a9e8aed7",
 "research/reviews/native_VAL000_downstream_IO_efficiency_source_20261016_v1/efficient_downstream_v1.py":"bab7057a958e9cfd51f503b5680034c0067226a661b820eb42bb16c7133a94b9",
 "research/reviews/American_C3_native_MLM_amendment_v12/abstract_unix_bootstrap_v1.py":"4550ead44f36eeb0fda8a817a2002082c3d067cf1a84fd817301532577d8b014",
}
SETTINGS = dict(variant="author_release_per_ARG_CURRENT_RECONSTRUCTION_not_exact_Table1",
    MLM=dict(epochs=5,batch=512,microbatch=512,precision="amp_bf16",seed=17,
             masking_probability=.3,corruption="native80_10_10",drop_last=True),
    contrastive=dict(epochs=1,batch=4096,microbatch=4096,precision="fp32",seed=36,
                     temperature=.3,uniform_class_weight_fallback=True,drop_last=True),
    embeddings=dict(pooling="head_norm",batch=256,max_samples=0,dimension=768),
    retrieval=dict(topk=1,q_batch=1024,e_chunk=100000),
    checkpoint_retention=1,additional_final_checkpoint=True,
    paper_conflicts=["native80/10/10 masking versus literal paper MASK replacement",
                     "released contrastive temperature.30 versus paper default.05",
                     "released uniform class-weight fallback versus inverse-frequency weighting"])
PATH_CAPS = ("max_unique_paths","max_path_tokens","max_path_occurrences","max_vocabulary_nodes","max_raw_path_nodes")
VIEWS = ("contrastive/train","contrastive/val","query_paths/train","query_paths/val")


def need(ok,message):
    if not ok: raise ValueError(message)


def path(value): return generation._path(value)
def ref(value): return generation._reference(value)
def read(record): return focals._read(record)


def requirements():
    refs = [AUTHOR_MANIFEST,BASE_CONFIG,*[dict(path=p,sha256=h) for p,h in PIN_HASHES.items()]]
    refs += generation._read(AUTHOR_MANIFEST)["refs"]
    refs += focals.requirements()["source_refs"]
    refs += [ref(__file__),ref("src/arggnn/benchmarking/american_competitor_worker.py")]
    refs = list({r["path"]:r for r in refs}.values())
    for r in refs: need(ref(r["path"])==r,"Pinned native scientific source changed: "+r["path"])
    return dict(source_refs=refs,native_python=NATIVE_PYTHON,settings=copy.deepcopy(SETTINGS),
                metadata_only=True,query_truth_inputs=False,launch_authority=False)


def inputs(inference_ref,focal_preparation_ref):
    source,fp = read(inference_ref),read(focal_preparation_ref)
    need(source.get("schema")=="american_fresh_inference_ready_v1"
         and source.get("requires_tsinfer_tsdate") is False
         and source.get("query_truth_in_training_inputs") is False
         and source.get("truth_scoring_authorized") is False
         and source.get("declared_conventions")==generation.CONVENTIONS
         and source.get("declared_inference_conventions")==observation.INFERENCE_CONVENTIONS,
         "Complete fresh observed inference required")
    m = read(source["observation_ref"])
    need(m.get("schema")=="american_c3_observed_v1" and m.get("genealogy_kind")==observation.GENEALOGY
         and m.get("query_ancestry_truth_in_inputs") is False and m.get("sequence_length_bp")==30000000
         and m.get("sites_and_sample_identities_verified") is True
         and m.get("parent_id")==source["parent_id"] and m.get("split")==source["split"]
         and m.get("class_order")==["EUR","EAS","AFR"],"Whole shared inferred parent required")
    need(fp.get("schema")==focals.RESULT_SCHEMA and fp.get("status")=="completed_compact_query_focals"
         and fp.get("parent_id")==source["parent_id"] and fp.get("observation_ref")==source["observation_ref"]
         and fp.get("class_order")==m["class_order"] and fp.get("genealogy_kind")==m["genealogy_kind"]
         and fp.get("query_focal_count")==10000 and fp.get("original_record_ids_preserved") is True
         and fp.get("path_sampling_seed")==source["seeds"]["sampling"]
         and fp.get("query_ancestry_truth_opened") is False,"Exact same-parent label-blind compact focal identity required")
    for r in [fp["focals_ref"],*[m[k] for k in ("tree_ref","roster_ref","genotypes_ref","positions_ref","alleles_ref")]]:
        focals._public_ref(r)
    return source,m,fp


def _out(value,parent,fresh=True):
    out=path(value)
    need(out.is_relative_to(ROOT/"experiments/results") and parent in out.parts
         and (not fresh or not out.exists()),"Fresh canonical same-parent competitor stage output required")
    return out


def _limits(limits):
    need(type(limits) is dict and set(limits)=={"max_seconds","max_rss_bytes","max_output_bytes"}
         and all(type(v) is int and v>0 for v in limits.values()),"Explicit finite whole-stage wall/RSS/output limits required")
    return copy.deepcopy(limits)


def _base(method,stage,inference_ref,focal_preparation_ref,output_directory,limits,*,fresh=True):
    source,m,fp=inputs(inference_ref,focal_preparation_ref)
    out=_out(output_directory,source["parent_id"],fresh=fresh)
    return dict(schema=SCHEMA,method=method,stage=stage,parent_id=source["parent_id"],split=source["split"],
                inference_ref=copy.deepcopy(inference_ref),focal_preparation_ref=copy.deepcopy(focal_preparation_ref),
                observation_ref=source["observation_ref"],focal_registry_ref=fp["focals_ref"],
                output_directory=str(out.relative_to(ROOT)),limits=_limits(limits),requirements=requirements())


def _prep(record,spec):
    p=read(record)
    need(p.get("schema")=="american_fresh_ARGformer_preparation_v1"
         and p.get("status")=="completed_native_C3_path_preparation"
         and p.get("parent_id")==spec["parent_id"] and p.get("observation_ref")==spec["observation_ref"]
         and p.get("focals_ref")==spec["focal_registry_ref"] and p.get("query_ancestry_truth_opened") is False
         and p.get("class_order")==["EUR","EAS","AFR"]
         and p.get("genealogy_kind")==observation.GENEALOGY
         and p.get("MLM_optimizer_updates",0)>0 and p.get("contrastive_optimizer_updates",0)>0
         and p["MLM_optimizer_updates"]==5*(p["datasets"]["mlm/train"]["rows"]//512)
         and p["contrastive_optimizer_updates"]==p["datasets"]["contrastive/train"]["rows"]//4096,
         "Actual complete native per-ARG preparation required")
    return p


def _producer(record,spec,stage):
    p=read(record)
    need(p.get("schema")=="american_fresh_ARGformer_stage_completed_v1" and p.get("stage")==stage
         and p.get("parent_id")==spec["parent_id"] and p.get("preparation_ref")==spec["preparation_ref"]
         and p.get("focal_registry_ref")==spec["focal_registry_ref"]
         and p.get("query_ancestry_truth_opened") is False,"Actual same-parent native predecessor required")
    validate_wait(p["native_wait"])
    prep=_prep(spec["preparation_ref"],spec)
    if stage in ("mlm","contrastive"):
        updates=prep["MLM_optimizer_updates"] if stage=="mlm" else prep["contrastive_optimizer_updates"]
        need(p.get("completed_epochs")== (5 if stage=="mlm" else 1)
             and p.get("actual_optimizer_updates")==updates
             and p.get("optimizer_unique_step_values")==[updates],"Actual full predecessor saved training budget required")
    else:
        need(set(p.get("outputs",{}))==set(VIEWS) and p.get("pooling")=="head_norm"
             and p.get("max_samples")==0 and all(p["outputs"][k]["rows"]==prep["datasets"][k]["rows"] for k in VIEWS),
             "Every complete native embedding split required")
    return p


def build_argformer_spec(*,stage,inference_ref,focal_preparation_ref,output_directory,limits,
                         preparation_ref=None,producer_ref=None,path_limits=None,_fresh=True):
    need(stage in ("prepare","mlm","contrastive","embeddings","retrieval"),"Exact native ARGformer stage required")
    spec=_base("argformer",stage,inference_ref,focal_preparation_ref,output_directory,limits,fresh=_fresh)
    if stage=="prepare":
        need(preparation_ref is producer_ref is None and type(path_limits) is dict
             and set(path_limits)==set(PATH_CAPS) and all(type(v) is int and v>0 for v in path_limits.values()),
             "Explicit complete raw-corpus count limits required")
        spec["path_limits"]=copy.deepcopy(path_limits)
        original=importlib.import_module("arggnn.american_c3_paths_v1")
        spec["full_corpus_serialization_bound_bytes"]=original.storage_bound(path_limits)+4*2**20
        need(spec["full_corpus_serialization_bound_bytes"]<=limits["max_output_bytes"],
             "Whole complete native corpus serialization bound exceeds cap")
    else:
        need(preparation_ref is not None and path_limits is None,"Actual per-ARG native corpus reference required")
        spec["preparation_ref"]=copy.deepcopy(preparation_ref)
        p=_prep(preparation_ref,spec)
        need((producer_ref is None)==(stage=="mlm"),"Only MLM starts a fresh parent fit")
        if producer_ref is not None:
            spec["producer_ref"]=copy.deepcopy(producer_ref)
            prev=_producer(producer_ref,spec,{"contrastive":"mlm","embeddings":"contrastive","retrieval":"embeddings"}[stage])
            spec["checkpoint_ref"]=prev["checkpoint_ref"]
        spec["expected_updates"]=p["MLM_optimizer_updates"] if stage=="mlm" else p["contrastive_optimizer_updates"]
        if stage in ("mlm","contrastive"):
            padded_vocab=((p["vocabulary_nodes"]+4+7)//8)*8
            checkpoint_bound=16*(100000000+padded_vocab*768)+64*2**20
            spec["max_single_checkpoint_planning_bytes"]=checkpoint_bound
            spec["training_serialization_bound_bytes"]=2*checkpoint_bound+2**30
            need(spec["training_serialization_bound_bytes"]<=limits["max_output_bytes"],
                 "Keep1 plus mandatory final native checkpoint/cache/log bound exceeds cap")
        if stage=="embeddings":
            planned=sum(p["datasets"][k]["rows"]*(768*4+128*8*2+16) for k in VIEWS)+128*2**20
            need(planned<=limits["max_output_bytes"],"All full native embedding outputs exceed admitted cap")
            spec["embedding_serialization_bound_bytes"]=planned
    return spec


def validate_spec(spec,*,fresh=True):
    common={k:spec[k] for k in ("stage","inference_ref","focal_preparation_ref","output_directory","limits")}
    if spec.get("method")=="argformer":
        expected=build_argformer_spec(**common,preparation_ref=spec.get("preparation_ref"),
            producer_ref=spec.get("producer_ref"),path_limits=spec.get("path_limits"),_fresh=fresh)
    else:
        need(spec.get("method")=="flare","Explicit native competitor required")
        expected=build_flare_spec(**common,conversion_ref=spec.get("conversion_ref"),
            native_runtime=spec.get("native_runtime"),max_sites=spec["max_sites"],_fresh=fresh)
    need(spec==expected,"Exact full native stage settings and data-derived budget required")
    return _out(spec["output_directory"],spec["parent_id"],fresh=fresh)


class Writer(observation._StageWriter):
    """New bounded writer. Legacy converter callback is forwarded to ROOT only.

    adapter/storage are solely the old numerical function's callback interface;
    no old admission, account, policy, guard or authority is created or claimed.
    """
    def __init__(self,out,cap,check,*,completion_name="scientific_completed.json",create=True):
        if create: super().__init__(out,cap,check)
        else:
            self.path,self.cap,self.check=out,cap,check
            self.pending,self.finished={},sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink())
        self.completion_name=completion_name
        self.adapter=SimpleNamespace(budget_check_v9=lambda *a,**k:check("original_native_numerical_boundary"))
        self.storage=None
    def prepare(self,name,planned):
        return super().prepare(self.completion_name if name=="completed.json" else name,planned)


def validate_wait(wait):
    need(wait.get("actual_direct_wait") is True and wait.get("actual_direct_returncode")==0
         and wait.get("native_process_absent") is True
         and wait.get("outer_owned_group_cleanup_pending") is True,
         "Actual successful direct native wait and process absence; outer group cleanup remains ROOT-owned")
    return wait


def run_argformer(spec,*,bounded_runner=None,resource_check):
    out=validate_spec(spec)
    need(spec["method"]=="argformer" and callable(resource_check),"Admitted ARGformer stage callback required")
    if spec["stage"]=="prepare": return _argformer_prepare(spec,out,resource_check)
    need(callable(bounded_runner),"Injected ROOT bounded native runner required")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")==GPU0_UUID and os.environ.get("ARGGNN_PHYSICAL_GPU")=="0",
         "Native stages must inherit ROOT-admitted physical GPU0 visibility")
    resource_check("before_native_stage_scope")
    out.mkdir(parents=True,exist_ok=False)
    w=Writer(out,spec["limits"]["max_output_bytes"],resource_check,create=False)
    spec_ref=w.json("native_stage_spec.json",spec)
    env=dict(os.environ,
             PYTHONPATH=str(ROOT/"src")+":"+str(ROOT),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
             TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    wait=bounded_runner([str(ROOT/NATIVE_PYTHON),"-B","-s","-m",WORKER,
                         "--spec-ref",json.dumps(spec_ref,sort_keys=True)],str(ROOT),env,
                        copy.deepcopy(spec["limits"]),str(out/"native.log"))
    validate_wait(wait)
    native=read(ref(out/"scientific_completed.json"))
    need(native["parent_id"]==spec["parent_id"] and native["stage"]==spec["stage"]
         and native["preparation_ref"]==spec["preparation_ref"]
         and native["focal_registry_ref"]==spec["focal_registry_ref"]
         and native["query_ancestry_truth_opened"] is False,"Exact native scientific return required")
    if spec["stage"] in ("mlm","contrastive"):
        need(native.get("completed_epochs")== (5 if spec["stage"]=="mlm" else 1)
             and native.get("actual_optimizer_updates")==spec["expected_updates"]
             and native.get("optimizer_unique_step_values")==[spec["expected_updates"]],
             "Actual complete scientific training return required")
    w=Writer(out,spec["limits"]["max_output_bytes"],resource_check,create=False)
    return w.json("stage_completed.json",dict(native,schema="american_fresh_ARGformer_stage_completed_v1",
                                             native_wait=wait,outer_worker_wait_still_required=True))


def _argformer_prepare(spec,out,check):
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Native path preparation is CPU-only")
    original=importlib.import_module("arggnn.american_c3_paths_v1")
    cfg=dict(original.DEFAULTS,**spec["path_limits"],execution_enabled=True,observation_ref=spec["observation_ref"],
             path_sampling_seed=read(spec["inference_ref"])["seeds"]["sampling"],query_focal_count=10000,
             output_cap_bytes=spec["limits"]["max_output_bytes"]-4*2**20)
    need(original.storage_bound(cfg)<=cfg["output_cap_bytes"],"Whole original full-corpus production bound exceeds cap")
    out.mkdir(parents=True,exist_ok=False)
    scratch=out/"scratch"
    w=Writer(scratch,cfg["output_cap_bytes"],check,completion_name="original_preparation.json")
    completion_ref=original.execute(cfg,w)
    p=read(completion_ref)
    need(p["status"]=="completed_native_C3_path_preparation" and p["parent_id"]==spec["parent_id"]
         and p["observation_ref"]==spec["observation_ref"] and p["query_ancestry_truth_opened"] is False,
         "Genuine full native corpus completion required")
    import numpy as np
    from arggnn.american_c3_common_v1 import bound
    with np.load(bound(p["focals_ref"]),allow_pickle=False) as a, np.load(bound(spec["focal_registry_ref"]),allow_pickle=False) as b:
        need(set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),
             "Full native focal registry must exactly equal shared compact original IDs and coordinates")
    datasets={}
    for key,r in p["full_path_id_order_refs"].items():
        check("native_full_corpus_row_count")
        with np.load(bound(r),allow_pickle=False) as z: rows=len(z["record_ids"])
        datasets[key]=dict(rows=rows,dataset_ref=p["native_dataset_refs"][key],row_order_ref=r)
    labels={s:ref(scratch/"contrastive"/s/"labels.pkl") for s in ("train","val")}
    result=dict(p,schema="american_fresh_ARGformer_preparation_v1",focals_ref=spec["focal_registry_ref"],
                native_generated_focals_ref=p["focals_ref"],native_focals_equal_shared_registry=True,
                datasets=datasets,reference_label_refs=labels,MLM_optimizer_updates=5*(datasets["mlm/train"]["rows"]//512),
                contrastive_optimizer_updates=datasets["contrastive/train"]["rows"]//4096,
                disposable_scratch_directory=str(scratch.relative_to(ROOT)),
                disposable_only_after_final_prediction_and_successful_outer_wait=True,
                source_refs=spec["requirements"]["source_refs"],native_config=cfg)
    need(result["MLM_optimizer_updates"]>0 and result["contrastive_optimizer_updates"]>0,"No zero-update native fit")
    return Writer(out,spec["limits"]["max_output_bytes"],check,create=False).json("preparation_completed.json",result)


def build_flare_spec(*,stage,inference_ref,focal_preparation_ref,output_directory,limits,
                     conversion_ref=None,native_runtime=None,max_sites=5000000,_fresh=True):
    need(stage in ("prepare","native"),"Explicit FLARE preparation/native stage required")
    spec=_base("flare",stage,inference_ref,focal_preparation_ref,output_directory,limits,fresh=_fresh)
    need(type(max_sites) is int and max_sites>0,"Finite complete marker cap required")
    spec["max_sites"]=max_sites
    if stage=="native":
        converted=read(conversion_ref)
        common=importlib.import_module("arggnn.american_c3_flare_common_v1")
        need(converted.get("status")=="completed_C3_FLARE_observation_conversion"
             and converted.get("parent_id")==spec["parent_id"]
             and converted.get("observation_ref")==spec["observation_ref"]
             and converted.get("focal_registry_ref")==spec["focal_registry_ref"]
             and converted.get("query_ancestry_truth_opened") is False
             and converted.get("native_options")==common.NATIVE_OPTIONS
             and converted.get("projection")==common.PROJECTION
             and converted.get("native_source_commit")==common.PIN_COMMIT
             and converted.get("reference_haplotypes")==600 and converted.get("query_haplotypes")==1000
             and converted.get("common_focal_records")==10000
             and converted.get("common_focal_registry_unchanged") is True,"Actual shared FLARE conversion required")
        need(type(native_runtime) is dict and set(native_runtime)=={"binary_provenance_ref","jar_ref","java_executable","java_executable_sha256","java_version_output_sha256"},
             "Explicit qualified unmodified FLARE/Java runtime required")
        spec.update(conversion_ref=copy.deepcopy(conversion_ref),native_runtime=copy.deepcopy(native_runtime))
    else: need(conversion_ref is native_runtime is None,"Fresh FLARE conversion has no native predecessor")
    return spec


def run_flare(spec,*,bounded_runner=None,resource_check):
    out=validate_spec(spec)
    need(spec["method"]=="flare" and callable(resource_check) and os.environ.get("CUDA_VISIBLE_DEVICES")=="",
         "Admitted CPU-only FLARE callback required")
    if spec["stage"]=="native": return _flare_native(spec,out,bounded_runner,resource_check)
    module=importlib.import_module("arggnn.american_c3_flare_convert_v1")
    source=read(spec["inference_ref"])
    known=source["paper_known_parameters"]
    observed=read(spec["observation_ref"])
    need(known["recombination_rate_per_bp_generation"]==1e-8
         and known["sequence_length_bp"]==30000000
         and observed["sample_unit"]==source["declared_conventions"]["sample_unit"]=="haploid"
         and observed["counts_haploid"]==known["sample_counts_as_written"],
         "Actual public generation rate/units/counts must bind the shared map")
    # Reconstruct only the already declared generation config, never reopen truth.
    # The exact biological rate/counts originate in the pinned generation recipe.
    simulation=dict(schema="american_c3_simulation_reconstruction_v1",parent_id=spec["parent_id"],split=spec["split"],
                    sequence_length_bp=known["sequence_length_bp"],
                    recombination_rate=known["recombination_rate_per_bp_generation"],
                    sample_unit=observed["sample_unit"],sample_counts=copy.deepcopy(known["sample_counts_as_written"]),
                    public_inference_lineage_ref=spec["inference_ref"],derived_from_declared_generation_conventions=True)
    w=Writer(out,spec["limits"]["max_output_bytes"],resource_check,completion_name="conversion_completed.json")
    simulation_ref=w.json("recorded_simulation_map_parameters.json",simulation)
    cfg=dict(module.DEFAULTS,execution_enabled=True,observation_ref=spec["observation_ref"],
             simulation_config_ref=simulation_ref,focal_registry_ref=spec["focal_registry_ref"],max_sites=spec["max_sites"],
             max_haplotypes=1600,output_cap_bytes=spec["limits"]["max_output_bytes"]-2**20)
    return module.execute(cfg,w)


def _flare_native(spec,out,runner,check):
    need(callable(runner),"Injected bounded native runner required for Java")
    common=importlib.import_module("arggnn.american_c3_flare_common_v1")
    rt=spec["native_runtime"]
    binary=read(rt["binary_provenance_ref"])
    need(binary.get("schema")=="root_qualified_FLARE_binary_v1" and binary.get("root_authorizer")=="root_agent"
         and binary.get("repository")==common.PIN_REPOSITORY and binary.get("tag")==common.PIN_TAG
         and binary.get("source_commit")==common.PIN_COMMIT and binary.get("native_sources_modified") is False
         and binary.get("source_commit_build_verified") is True and binary.get("reported_version")==common.EXPECTED_VERSION
         and binary.get("jar_ref")==rt["jar_ref"],"Qualified unchanged FLARE0.6.0 binary required")
    jar=path(rt["jar_ref"]["path"])
    need(ref(jar)==rt["jar_ref"],"Exact native JAR bytes required")
    java=Path(rt["java_executable"])
    need(java.is_absolute() and os.access(java,os.X_OK)
         and common.native_file_hash(java.resolve())==rt["java_executable_sha256"],"Qualified Java executable required")
    need(all(not os.environ.get(k) for k in ("JAVA_TOOL_OPTIONS","_JAVA_OPTIONS","JDK_JAVA_OPTIONS")),"Unrecorded Java options rejected")
    converted=read(spec["conversion_ref"])
    markers,samples=converted["native_expected_markers"],converted["query_haplotypes"]//2
    plan=common.gzip_bound(4194304+markers*(256+64*samples))
    caps={".anc.vcf.gz":plan,".model":2**20,".global.anc.gz":2**20,".log":16*2**20}
    need(sum(caps.values())+2*2**30<=spec["limits"]["max_output_bytes"],"Whole native FLARE output/tmp/seal bound exceeds cap")
    w=Writer(out,spec["limits"]["max_output_bytes"],check)
    for name in ("java_tmp","java_home","native"): (out/name).mkdir()
    flags=["-XX:-UsePerfData","-XX:ActiveProcessorCount=4","-Djava.io.tmpdir="+str(out/"java_tmp"),
           "-Duser.home="+str(out/"java_home"),"-XX:ErrorFile="+str(out/"java_home/hs_err_pid%p.log")]
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,resource.getrlimit(resource.RLIMIT_CORE)[1]))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="")
    version_log=out/"java_version.txt"
    wait=runner([str(java),*flags,"-version"],str(out),env,
                dict(spec["limits"],max_seconds=min(20,spec["limits"]["max_seconds"])),str(version_log))
    validate_wait(wait)
    need(version_log.stat().st_size<=65536 and hashlib.sha256(version_log.read_bytes()).hexdigest()==rt["java_version_output_sha256"],"Actual exact qualified Java version output required")
    args=[str(java),*flags,"-Xmx16g","-Duser.language=en","-Duser.country=US","-jar",str(jar)]
    from arggnn.american_c3_common_v1 import bound
    args += [key+"="+str(bound(converted[r])) for key,r in
             (("ref","ref_vcf_ref"),("ref-panel","ref_panel_ref"),("gt","gt_vcf_ref"),("map","map_ref"))]
    args += ["out="+str(out/"native/flare")]
    args += [k+"="+(str(v).lower() if type(v) is bool else str(v)) for k,v in common.NATIVE_OPTIONS.items()]
    argv_ref=w.json("native_argv.json",dict(argv=args,conversion_ref=spec["conversion_ref"],native_options=common.NATIVE_OPTIONS,
                                          no_model_or_gt_ancestries_supplied=True))
    wait=runner(args,str(out),env,copy.deepcopy(spec["limits"]),str(out/"native_stdout.txt"))
    validate_wait(wait)
    native_refs={}
    need({p.name for p in (out/"native").iterdir()}=={"flare"+s for s in caps},"Exact native output product set required")
    for suffix,cap in caps.items():
        p=out/"native"/("flare"+suffix)
        need(p.is_file() and 0<p.stat().st_size<=cap,"Complete bounded unmodified FLARE product required")
        native_refs[suffix]=ref(p)
    need(common.EXPECTED_VERSION in path(native_refs[".log"]["path"]).read_text(),"Exact reported native FLARE version required")
    w=Writer(out,spec["limits"]["max_output_bytes"],check,create=False)
    native_ref=w.json("native_completed.json",dict(status="completed_C3_native_FLARE",conversion_ref=spec["conversion_ref"],
        parent_id=spec["parent_id"],class_order=["EUR","EAS","AFR"],focal_registry_ref=spec["focal_registry_ref"],
        projection=common.PROJECTION,source_commit=common.PIN_COMMIT,native_options=common.NATIVE_OPTIONS,
        native_output_refs=native_refs,native_outputs_modified=False,query_ancestry_truth_opened=False,
        native_wait=wait,native_argv_ref=argv_ref,native_runtime=copy.deepcopy(rt),
        java_version_ref=ref(version_log),source_refs=spec["requirements"]["source_refs"],
        author_Table1_exact_reproduction=False))
    module=importlib.import_module("arggnn.american_c3_flare_evaluate_v1")
    cfg=dict(module.DEFAULTS,native_run_ref=native_ref,focal_registry_ref=spec["focal_registry_ref"],
             max_markers=spec["max_sites"],write_derived_marker_tracts=False,output_cap_bytes=spec["limits"]["max_output_bytes"])
    seal_writer=Writer(out/"seal",min(64*2**20,spec["limits"]["max_output_bytes"]),check,completion_name="prediction_seal.json")
    return module.seal_native(cfg,seal_writer)


def argformer_factory(**kwargs):
    """ROOT workflow.materialize descriptor, with no scientific payload access."""
    spec=build_argformer_spec(**kwargs)
    return dict(module="arggnn.benchmarking.american_competitors",function="run_argformer",
                spec=spec,python=".runtime/env/bin/python",source_refs=spec["requirements"]["source_refs"])


def flare_factory(**kwargs):
    """ROOT workflow.materialize descriptor, with no launch or truth access."""
    spec=build_flare_spec(**kwargs)
    return dict(module="arggnn.benchmarking.american_competitors",function="run_flare",
                spec=spec,python=".runtime/env/bin/python",source_refs=spec["requirements"]["source_refs"])
