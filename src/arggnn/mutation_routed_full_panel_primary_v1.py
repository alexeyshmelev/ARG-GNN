"""H038 full686 C4 external target boundary and fast Present182 scoring."""
from __future__ import annotations
from dataclasses import dataclass
import io
from pathlib import Path
import numpy as np
from .mutation_routed_fast_primary_v1 import verify_seal_v1, mass_cache_v1, score_primary_v1
PRIMARY="primary_present_day_182"


def sealed_probabilities_v1(record,parent,reader):
    current=getattr(reader,"current_probability_seal_v1",None)
    if current is not None:
        probabilities=current(record,parent)
        if probabilities is not None:return probabilities
    return verify_seal_v1(record,parent,reader)


OUTER_POLICY_PATH="research/reviews/expanded_full_panel_v1_root/outer_score_policy.json"
OUTER_SUPPORT_PREFIX="research/reviews/expanded_full_panel_v1_root/outer_support"
REGISTRATION_REF={
    "path":"research/reviews/expanded_full_panel_v1_root/registration.json",
    "sha256":"3e3f5fab53cc9fb6142e0b11926deb9a2f8e91ddabafc9b5e2f9c5d19c2db9ac"}
GENERATION_REF={
    "path":"configs/expanded_full_panel_v1.json",
    "sha256":"662ba8e4613ce85b9aaec9909e873c9cddf9c62672b895e7d9e02f851d518d08"}
OUTER_PARENT_IDS=tuple("ap_expanded_outer_%03d"%i for i in range(8))
OUTER_SUPPORT_FIELDS=("positions","edges","retained_indexes","query_slots","raw_query_nodes","cohorts")


def _outer_record_v1(record,*,suffix,expected_path=None):
    if not isinstance(record,dict) or set(record)!={"path","sha256"}:
        raise ValueError("Exact outer metadata/seal path/SHA record required")
    path,digest=record["path"],record["sha256"]
    if (not isinstance(path,str) or Path(path).is_absolute() or Path(path).as_posix()!=path
            or ".." in Path(path).parts or not path.endswith(suffix)
            or not isinstance(digest,str) or len(digest)!=64
            or any(c not in "0123456789abcdef" for c in digest)
            or expected_path is not None and path!=expected_path):
        raise ValueError("Canonical externally frozen outer record required")
    if expected_path is None and any(part in {"truth","oracle"} for part in Path(path).parts):
        raise ValueError("Outer policy, support and predictions cannot read truth/oracle paths")
    return record


def _same_outer_support_v1(candidate,authoritative):
    if (not isinstance(candidate,dict) or candidate.get("parent_id")!=authoritative["parent_id"]
            or candidate.get("split")!="outer"
            or any(key not in candidate or not np.array_equal(candidate[key],authoritative[key])
                   for key in OUTER_SUPPORT_FIELDS)):
        raise ValueError("Outer parent support must equal frozen observed metadata")


@dataclass(frozen=True)
class VerifiedOuterProductSealsV1:
    """Transport only: the public constructor is never an authorization boundary."""
    records: tuple
    parent_ids: tuple
    product_names: tuple
    products: object=None
    parents: object=None
    outer_policy_ref: object=None


def _verified_outer_matrix_v1(products,parents,reader,*,outer_policy_ref,required_product_names=None):
    _outer_record_v1(outer_policy_ref,suffix=".json",expected_path=OUTER_POLICY_PATH)
    policy=reader.decode_json(outer_policy_ref)
    names=policy.get("required_product_names")
    if (policy.get("schema")!="mutation_routed_full_panel_outer_policy_v1"
            or policy.get("hypothesis_id")!="H038" or policy.get("root_authorizer")!="root_agent"
            or policy.get("status")!="frozen" or policy.get("outer_evaluation_authorized") is not True
            or policy.get("registration_ref")!=REGISTRATION_REF
            or policy.get("generation_config_ref")!=GENERATION_REF
            or policy.get("outer_parent_ids")!=list(OUTER_PARENT_IDS)
            or not isinstance(names,list) or not names
            or any(not isinstance(name,str) or not name for name in names)
            or len(set(names))!=len(names)
            or required_product_names is not None and tuple(required_product_names)!=tuple(names)):
        raise ValueError("Frozen root H038 outer authorization, registry and product declaration required")
    registration=reader.decode_json(REGISTRATION_REF)
    generation=reader.decode_json(GENERATION_REF)
    from .mutation_routed_full_panel_observations_v1 import validate_registry_v1
    roles=validate_registry_v1(generation["parent_registry"])
    if (registration.get("hypothesis_id")!="H038" or registration.get("config_ref")!=GENERATION_REF
            or registration.get("parent_registry")!=generation["parent_registry"]
            or tuple(roles["outer"])!=OUTER_PARENT_IDS):
        raise ValueError("Exact frozen full48 registration and eight outer parents required")
    support_refs=policy.get("outer_parent_support_refs")
    bindings=policy.get("outer_target_bindings")
    frozen_products=policy.get("product_seal_refs")
    if (not isinstance(parents,dict) or set(parents)!=set(OUTER_PARENT_IDS)
            or not isinstance(products,dict) or set(products)!=set(names)
            or not isinstance(support_refs,dict) or set(support_refs)!=set(OUTER_PARENT_IDS)
            or not isinstance(bindings,dict) or set(bindings)!=set(OUTER_PARENT_IDS)
            or not isinstance(frozen_products,dict) or set(frozen_products)!=set(names)):
        raise ValueError("Complete externally frozen products by registered eight-parent matrix required")
    records=[]
    for name in names:
        if (not isinstance(products[name],dict) or set(products[name])!=set(OUTER_PARENT_IDS)
                or not isinstance(frozen_products[name],dict)
                or set(frozen_products[name])!=set(OUTER_PARENT_IDS)
                or products[name]!=frozen_products[name]):
            raise ValueError("Every declared product must bind its exact eight distinct prediction seals")
        for parent_id in OUTER_PARENT_IDS:
            records.append(_outer_record_v1(products[name][parent_id],suffix=".npz"))
    if len({r["path"] for r in records})!=len(records):
        raise ValueError("One prediction archive cannot be reused across product/parent cells")
    for parent_id in OUTER_PARENT_IDS:
        _outer_record_v1(support_refs[parent_id],suffix=".json",
            expected_path=OUTER_SUPPORT_PREFIX+"/"+parent_id+".json")
        binding=bindings[parent_id]
        if not isinstance(binding,dict) or set(binding)!={"split","truth_npz","generation_receipt"} or binding["split"]!="outer":
            raise ValueError("Frozen same-parent outer truth binding required")
        _outer_record_v1(binding["truth_npz"],suffix=".npz",
            expected_path=generation["output_root"]+"/"+parent_id+"/truth/full_truth_intervals.npz")
        _outer_record_v1(binding["generation_receipt"],suffix=".json")
    authoritative={}
    for parent_id in OUTER_PARENT_IDS:
        support=reader.decode_json(support_refs[parent_id])
        if (not isinstance(support,dict)
                or set(support)!={"schema","parent_id","split","observations_only","truth_or_oracle_read",*OUTER_SUPPORT_FIELDS}
                or support["schema"]!="mutation_routed_full_panel_outer_support_v1"
                or support["parent_id"]!=parent_id or support["split"]!="outer"
                or support["observations_only"] is not True or support["truth_or_oracle_read"] is not False):
            raise ValueError("Frozen observed-only outer scoring support required")
        parent=dict(parent_id=parent_id,split="outer")
        for key in OUTER_SUPPORT_FIELDS:
            parent[key]=np.asarray(support[key],dtype=(
                np.float64 if key in {"positions","edges"} else str if key=="cohorts" else np.int64))
        if (parent["positions"].ndim!=1 or not len(parent["positions"])
                or parent["edges"].shape!=(len(parent["positions"])+1,)
                or parent["retained_indexes"].shape!=parent["positions"].shape
                or any(parent[k].shape!=(686,) for k in ("query_slots","raw_query_nodes","cohorts"))
                or not np.isfinite(parent["positions"]).all() or not np.isfinite(parent["edges"]).all()
                or np.any(np.diff(parent["edges"])<=0) or not 0<=parent["edges"][0]<parent["edges"][-1]<=2000000
                or len(set(parent["query_slots"]))!=686 or len(set(parent["raw_query_nodes"]))!=686
                or np.any((parent["query_slots"]<0)|(parent["query_slots"]>=952))
                or {c:int((parent["cohorts"]==c).sum()) for c in ("present_day","Bronze_Age","BAA","Neo","Yam")}
                    !={"present_day":182,"Bronze_Age":162,"BAA":14,"Neo":302,"Yam":26}):
            raise ValueError("Original complete686 observed identity and physical support required")
        _same_outer_support_v1(parents[parent_id],parent)
        authoritative[parent_id]=parent
    # Reverify all product bytes against authoritative support before any truth read.
    for name in names:
        for parent_id in OUTER_PARENT_IDS:
            verify_seal_v1(products[name][parent_id],authoritative[parent_id],reader)
    for parent_id in OUTER_PARENT_IDS:
        binding=bindings[parent_id]
        receipt=reader.decode_json(binding["generation_receipt"])
        if (receipt.get("parent_id")!=parent_id
                or receipt.get("files",{}).get("truth/full_truth_intervals.npz",{}).get("sha256")!=binding["truth_npz"]["sha256"]):
            raise ValueError("Frozen outer truth hash must bind same-parent generation provenance")
    proof=VerifiedOuterProductSealsV1(tuple(records),OUTER_PARENT_IDS,tuple(names),
        {name:{p:dict(products[name][p]) for p in OUTER_PARENT_IDS} for name in names},
        authoritative,dict(outer_policy_ref))
    return proof,bindings


class FullPanelExternalTargetV1:
    def __init__(self,parent,binding,purpose,reader,*,sealed_records=None,required_outer_products=None,outer_policy_ref=None):
        allowed={"TRAIN_loss":"train","VAL_after_sealed_predictions":"val","OUTER_after_all_product_seals":"outer"}
        if purpose not in allowed or parent["split"]!=allowed[purpose] or binding["split"]!=parent["split"]:
            raise ValueError("Strict target purpose/split required")
        if purpose=="OUTER_after_all_product_seals":
            if (not isinstance(sealed_records,VerifiedOuterProductSealsV1)
                    or outer_policy_ref is None or sealed_records.outer_policy_ref!=outer_policy_ref):
                raise ValueError("Separate frozen root policy and complete matrix required before outer truth")
            proof,outer_bindings=_verified_outer_matrix_v1(
                sealed_records.products,sealed_records.parents,reader,
                outer_policy_ref=outer_policy_ref,required_product_names=required_outer_products)
            if (sealed_records.records!=proof.records or sealed_records.parent_ids!=proof.parent_ids
                    or sealed_records.product_names!=proof.product_names):
                raise ValueError("Outer proof transport differs from independently reverified full matrix")
            _same_outer_support_v1(parent,proof.parents[parent["parent_id"]])
            if binding!=outer_bindings[parent["parent_id"]]:
                raise ValueError("Outer target must equal externally frozen same-parent truth binding")
            sealed_records=proof.records
        if purpose!="TRAIN_loss":
            if not sealed_records:raise ValueError("Complete prediction seals required before heldout truth")
            for record in sealed_records:reader.read(record)
        with np.load(io.BytesIO(reader.read(binding["truth_npz"])),allow_pickle=False) as archive:
            if set(archive.files)!={"intervals_bp","query_raw_node_ids","native_paths_1based","source_classes_0based"}:
                raise ValueError("Original complete interval truth schema required")
            self.intervals=archive["intervals_bp"].copy()
            self.c4=archive["source_classes_0based"].copy()
            ids=archive["query_raw_node_ids"].copy()
        if (self.intervals.dtype!=np.float64 or self.intervals.ndim!=2 or self.intervals.shape[1]!=2
                or not len(self.intervals) or self.intervals[0,0]!=0 or self.intervals[-1,1]!=2000000
                or not np.array_equal(self.intervals[:-1,1],self.intervals[1:,0])
                or not np.isfinite(self.intervals).all() or np.any(np.diff(self.intervals,axis=1)<=0)
                or self.c4.shape!=(len(self.intervals),686) or self.c4.dtype.kind not in "iu"
                or np.any((self.c4<0)|(self.c4>3)) or not np.array_equal(ids,parent["raw_query_nodes"])):
            raise ValueError("Exact full686 C4 interval/raw-query identity required")
        self.provenance=dict(record=binding["truth_npz"],purpose=purpose,sealed_records=sealed_records)
    def target_mass(self,parent,helpers):
        bins,rows,spans=helpers.intersections(parent["edges"],self.intervals)
        mass=np.zeros((len(parent["positions"]),686,4),dtype=np.float64)
        np.add.at(mass,(bins[:,None],np.arange(686)[None,:],self.c4[rows]),spans[:,None])
        if not np.allclose(mass.sum(-1),np.diff(parent["edges"])[:,None],atol=1e-8,rtol=0):
            raise ValueError("Original full physical span mass required")
        return mass


def means_v1(cells,*,full=False):
    result={}
    for name in cells[0]["groups"]:
        groups=[c["groups"][name] for c in cells]
        infinite=any(g.get("NLL_is_infinite",bool(g.get("infinite_nll_query_bp",0))) for g in groups)
        value=dict(accuracy=float(np.mean([g["accuracy"] for g in groups],dtype=np.float32)),
            nll=None if infinite else float(np.mean([g["nll"] for g in groups],dtype=np.float32)),
            NLL_is_infinite=infinite)
        if full:
            value.update({key:float(np.mean([g[key] for g in groups])) for key in ("brier","calibration_ece")})
        result[name]=value
    return result


class FullPanelPrimaryValidationV1:
    def __init__(self,parents,bindings,reader,helpers,cfg):
        self.parents={p:row for p,row in parents.items() if row["split"]=="val"}
        if len(self.parents)!=8:raise ValueError("Exactly8 registered internalVAL parents required")
        self.bindings,self.reader,self.helpers,self.cfg=bindings,reader,helpers,cfg
        self.caches={};self.full_calls=0

    def _verify(self,seals):
        records=dict(seals)
        if len(seals)!=8 or set(records)!=set(self.parents):
            raise ValueError("All8 current VAL probability seals required")
        probabilities={p:sealed_probabilities_v1(records[p],parent,self.reader) for p,parent in self.parents.items()}
        return records,probabilities

    def score(self,seals):
        records,probabilities=self._verify(seals)
        all_records=list(records.values())  # All8 verification above precedes any target.
        for p,parent in self.parents.items():
            if p not in self.caches:
                target=FullPanelExternalTargetV1(parent,self.bindings[p],"VAL_after_sealed_predictions",
                                                self.reader,sealed_records=all_records)
                self.caches[p]=mass_cache_v1(parent,target.target_mass(parent,self.helpers),
                    target_provenance=target.provenance,expected_present_queries=182)
        cells=[dict(parent_id=p,split="val",prediction_seal=records[p],
                    groups={PRIMARY:score_primary_v1(probabilities[p],self.caches[p])})
               for p in self.parents]
        return cells,means_v1(cells)

    def full_final(self,seals):
        if self.full_calls:raise ValueError("Full selected diagnostics already computed")
        records,probabilities=self._verify(seals)
        cells=[]
        for p,parent in self.parents.items():
            target=FullPanelExternalTargetV1(parent,self.bindings[p],"VAL_after_sealed_predictions",
                self.reader,sealed_records=list(records.values()))
            cohorts=parent["cohorts"]
            strata={PRIMARY:np.flatnonzero(cohorts=="present_day"),
                "secondary_ancient_504":np.flatnonzero(cohorts!="present_day"),"all_queries_686":np.arange(686)}
            strata.update({"age_stratum_"+c:np.flatnonzero(cohorts==c) for c in ("Bronze_Age","BAA","Neo","Yam")})
            groups={}
            for name,columns in strata.items():
                group=self.helpers.full_span_score(probabilities[p],parent,target,columns,self.cfg["metrics"])
                group.pop("per_query",None);group.pop("retained_original_grid_indexes",None)
                groups[name]=group
            cells.append(dict(parent_id=p,split="val",prediction_seal=records[p],groups=groups))
        self.full_calls+=1
        return cells,means_v1(cells,full=True)

    def receipt(self):
        return dict(primary_group=PRIMARY,VAL_parents=8,all_VAL_seals_before_truth=True,
            primary_mass_cache_bytes=sum(c.mass.nbytes for c in self.caches.values()),
            full_selected_diagnostic_calls=self.full_calls,target_mass_model_input=False,target_mass_VAL_loss=False)


def verify_outer_product_seals_v1(products,parents,reader,*,outer_policy_ref=None,required_product_names=None):
    """Verify external policy and full matrix; target construction repeats this gate."""
    proof,_=_verified_outer_matrix_v1(products,parents,reader,outer_policy_ref=outer_policy_ref,
        required_product_names=required_product_names)
    return proof
