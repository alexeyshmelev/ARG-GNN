"""VAL000's single missing operational auth field, never a corpus/science rewrite.

The original VAL serializer omitted a budget field introduced by a later TRAIN
producer. Derive its genuine original serializer bound from authenticated config.
The exact inherited auth body still authenticates all consumed bytes once.
"""
from pathlib import Path
import ast,sys,types
from research.reviews.native_VAL000_full_state_resume_source_20261016_v9 import amendment_contract_v1 as c
SOURCE=c.HERE+"/corpus_metadata_bridge_v1.py"
AUTH_REF={"path":"research/reviews/American_C3_native_MLM_amendment_v12/corpus_auth_v2.py","sha256":"2f02c99d49502493abab82e5b6c08c8abc782538d7d821a71cd5116f62b0cd98"}
PRODUCER_REF={"path":"src/arggnn/american_c3_paths_v1.py","sha256":"3b32f16e0a81160852e5b93fe9033abf9bc2eeb56c3b15ca552d73f4aacf374c"}
PREP_ADMISSION_REF={"path":"research/reviews/prospective_980GB_capacity_source_v1/admitted_native_paths_v1/admission.json","sha256":"b3b2344b864dd4140d82b9ca2ca04d3eee5fcbd69f01f2c290d54d50af57d4cb"}
FIELD="native_serialization_bound_bytes"
EXPECTED_BOUND=15218324396
EXPECTED_SERIALIZER_AST="3 * 5 * t + 256 * p + 5 * o + 7 * 64 * v + 16 * 16777216"

def _positive_int(value,message):
    c.require(type(value) is int and value>0,message)
    return value

def _exact_record(record,message):
    c.require(type(record) is dict and set(record)=={"path","sha256"}
        and type(record["path"]) is str and type(record["sha256"]) is str
        and len(record["sha256"])==64 and all(x in "0123456789abcdef" for x in record["sha256"]),message)
    c.local(record["path"])

def preparation_metadata_view_v1(prep,prepared_dir):
    c.active_amendment_v1()
    c.require(type(prep) is dict and prep==c.read(c.PREP_REF)
        and prepared_dir==c.local(c.PREP_OUTPUT) and FIELD not in prep,
        "Exact immutable actual VAL000 preparation and one missing field only")
    counts=c.read(c.COUNTS_REF);cfg=c.read(c.PREP_CONFIG_REF);admission=c.read(PREP_ADMISSION_REF)
    c.require(admission["source_ref"]==PRODUCER_REF and admission["config_ref"]==c.PREP_CONFIG_REF
        and admission["output_directory"]==c.PREP_OUTPUT
        and counts["preparation_ref"]==c.PREP_REF and counts["preparation_config_ref"]==c.PREP_CONFIG_REF,
        "Bound comes from actual original admitted serializer and exact counted corpus")
    tree=ast.parse(c.raw(PRODUCER_REF).decode("utf-8"))
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="storage_bound"]
    c.require(len(definitions)==1,"Exact original serializer bound definition")
    returns=[n for n in ast.walk(definitions[0]) if isinstance(n,ast.Return)]
    c.require(len(returns)==1 and ast.dump(returns[0].value,include_attributes=False)==
        ast.dump(ast.parse(EXPECTED_SERIALIZER_AST,mode="eval").body,include_attributes=False),
        "Operational bridge derives exactly the original serializer bound expression")
    p,t,o,v=(_positive_int(cfg[k],"Actual positive original serializer dimension") for k in
        ("max_unique_paths","max_path_tokens","max_path_occurrences","max_vocabulary_nodes"))
    maximum=3*5*t+256*p+5*o+7*64*v+16*16777216
    c.require(maximum==EXPECTED_BOUND and maximum<=_positive_int(cfg["output_cap_bytes"],"Original serialization allocation"),
        "Genuine original serialization bound, not new storage/output capacity")
    vocabulary=_positive_int(prep["vocabulary_nodes"],"Actual original vocabulary node count")
    c.require(vocabulary==counts["vocabulary_nodes"]==107140 and vocabulary<=v,
        "Actual observed vocabulary preserved")
    consumed=("mlm/train","mlm/val")
    c.require(type(prep["native_dataset_refs"]) is dict and type(counts["datasets"]) is dict
        and set(prep["native_dataset_refs"])==set(counts["datasets"])
        and set(consumed)<=set(counts["datasets"]),"Complete original native dataset metadata")
    file_bounds={}
    for name,item in counts["datasets"].items():
        c.require(type(name) is str and name.count("/")==1 and type(item) is dict,
            "Original view/split record schema")
        view,split=name.split("/")
        c.require(view in ("mlm","contrastive","query_paths") and split in ("train","val"),"Exact native view/split namespace")
        record=prep["native_dataset_refs"][name];_exact_record(record,"Exact original path/SHA record")
        expected=c.local(c.PREP_OUTPUT)/view/split/("raw_"+split+"_sequences_and_vocab.pkl")
        c.require(c.local(record["path"])==expected and item["dataset_ref"]==record,"Exact original counted raw pickle identity")
        rows=_positive_int(item["rows"],"Actual complete source rows")
        tokens=_positive_int(item["raw_path_tokens"],"Actual complete raw tokens")
        raw_bytes=_positive_int(item["raw_pickle_bytes"],"Actual count receipt serialized bytes")
        planned=5*tokens+32*rows+64*vocabulary+16777216
        c.require(rows<=p and rows<=tokens<=t and raw_bytes<=planned<=maximum,
            "Authenticated actual serialized bytes fit unchanged original per-file and global bounds")
        file_bounds[name]=dict(rows=rows,raw_path_tokens=tokens,counted_pickle_bytes=raw_bytes,
            original_producer_planned_bytes=planned)
    _exact_record(prep["vocab_ref"],"Exact standalone vocabulary path/SHA record")
    c.require(c.local(prep["vocab_ref"]["path"])==prepared_dir/"vocab.pkl"
        and 64*vocabulary+16777216<=maximum,"Exact original standalone vocabulary serializer bound")
    view=dict(prep);view[FIELD]=maximum
    c.require(set(view)==set(prep)|{FIELD} and all(view[k]==v for k,v in prep.items()),
        "Only one missing operational int added to in-memory metadata copy")
    provenance=dict(schema="actual_VAL000_original_serializer_metadata_bridge_v1",
        source_ref=c.ref(SOURCE),inherited_authentication_source_ref=AUTH_REF,
        original_preparation_ref=c.PREP_REF,actual_preparation_admission_ref=PREP_ADMISSION_REF,
        original_producer_ref=PRODUCER_REF,original_preparation_config_ref=c.PREP_CONFIG_REF,
        actual_counts_ref=c.COUNTS_REF,injected_field=FIELD,injected_value_bytes=maximum,
        original_serializer_bound_expression=EXPECTED_SERIALIZER_AST,
        all_six_counted_raw_files_fit_original_planned_bounds=file_bounds,
        standalone_vocabulary_original_planned_bytes=64*vocabulary+16777216,
        original_preparation_modified=False,corpus_modified_or_decoded=False,
        original_path_and_sha_records_unchanged=True,scientific_inputs_or_author_algorithm_changed=False,
        new_storage_capacity_or_reservation=False)
    return view,provenance

def install_private_authentication_bridge_v1():
    c.active_amendment_v1()
    c.require("corpus_auth_v2" not in sys.modules and "_VAL000_original_corpus_auth_v3" not in sys.modules,
        "Fresh private interpreter-local auth bridge, no module replacement/replay")
    preparation_metadata_view_v1(c.read(c.PREP_REF),c.local(c.PREP_OUTPUT))
    inherited=c.load(AUTH_REF,"_VAL000_original_corpus_auth_v3")
    def authenticate_inputs(prep,prepared_dir):
        view,provenance=preparation_metadata_view_v1(prep,prepared_dir)
        # Original streaming SHA/stability/canonical/single-link/size/shadow checks.
        result=inherited.authenticate_inputs(view,prepared_dir)
        result["operational_preparation_metadata_bridge"]=provenance
        return result
    module=types.ModuleType("corpus_auth_v2",__doc__)
    module.__file__=str(c.ROOT/SOURCE)
    module.authenticate_inputs=authenticate_inputs
    module.confirm_unchanged=inherited.confirm_unchanged
    c.active_amendment_v1()
    sys.modules["corpus_auth_v2"]=module
    return dict(bridge_source_ref=c.ref(SOURCE),inherited_source_ref=AUTH_REF,
        original_auth_functions_unchanged=True,private_current_interpreter_only=True,metadata_only_preflight_completed=True)
