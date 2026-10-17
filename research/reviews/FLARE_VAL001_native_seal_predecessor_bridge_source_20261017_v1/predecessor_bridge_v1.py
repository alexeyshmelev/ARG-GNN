"""Inactive exact v4-conversion predecessor bridge. No IO or guard on import."""
import ast,copy
HERE="research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v1"
CONTRACT_REF=dict(path="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1/root_contract_v1.py",sha256="301d36326a34e4ee23717cb5d033dee5a11cb836b679088a47b4eaa7120422e3")
V4_DRIVER_REF=dict(path="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v4/root_driver_v1.py",sha256="bcd7eba4f02dd7425a4d43ded8df71402e83cb5ab297886172a26a1d6d395232")
V4_ENTRY_REF=dict(path="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v4/worker_entry_v1.py",sha256="cc6a9b5b98034985876a88ecff6c0da6d0b16f2d23ed9195b11b4a0c0a78751b")
V4_MANIFEST_REF=dict(path="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v4/source_manifest.v1.json",sha256="3365dcad1e1398c94ded1ab81b3bc6cd75ca096d9462bcaf59fa81800fef7865")
INPUT_REF=dict(path="research/reviews/FLARE_remaining_ROOT_actual_VAL001_convert_3h_setupv4_20261017_v1/input.json",sha256="95ca4f5bff8bd6723a0cfcfeb233e7517db10d24b443e8b70c78d5b1979492bd")
WAIT_REF=dict(path="research/reviews/FLARE_remaining_ROOT_actual_VAL001_convert_3h_setupv4_20261017_v1/worker_aux/actual_wait.json",sha256="51613247e73166fbbff8e4c1e0f70130924f4e73be9ba86a0ad1ed631bf50391")
COMPLETED_REF=dict(path="data/FLARE_original_remaining_VAL001_convert_3h_setupv4_20261017_v1/completed.json",sha256="3d8b6c3711bda0bd0224d57bd3210aabdcebdb707a26051b3b21bdd149b33366")
SELECTOR_REF=dict(path="research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json",sha256="e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642")
PARENT="american_c3_val_001"
KEY="H041/FLARE-original-convert-FLARE_remaining_ROOT_actual_VAL001_convert_3h_setupv4_20261017_v1"

def require(ok,message):
    if not ok:raise ValueError(message)

def derive_stage_node_v1(node):
    """Only three current-source comparison operands change; original proof is intact."""
    require(isinstance(node,ast.FunctionDef)and node.name=="stage_evidence"and not node.decorator_list,"Exact original stage_evidence")
    patterns=[(ast.parse(text,mode="eval").body,name)for text,name in (
        ("ref(DRIVER)","V4_DRIVER_REF"),("ref(ENTRY)","V4_ENTRY_REF"),
        ("sources()['source_sha256']","V4_SOURCE_SHA256"))]
    counts=[0,0,0]
    class ExactSources(ast.NodeTransformer):
        def visit(self,n):
            for i,(pattern,name)in enumerate(patterns):
                if ast.dump(n,include_attributes=False)==ast.dump(pattern,include_attributes=False):
                    counts[i]+=1
                    return ast.copy_location(ast.Name(id=name,ctx=ast.Load()),n)
            return super().visit(n)
    result=ExactSources().visit(copy.deepcopy(node));result.name="_exact_v4_stage_evidence"
    require(counts==[1,1,1],"Exactly three original predecessor-source operand sites")
    return ast.fix_missing_locations(result)

def close_acceptance_v1(e,read):
    """All certificates below must be genuine new ROOT receipts, never templates."""
    record=e.get("ordinary_close_terminal_acceptance_ref")
    require(isinstance(record,dict)and set(record)=={"path","sha256"},"Genuine ROOT ordinary-close acceptance ref required")
    a=read(record)
    expected=dict(schema="ROOT_FLARE_exact_v4_conversion_close_acceptance_v1",root_authorizer="root_agent",
        accepted=True,input_ref=INPUT_REF,actual_wait_ref=WAIT_REF,completed_ref=COMPLETED_REF,
        conversion_outer_terminal_ref=e["outer_terminal_ref"],ordinary_close_receipt_ref=e["ordinary_close_receipt_ref"],
        reservation_key=KEY,original_files_scopes_charges_clocks_and_failure_history_preserved=True,
        all_conversion_future_write_resume_relaunch_capabilities_revoked=True,native_or_seal_execution_authorized=False)
    require(all(type(a.get(k))is type(v)and a.get(k)==v for k,v in expected.items()),"Exact genuine v4 conversion/close acceptance, no inherited execution authority")
    close_outer=read(a["ordinary_close_outer_terminal_ref"])
    exit0=any(type(close_outer.get(k))is int and close_outer[k]==0 for k in ("actual_exit_code","actual_tool_exit_code"))
    require(close_outer.get("actual_terminal_consumed")is True and exit0
        and (close_outer.get("ordinary_close_receipt_ref")==e["ordinary_close_receipt_ref"]or close_outer.get("receipt_ref")==e["ordinary_close_receipt_ref"]),
        "Genuine consumed ordinary-close outer0 bound to this exact receipt")
    final=read(a["final_original_journal_acceptance_ref"])
    require(final.get("root_authorizer")=="root_agent"and final.get("final_original_journal_accepted")is True
        and final.get("ordinary_close_receipt_ref")==e["ordinary_close_receipt_ref"]and final.get("reservation_key")==KEY
        and final.get("storage_policy_selector_ref")==SELECTOR_REF,
        "Actual final original current-child journal acceptance for exact close")
    return a

def build_stage_bridge_v1(original,derived,read):
    def stage_evidence(record,*,stage,parent,completion_ref):
        if stage!="convert":
            return original(record,stage=stage,parent=parent,completion_ref=completion_ref)
        require(parent==PARENT and completion_ref==COMPLETED_REF,"Only exact genuine completed VAL001 v4 conversion")
        e=read(record)
        require(e.get("input_ref")==INPUT_REF and e.get("actual_wait_ref")==WAIT_REF
            and e.get("completion_ref")==COMPLETED_REF and e.get("stage")=="convert"and e.get("parent_id")==PARENT,
            "Exact immutable original v4 input/wait/completion; no arbitrary prior source")
        close_acceptance_v1(e,read)
        return derived(record,stage=stage,parent=parent,completion_ref=completion_ref)
    return stage_evidence

def install_stage_bridge_v1(c,authenticated_base):
    # Original exact stable source SHA, module and callable/code identity checks.
    authenticated_base.functions_from_source(authenticated_base,
        dict(path="research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1/root_mutation_memo_v1.py",sha256="2923df9f5a48e76da0b59e6394f31d0b8f901c214a76cc62d65969d010655d98"),
        ("source_bytes","functions_from_source"))
    original,node=authenticated_base.functions_from_source(c,CONTRACT_REF,("stage_evidence",))["stage_evidence"]
    v4_manifest=c.read(V4_MANIFEST_REF)
    require(isinstance(v4_manifest.get("source_sha256"),dict),"Exact v4 closure mapping")
    namespace=dict(original.__globals__,V4_DRIVER_REF=V4_DRIVER_REF,V4_ENTRY_REF=V4_ENTRY_REF,V4_SOURCE_SHA256=v4_manifest["source_sha256"])
    derived=derive_stage_node_v1(node)
    exec(compile(ast.Module(body=[derived],type_ignores=[]),HERE+"/predecessor_bridge_v1.py#exact-original-stage-evidence","exec",dont_inherit=True,optimize=0),namespace)
    c.stage_evidence=build_stage_bridge_v1(original,namespace[derived.name],c.read)
