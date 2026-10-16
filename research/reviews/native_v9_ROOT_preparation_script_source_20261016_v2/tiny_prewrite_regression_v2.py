"""Tiny inert regressions for the two v2 prewrite corrections. No real guard."""
from pathlib import Path
import ast,hashlib,importlib,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_v9_ROOT_preparation_script_source_20261016_v2"

def cell(value):
    def inner():return value
    return inner.__closure__[0]

def run():
    assert sys.dont_write_bytecode and sys.flags.no_user_site
    original=importlib.import_module("research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2.common_v1")
    module=importlib.import_module(HERE.replace("/",".")+".prepare_root_v1")
    science_ref=original.ref("research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/science.template.json")
    cfg=original.read(science_ref)
    assert cfg==dict(schema="native_VAL000_GPU0_borrowed32_science_v1",scientific_protocol=original.SCIENCE,
        max_checkpoint_files=2,max_checkpoint_bytes=3*2**30)
    assert "output_cap_bytes" not in cfg
    tree=ast.parse((ROOT/HERE/"prepare_root_v1.py").read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=="validate_storage_v9")
    view=eval(compile(ast.Expression(call.args[1]),"<actual_v2_config_argument>","eval"),
        dict(c=types.SimpleNamespace(CAPS=original.CAPS),dict=dict))
    def boundary(value):
        assert value["output_cap_bytes"]==original.CAPS["reservation_bytes"],"Exact real guard output cap"
    boundary(view)
    assert view==dict(output_cap_bytes=7*2**30)
    try:boundary(cfg)
    except KeyError as error:assert error.args==("output_cap_bytes",)
    else:raise AssertionError("Genuine original scientific config incorrectly satisfies runtime guard")
    try:boundary(dict(output_cap_bytes=1))
    except AssertionError:pass
    else:raise AssertionError("Wrong guard output cap accepted")
    nested=next(v for v in module.prepare_root_v1.__code__.co_consts
        if isinstance(v,types.CodeType) and v.co_name=="save")
    value=dict(SYNTHETIC_ONLY=True)
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    projection=len(raw)+8192
    synthetic_write_charge=len(raw)+4096
    checks=[]
    for block_at in (None,1,2,3):
        events=[];saved=[]
        # These are synthetic charged-size rows. Whole7GiB still fits in every
        # negative, but the real original separate AUX expression must reject.
        charged=[100 if block_at is None else original.CAPS["auxiliary_bytes"]-projection-(block_at-1)*synthetic_write_charge+1]
        spec=dict(output_directory="SYNTHETIC_OUTPUT",worker_auxiliary_scope="SYNTHETIC_AUX")
        sizes=lambda path,**kwargs:[(("SYNTHETIC_METADATA",),100 if path=="SYNTHETIC_OUTPUT" else charged[0],"regular")]
        census_namespace=dict(original.charged_census_v1.__globals__,local=lambda path:path,_sizes=sizes)
        census=types.FunctionType(original.charged_census_v1.__code__,census_namespace)
        assert census.__code__ is original.charged_census_v1.__code__
        def checked_census(actual_spec,*,required_write_bytes):
            events.append("actual_original_census_code_on_synthetic_sizes")
            assert actual_spec is spec and required_write_bytes==projection
            return census(actual_spec,required_write_bytes=required_write_bytes)
        def fake_save(aux,name,actual_value):
            events.append("fake_metadata_save")
            assert actual_value==value
            saved.append(name);charged[0]+=synthetic_write_charge
            return dict(path="SYNTHETIC/"+name,sha256="0"*64)
        def fake_budget(storage,*,boundary,required_write_bytes):
            events.append("fake_whole_budget")
            assert boundary is True and required_write_bytes==projection
        fake_c=types.SimpleNamespace(original=types.SimpleNamespace(charged_census_v1=checked_census,save=fake_save))
        fake_os=types.SimpleNamespace(O_RDONLY=0,O_DIRECTORY=1,open=lambda *args:91,
            fsync=lambda fd:events.append("fake_dir_fsync"),close=lambda fd:None)
        closure=dict(aux="SYNTHETIC_AUX",c=fake_c,g=types.SimpleNamespace(budget_check_v9=fake_budget),
            original_spec=spec,require=original.require,storage=dict(SYNTHETIC_ONLY=True),tick=lambda:None)
        save=types.FunctionType(nested,dict(module.prepare_root_v1.__globals__,os=fake_os),
            closure=tuple(cell(closure[name]) for name in nested.co_freevars))
        assert save.__code__ is nested
        rejected=False
        for i in range(1,4):
            try:save("synthetic_"+str(i)+".json",value)
            except ValueError as error:
                assert block_at==i and "separate output/AUX and whole7GiB" in str(error)
                rejected=True;break
        if block_at is None:
            assert len(saved)==3 and events.count("fake_whole_budget")==3 and not rejected
        else:
            assert rejected and len(saved)==block_at-1
            assert events[-1]=="actual_original_census_code_on_synthetic_sizes"
            assert events.count("fake_whole_budget")==block_at-1
            assert 100+charged[0]+projection<original.CAPS["reservation_bytes"]
        checks.append(dict(block_at_synthetic_write=block_at,prior_fake_writes=len(saved),
            status="PASS_ALL_THREE_FAKE_WRITES" if block_at is None else "PASS_AUX_OVERFLOW_REJECTED_BEFORE_WRITE"))
    assert not any(name in sys.modules for name in ("torch","torch_geometric","composer"))
    return dict(schema="native_v9_ROOT_prep_v2_tiny_inert_prewrite_regression",
        status="PASS_SOFTWARE_ONLY",genuine_static_scientific_config_ref=science_ref,
        actual_v2_argument_view=view,old_missing_key_reproduced=True,wrong_cap_rejected=True,
        actual_original_census_code_object_on_synthetic_sizes=True,
        actual_nested_v2_save_code_object_used_with_explicit_fakes=True,checks=checks,
        no_actual_guard_authority_capacity_accounting_path_execution_or_resource_probe=True,
        script_ref=original.ref(HERE+"/prepare_root_v1.py"))
if __name__=="__main__":
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit --run for inert-only regression")
    print(json.dumps(run(),sort_keys=True,indent=2))
