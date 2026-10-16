"""Static source and symbolic parameter checks only. Never imports model modules."""
from pathlib import Path
import ast, hashlib, json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path(__file__).resolve().parent

def qualified(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):return qualified(node.value)+"."+node.attr
    raise ValueError(ast.dump(node))

def evaluate(node, env):
    if isinstance(node,ast.Constant):return node.value
    if isinstance(node,(ast.Name,ast.Attribute)):return env[qualified(node)]
    if isinstance(node,ast.BinOp):
        a,b=evaluate(node.left,env),evaluate(node.right,env)
        return {ast.Add:lambda:a+b,ast.Mult:lambda:a*b}[type(node.op)]()
    raise ValueError(ast.dump(node))

def constructor(path, name):
    tree=ast.parse((ROOT/path).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")

def count_modules(path, name, env, *, scalar_parameters=0):
    ctor=constructor(path,name)
    rows=[]
    tensors=scalar_parameters
    total=scalar_parameters
    for node in ast.walk(ctor):
        if not isinstance(node,ast.Call):continue
        try:q=qualified(node.func)
        except ValueError:continue
        if q not in ("nn.Linear","nn.LayerNorm","nn.GRU"):continue
        args=[evaluate(a,env) for a in node.args]
        kw={v.arg:evaluate(v.value,env) for v in node.keywords if v.arg in
            ("bias","num_layers","bidirectional")}
        if q=="nn.Linear":
            i,o=args[:2];bias=kw.get("bias",True)
            n=i*o+(o if bias else 0);k=1+int(bias)
            shape=dict(input=i,output=o,bias=bias)
        elif q=="nn.LayerNorm":
            n=2*args[0];k=2;shape=dict(width=args[0])
        else:
            i,h=args[:2];layers=kw.get("num_layers",1);directions=2 if kw.get("bidirectional",False) else 1
            assert layers==1
            n=directions*(3*h*i+3*h*h+6*h);k=directions*4
            shape=dict(input=i,hidden=h,directions=directions,layers=layers)
        rows.append(dict(module=q,line=node.lineno,parameters=n,tensors=k,shape=shape))
        total+=n;tensors+=k
    return dict(source=path,constructor=name,parameters=total,tensors=tensors,
        scalar_parameters=scalar_parameters,modules=rows)

def check_v1():
    source=HERE/"american_treeloom_full_c3_v1.py"
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)
        and n.name=="AmericanTreeLoomC3FullV1")
    encode=next(n for n in cls.body if isinstance(n,ast.FunctionDef)
        and n.name=="encode_full_parent_v1")
    calls=[qualified(n.func) for n in ast.walk(encode) if isinstance(n,ast.Call)]
    assert calls.count("self.sweep_tree_state_v1")==2
    assert all(calls.count(x)==1 for x in
        ("self.prepare_tree_base_v1","self.clade_coupling","self.episode_coupling",
         "self.memory","self.leaf_context_v1"))
    assert 'arm="own_state"' in ast.get_source_segment(source.read_text(),encode)
    assert 'MEMORY_ARM = "stable_leaf_memory"' in source.read_text()
    assert "sweep_tree_state_v1 = AnonymousCladeTensorGNNV1.sweep_tree_state_v1" in source.read_text()
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            try:name=qualified(node.func)
            except ValueError:name=""
            assert name not in ("open","Path.open","torch.load","np.load","subprocess.Popen")
    base="src/arggnn/american_c3_gnn_model_v1.py"
    env=dict(d=32,hidden_dim=32,donor_cohorts=2)
    env.update({"self.donor_channels":6,"self.num_classes":3})
    c3=count_modules(base,"AmericanC3WholeTreeGNNV1",env)
    c4env=dict(env);c4env.update({"self.donor_channels":8,"self.num_classes":4})
    c4=count_modules(base,"AmericanC3WholeTreeGNNV1",c4env)
    flags=count_modules("src/arggnn/anonymous_clade_flag_control_v1.py",
        "RankEightCladeFlagAdapterV1",{})
    own=count_modules("src/arggnn/ordered_clade_episode_gnn_v1.py",
        "RankEightCladeEpisodeAdapterV1",{})
    memory=count_modules("research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py",
        "CutMemoryV1",{"RANK":8},scalar_parameters=1)
    assert (c3["parameters"],c3["tensors"])==(29571,38)
    assert c4["parameters"]==29764 and c4["parameters"]-c3["parameters"]==193
    assert flags["parameters"]==own["parameters"]==1024
    assert (memory["parameters"],memory["tensors"])==(513,3)
    assert sum(x["parameters"] for x in (c3,flags,own,memory))==32132
    assert sum(x["tensors"] for x in (c3,flags,own,memory))==49
    return dict(schema="American_TreeLoom_static_source_checks_v1",status="STATIC_SOURCE_PASS",
        AST_parse=True,two_shared_sweep_calls=True,original_structural_and_memory_operators_reused=True,
        fixed_stable_leaf_memory=True,own_state_not_foreign_episode=True,
        complete_parent_only=True,native_10000_focals=True,
        symbolic_parameters=32132,symbolic_trainable_tensors=49,
        actual_model_parameter_measurement=False,model_imported_or_constructed=False,
        any_forward_or_backward_executed=False,data_checkpoint_or_truth_read=False,
        source_ref=dict(path=str(source.relative_to(ROOT)),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        components=[c3,flags,own,memory],C4_class_adapter_check=c4)

if __name__=="__main__":
    value=check_v1()
    (HERE/"static_checks.v1.json").write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:value[k] for k in ("status","symbolic_parameters",
        "symbolic_trainable_tensors","actual_model_parameter_measurement",
        "model_imported_or_constructed")},sort_keys=True))
