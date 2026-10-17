"""RAM-only fixtures. No guard, data, checkpoint, CUDA, forward or optimizer probes."""
from pathlib import Path
import copy, hashlib, importlib.util, io, json, math, sys, types

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_saved_MLM_bridge_RAM_fixture", HERE / "saved_mlm_bridge_v1.py")
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)
TESTS = []


def check(name, fn):
    fn()
    TESTS.append({"name": name, "status": "PASS"})


def rejects(fn):
    try:
        fn()
    except (ValueError, KeyError, TypeError, AttributeError):
        return
    raise AssertionError("Expected fail-closed rejection")


class Flag:
    def __init__(self, value): self.value = value
    def all(self): return self
    def item(self): return self.value


class Tensor:
    def __init__(self, values=(1.0, 2.0), shape=(2,), dtype="float32"):
        self.values, self.shape, self.dtype = tuple(values), tuple(shape), dtype
    def is_floating_point(self): return True
    def is_complex(self): return False
    def detach(self): return self
    def to(self, *, device):
        assert device == "cpu"
        return self


class Torch:
    def __init__(self, payload=None, events=None):
        self.payload, self.events = payload, events
    @staticmethod
    def is_tensor(x): return type(x) is Tensor
    @staticmethod
    def isfinite(x): return Flag(all(math.isfinite(v) for v in x.values))
    @staticmethod
    def equal(a, c):
        return a.values == c.values and a.shape == c.shape and a.dtype == c.dtype
    def load(self, path):
        self.events.append("original_checkpoint_load")
        assert path == str(b.ROOT / b.CHECKPOINT_REF["path"])
        return self.payload


class Model:
    def __init__(self, state): self.state = state
    def state_dict(self): return dict(self.state)


class Tokenizer:
    def __init__(self, reference):
        self.arg_tokenizer = types.SimpleNamespace(vocab=reference)
        self.model_max_length, self.pad_token_id, self.mask_token_id = 128, 0, 1
        self.cls_token_id, self.sep_token_id = 2, 3
    def get_vocab(self):
        return dict(self.arg_tokenizer.vocab["special"], **{}) | self.arg_tokenizer.vocab["node_id_tokens"]
    def __len__(self): return len(self.get_vocab())


def reference_vocab():
    return {"special": {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[SEP]": 3},
            "node_id_tokens": {i: i + 4 for i in range(107140)}}


def states():
    source = {"model.bert.parameter_" + str(i): Tensor() for i in range(74)}
    source.update({"model.head.weight": Tensor(), "model.head.bias": Tensor(),
                   "model.decoder.weight": Tensor(), "model.decoder.bias": Tensor()})
    target = {k[len("model."):]: Tensor(v.values, v.shape, v.dtype)
              for k, v in source.items() if not k.startswith("model.decoder.")}
    target.update({"classifier.weight": Tensor(), "classifier.bias": Tensor()})
    return source, target


fixture = json.loads((HERE / "fixture_inputs.v1.json").read_text())
blobs = {k: v.encode() for k, v in fixture["raw_metadata_text_by_role"].items()}
eligibility = b.eligibility_from_bytes_v1(blobs)


def exact_eligibility():
    r = eligibility.record()
    assert r["historical_outer_exit_code"] == 1
    assert r["saved_optimizer_updates"] == 227990
    assert not r["successful_wrapper_claimed"] and not r["launch_authority"]
    assert not r["full_fit_completion_claimed"]
    assert not r["finite_tensors_or_actual_encoder_transfer_verified"]
check("exact existing metadata permits artifact eligibility only; outer1 preserved", exact_eligibility)

for role in blobs:
    def bad_pin(role=role):
        bad = dict(blobs)
        bad[role] += b" "
        rejects(lambda: b.eligibility_from_bytes_v1(bad))
    check("reject changed pinned bytes: " + role, bad_pin)

fact_changes = [
    ("worker1", "wait", "returncode", 1),
    ("incomplete drain", "wait", "all_owned_processes_and_groups_drained", False),
    ("missing EOF", "wait", "pipe_EOF_observed", False),
    ("outer0 relabel", "outer", "actual_tool_terminal_exit_code", 0),
    ("missing consumption", "outer", "terminal_consumed", False),
    ("saved partial budget", "saved", "saved_batch", 22000),
    ("weights-only restore", "completed", "actual_Composer_restore", {"load_weights_only": True}),
]
for name, role, key, value in fact_changes:
    def bad_fact(role=role, key=key, value=value):
        docs = {k: json.loads(v) for k, v in blobs.items()}
        docs[role][key] = value
        rejects(lambda: b._facts(docs))
    check("reject fact predicate: " + name, bad_fact)


def transfer_pass():
    source, target = states()
    r = b._verify_transfer(source, Model(target), Torch())
    assert r["complete_encoder_key_count"] == 74
    assert r["all_source_and_destination_floating_tensors_finite"]
    assert r["missing_task_head_keys"] == ["classifier.bias", "classifier.weight"]
    assert r["unexpected_MLM_head_keys"] == ["decoder.bias", "decoder.weight"]
check("complete finite key/shape/dtype/value transfer with expected task heads", transfer_pass)

for failure in ("source_nan", "destination_inf", "missing_core", "wrong_shape", "wrong_dtype",
                "wrong_value", "unknown_namespace", "prefix_collision"):
    def transfer_fail(failure=failure):
        source, target = states()
        key = "bert.parameter_0"
        if failure == "source_nan": source["model." + key] = Tensor((float("nan"), 2.0))
        if failure == "destination_inf": target[key] = Tensor((float("inf"), 2.0))
        if failure == "missing_core": target.pop(key)
        if failure == "wrong_shape": target[key] = Tensor(shape=(1, 2))
        if failure == "wrong_dtype": target[key] = Tensor(dtype="bfloat16")
        if failure == "wrong_value": target[key] = Tensor((2.0, 1.0))
        if failure == "unknown_namespace": target["random.weight"] = Tensor()
        if failure == "prefix_collision":
            value = source.pop("model." + key)
            source[key] = value
        rejects(lambda: b._verify_transfer(source, Model(target), Torch()))
    check("reject transfer defect: " + failure, transfer_fail)

ref_vocab = reference_vocab()
tok = Tokenizer(copy.deepcopy(ref_vocab))
def tokenizer_pass():
    r = b._verify_tokenizer(ref_vocab, tok)
    assert r["same_exact_mapping"] and r["vocab_size"] == 107144
check("complete original tokenizer mapping including typed node IDs and special tokens", tokenizer_pass)

for failure in ("swapped_IDs", "special_ID", "short_vocab", "wrong_length"):
    def tokenizer_fail(failure=failure):
        vocab = copy.deepcopy(ref_vocab)
        t = Tokenizer(vocab)
        if failure == "swapped_IDs":
            vocab["node_id_tokens"][0], vocab["node_id_tokens"][1] = 5, 4
        if failure == "special_ID": vocab["special"]["[PAD]"] = 1
        if failure == "short_vocab": vocab["node_id_tokens"].pop(0)
        if failure == "wrong_length": t.model_max_length = 64
        rejects(lambda: b._verify_tokenizer(ref_vocab, t))
    check("reject tokenizer mismatch: " + failure, tokenizer_fail)


def fake_training(bad_value=False, double_build=False, skip_checkpoint=False):
    events = []
    source, target = states()
    if bad_value: target["bert.parameter_0"] = Tensor((8.0, 9.0))
    torch = Torch({"state": {"model": source}}, events)
    loader_env = {"torch": torch, "Model": Model, "target": target}
    exec("def from_composer(cls, pretrained_checkpoint, config=None):\n"
         "    value = torch.load(pretrained_checkpoint)\n"
         "    return Model(target)\n", loader_env)
    cls = type("FakeOriginalClassification", (), {
        "from_composer": classmethod(loader_env["from_composer"])})
    layers = types.SimpleNamespace(FlexBertForSequenceClassification=cls)
    flex_env = {"bert_layers_module": layers, "Tokenizer": Tokenizer,
                "reference": copy.deepcopy(ref_vocab), "types": types,
                "skip_checkpoint": skip_checkpoint}
    exec("def create_flex_bert_classification(pretrained_checkpoint=None, config=None):\n"
         "    if skip_checkpoint:\n"
         "        model = bert_layers_module.FlexBertForSequenceClassification(config)\n"
         "    else:\n"
         "        model = bert_layers_module.FlexBertForSequenceClassification.from_composer(\n"
         "            pretrained_checkpoint=pretrained_checkpoint, config=config)\n"
         "    return types.SimpleNamespace(model=model, tokenizer=Tokenizer(reference))\n", flex_env)
    flex = types.SimpleNamespace(bert_layers_module=layers,
        create_flex_bert_classification=flex_env["create_flex_bert_classification"])
    build_env = {"flex_bert_module": flex}
    exec("def build_contrastive_model(cfg):\n"
         "    return flex_bert_module.create_flex_bert_classification(\n"
         "        pretrained_checkpoint=cfg['pretrained_checkpoint'], config=None)\n", build_env)
    def original_optimizer(*args, **kwargs):
        events.append("original_optimizer_construction")
        return object()
    train_env = {"build_contrastive_model": build_env["build_contrastive_model"],
                 "build_optimizer": original_optimizer, "events": events, "double_build": double_build}
    exec("def train(cfg):\n"
         "    model = build_contrastive_model(cfg['model'])\n"
         "    if double_build:\n"
         "        build_contrastive_model(cfg['model'])\n"
         "    optimizer = build_optimizer({}, model)\n"
         "    events.append('original_complete_training_call_placeholder')\n", train_env)
    def reference_loader():
        events.append("reference_vocabulary_during_initialization")
        return ref_vocab
    def checkpoint_loader(original_torch, path, *args, **kwargs):
        assert original_torch is torch and not args and not kwargs
        return original_torch.load(path)
    checked, state = b._instrument_training(train_env["train"], checkpoint_loader, reference_loader)
    assert checked.__code__ is train_env["train"].__code__
    assert flex.create_flex_bert_classification is flex_env["create_flex_bert_classification"]
    return checked, state, events, cls, loader_env["from_composer"]


def inline_pass():
    global initialization_record
    checked, state, events, cls, loader = fake_training()
    checked({"model": {"pretrained_checkpoint": str(b.ROOT / b.CHECKPOINT_REF["path"])}})
    assert events == ["reference_vocabulary_during_initialization", "original_checkpoint_load",
                      "original_optimizer_construction", "original_complete_training_call_placeholder"]
    assert state["checkpoint_loads"] == state["builder_calls"] == 1
    assert state["source"] is None
    assert state["verified"]["verified_before_optimizer_construction"]
    assert cls.from_composer.__func__ is loader
    initialization_record = dict(state["verified"], eligibility=eligibility.record())
check("inline first original initialization; one load; verification before optimizer; no global edits", inline_pass)

for opts, name in (({"bad_value": True}, "bad transfer"),
                   ({"double_build": True}, "second initialization"),
                   ({"skip_checkpoint": True}, "random fallback")):
    def inline_fail(opts=opts):
        checked, state, events, cls, loader = fake_training(**opts)
        rejects(lambda: checked({"model": {"pretrained_checkpoint": str(b.ROOT / b.CHECKPOINT_REF["path"])}}))
        assert "original_optimizer_construction" not in events
        assert cls.from_composer.__func__ is loader
    check("inline fail before optimizer: " + name, inline_fail)


def document(name, value):
    raw = b._encoded(value)
    ref = {"path": b.HERE + "/RAM_ONLY_NOT_ACTUAL/" + name + ".json",
           "sha256": hashlib.sha256(raw).hexdigest()}
    return b.document_from_bytes_v1(ref, raw)


def bundle():
    init = document("initialization", initialization_record)
    completion = dict(status="completed_C3_native_child", stage="contrastive", parent_id=b.PARENT,
        preparation_ref=b.PREPARATION_REF, counts_ref=b.COUNTS_REF,
        author_source_manifest_ref=b.AUTHOR_MANIFEST_REF, initializer_checkpoint_ref=b.CHECKPOINT_REF,
        saved_MLM_initialization_ref=init.ref, completed_epochs=1, actual_optimizer_updates=2135,
        uniform_class_weight_fallback_preserved=True, training_config_plumbing_correction=False,
        query_ancestry_truth_opened=False,
        checkpoint_ref={"path": b.HERE + "/RAM_ONLY_NOT_ACTUAL/contrastive.pt", "sha256": "1" * 64})
    done = document("contrastive", completion)
    wait = document("contrastive_wait", dict(returncode=0, actual_direct_child_wait=True,
        process_group_ESRCH=True, child_completion_ref=done.ref, operational_failure=None))
    outer = document("outer", dict(schema="native_saved_MLM_downstream_actual_outer_terminal_v1",
        terminal_consumed=True, actual_tool_terminal_exit_code=0, tool_session_id=123,
        actual_wait_ref=wait.ref, child_completion_ref=done.ref))
    return dict(initialization=init, contrastive_completion=done,
                contrastive_wait=wait, consumed_outer=outer)


def separate_qualification_pass():
    result = b.qualify_for_native_embeddings_v1(eligibility, **bundle())
    assert result["historical_MLM_outer_exit_code"] == 1
    assert not result["original_full_fit_qualifier_modified"]
    assert not result["completed_legacy_full_fit_wrapper_claimed"]
    assert result["requires_all_four_original_single_split_embeddings"]
    assert not result["launch_authority"]
check("RAM-only future qualification requires full contrastive and preserves separate provenance", separate_qualification_pass)

for role, key, value, name in [
    ("consumed_outer", "actual_tool_terminal_exit_code", 1, "contrastive outer1"),
    ("consumed_outer", "terminal_consumed", False, "unconsumed outer"),
    ("contrastive_wait", "actual_direct_child_wait", False, "unknown direct wait"),
    ("contrastive_wait", "process_group_ESRCH", False, "incomplete drain"),
    ("contrastive_completion", "actual_optimizer_updates", 2134, "partial contrastive"),
    ("contrastive_completion", "checkpoint_ref", b.CHECKPOINT_REF, "MLM used for head_norm"),
    ("initialization", "all_source_and_destination_floating_tensors_finite", False, "unverified finite state"),
    ("initialization", "complete_encoder_keys_shapes_dtypes_and_values_equal", False, "incomplete encoder transfer"),
]:
    def qualification_fail(role=role, key=key, value=value):
        docs = bundle()
        changed = docs[role].value
        changed[key] = value
        docs[role] = document("mutated_" + role, changed)
        rejects(lambda: b.qualify_for_native_embeddings_v1(eligibility, **docs))
    check("reject downstream qualification: " + name, qualification_fail)


def no_runtime_or_guard_import():
    assert "torch" not in sys.modules and "torch_geometric" not in sys.modules
    assert not any(n.startswith("arggnn") for n in sys.modules)
check("source and RAM fixtures import no actual tensor/runtime/guard modules", no_runtime_or_guard_import)



def fd_fixture(mode="valid", extra_kwargs=False):
    original_os, original_path = b.os, b._path
    expected = b.CHECKPOINT_STAMP
    def stat_value(changed=False):
        return types.SimpleNamespace(st_dev=expected[0], st_ino=expected[1],
            st_size=expected[2], st_mtime_ns=expected[3],
            st_ctime_ns=expected[4] + (1 if changed else 0), st_nlink=1, st_mode=0o100600)
    class Stream(io.BytesIO):
        def fileno(self): return 42
    stream = Stream(b"RAM_ONLY_NOT_A_CHECKPOINT")
    calls = []
    fstats = [0]
    class FakePath:
        def lstat(self): return stat_value(mode == "changed_after")
    def fake_open(path, flags):
        calls.append("open_readonly_no_follow")
        return 42
    def fake_fdopen(fd, mode_):
        assert fd == 42 and mode_ == "rb"
        return stream
    def fake_fstat(fd):
        fstats[0] += 1
        return stat_value(mode == "wrong_before" or
                          (mode == "changed_after" and fstats[0] > 1))
    class FakeTorch:
        @staticmethod
        def load(value):
            assert value is stream
            calls.append("one_original_load")
            return {"state": {"model": {"RAM_ONLY": "no actual tensors"}}}
    b.os = types.SimpleNamespace(open=fake_open, fdopen=fake_fdopen, fstat=fake_fstat,
        O_RDONLY=original_os.O_RDONLY, O_CLOEXEC=original_os.O_CLOEXEC,
        O_NOFOLLOW=original_os.O_NOFOLLOW)
    b._path = lambda name: FakePath()
    try:
        action = lambda: b._load_once_from_exact_FD(FakeTorch(),
            str(b.ROOT / b.CHECKPOINT_REF["path"]), **({"map_location": "cpu"} if extra_kwargs else {}))
        if mode == "valid" and not extra_kwargs:
            action()
            assert calls == ["open_readonly_no_follow", "one_original_load"]
        else:
            rejects(action)
            assert calls.count("one_original_load") == (1 if mode == "changed_after" else 0)
    finally:
        b.os, b._path = original_os, original_path

check("RAM stable FD: exact identity, one original load, no hash or extra load", fd_fixture)
check("RAM stable FD: reject wrong identity before original load",
      lambda: fd_fixture(mode="wrong_before"))
check("RAM stable FD: reject changed identity after original load",
      lambda: fd_fixture(mode="changed_after"))
check("RAM stable FD: reject added loader parameters", lambda: fd_fixture(extra_kwargs=True))


def immutable_documents():
    doc = document("document_auth", {"x": 1})
    corrupt = b.DocumentV1(doc.ref_bytes, b'{"x":2}\n', b._SEAL)
    rejects(lambda: corrupt.value)
    unsealed = b.DocumentV1(doc.ref_bytes, doc.json_bytes, object())
    rejects(lambda: unsealed.value)
check("bound future documents reject altered bytes and unverified constructor seal", immutable_documents)


def unauthenticated_eligibility():
    fake = b.EligibilityV1(b._encoded(eligibility.record()), object())
    rejects(lambda: fake.record())
    rejects(lambda: b.run_original_contrastive_train_v1({}, fake, author_train=lambda x: None))
check("unverified eligibility cannot enter original numerical wrapper", unauthenticated_eligibility)


def code_authentication():
    original = b._read_small
    raw = b"def train(cfg):\n    return cfg\n"
    ns = {}
    exec(compile(raw, "<RAM-source>", "exec"), ns)
    source_name = ".runtime/author_sources/argformer_f7545be7bca0/sequence_contrastive.py"
    b._read_small = lambda *args, **kwargs: raw
    try:
        b._authenticate_function(ns["train"], source_name, ("train",))
        different = {}
        exec("def train(cfg):\n    return None\n", different)
        rejects(lambda: b._authenticate_function(different["train"], source_name, ("train",)))
    finally:
        b._read_small = original
check("RAM original code-object authentication rejects altered author implementation", code_authentication)


def run():
    return dict(schema="native_saved_MLM_acceptance_bridge_RAM_tests_v1",
        status="PASS_RAM_ONLY_SOURCE_REVIEW_REQUIRED", tests=TESTS, count=len(TESTS),
        actual_new_contrastive_or_embedding_execution=False,
        real_checkpoint_opened_or_hashed=False, scientific_data_or_truth_read=False,
        guard_or_GPU_or_admission_calls=False, optimizer_or_forward_probes=False,
        synthetic_tensors_only=True, source_activation_or_authority=False)


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
