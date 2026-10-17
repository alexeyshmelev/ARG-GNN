"""INACTIVE exact saved-MLM eligibility and inline original-initialization checks.

Import and the default CLI use stdlib only. This module supplies no admission,
lease, guard, launch, retry, close or credit authority. The numerical wrapper may
only replace the existing author train call inside a separately admitted worker.
It runs the original training code, not a standalone validator or optimizer probe.
"""
from pathlib import Path
from dataclasses import dataclass
import hashlib, io, json, os, pickle, stat, types

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = "research/reviews/native_VAL000_saved_MLM_acceptance_bridge_source_20261017_v1"
PARENT = "american_c3_val_000"
EVIDENCE_PINS = {'wait': {'path': 'research/reviews/native_VAL000_fresh_continuation_ROOT_v12_root01_oct16/worker_aux/actual_wait.json', 'sha256': 'fe35274be6a945a134535634f92647d36e6f2600824c05d39233d617c226565e'}, 'minimal': {'path': 'research/reviews/native_VAL000_fresh_continuation_ROOT_v12_root01_oct16/worker_aux/actual_wait_minimal.json', 'sha256': '1f3f56a26bcda5a3efac5d41c10aeb6769848104624de17fb8d7f5f1c9430a28'}, 'outer': {'path': 'research/reviews/research_execution_ROOT_20261017_v19/native70445_actual_outer_terminal_consumed.root_v1.json', 'sha256': '1eaa378c5650a7c72423f9fba4b4b2603957c2a4e30f5b21b3db47bc24fa6371'}, 'completed': {'path': 'experiments/results/native_VAL000_verified22000_continuation_v12_root01_oct16/completed.json', 'sha256': '99f575d18d4c0728339c1fb7778f784fa6ce81e0e0b8492fc897a8251e782bc4'}, 'saved': {'path': 'research/reviews/research_execution_ROOT_20261017_v16/saved_native_optimizer_progress.root_v1.json', 'sha256': 'd7a91e07caef51b5346db64aa95fd09ff1a9e7c79d8e2a6eb528c623e61f0eae'}, 'milestone': {'path': 'research/reviews/research_execution_ROOT_20261017_v17/verified_training_milestone.root_v1.json', 'sha256': 'bff26f93084d91c2d51b5b3ef36fbede74c1f6c028727125854b1c4d250d91d0'}, 'corpus': {'path': 'experiments/results/native_VAL000_verified22000_continuation_v12_root01_oct16/corpus_authentication.json', 'sha256': '8e8dbbdc88754bdfcf8be65fed6dcfc7d740dd0ee813ee930d9fb6f391831ced'}, 'counts': {'path': 'experiments/results/American_C3_native_val000_full30mb_counts980_namespace_v1/completed.json', 'sha256': '502c111b443c46eea789dc03ee3ff8b6454546ce6e86e3f9fcddba25e82489a7'}}
AUTHOR_PINS = {'.runtime/author_sources/argformer_f7545be7bca0/sequence_contrastive.py': '7df3e0571aeb032b8ec745d49feb5d0335572a9a9594ee643603e961b4b98339', '.runtime/author_sources/argformer_f7545be7bca0/src/flex_bert.py': 'bf8f041c30a1af410b1b41f8b047b1c84fb3ad8b991d45a7cc7d38d66a4f227a', '.runtime/author_sources/argformer_f7545be7bca0/src/bert_layers/model.py': '6264b60199f2757e18e5c3ae760954f98dffe688913a151bae6713ce53c73e72', '.runtime/author_sources/argformer_f7545be7bca0/src/arg_data.py': '2e7df2c955ece89ced3f05c609a4450f997250df5a7cb06ff4a2ca957e49ffdf', '.runtime/author_sources/argformer_f7545be7bca0/yamls/contrastive.yaml': '3b4f87e149a2b56d8a3efa2e35bd52ec94b61d2d3c768f15df67e82376ce4847'}
CHECKPOINT_REF = {
    "path": "experiments/results/native_VAL000_verified22000_continuation_v12_root01_oct16/checkpoints/ep5-ba227990-rank0.pt",
    "sha256": "762d005aa403a787520f389f285a24b59a249c06f8f1145bc00db2ad060c499a",
}
CHECKPOINT_STAMP = (67, 9257095304052243217, 2018101574,
                    1792213794077481679, 1792213794077481679, 1)
VOCAB_REF = {
    "path": "experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/vocab.pkl",
    "sha256": "75ce862ef02c6dbeb755f65b3f7878d6487318f386233a1442adbbbc8a0e717f",
}
AUTHOR_MANIFEST_REF = {
    "path": "research/reviews/American_C3_native_consumer_source_v1/author_source_manifest.json",
    "sha256": "bed570db438f2c4981ebf606460dc4ff72a97b148ba371e3873ab0141376736a",
}
PREPARATION_REF = {
    "path": "experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/completed.json",
    "sha256": "40a3b588d0aa2b474c3f4aab492e34feb0a753a0f8cb73c20d9c0059d8a8c16b",
}
COUNTS_REF = {
    "path": "experiments/results/American_C3_native_val000_full30mb_counts980_namespace_v1/completed.json",
    "sha256": "502c111b443c46eea789dc03ee3ff8b6454546ce6e86e3f9fcddba25e82489a7",
}
_SEAL = object()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _ref(path, raw):
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}


def _stamp(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink)


def _path(name):
    p = ROOT / name
    require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts
            and p.resolve() == p and p.is_relative_to(ROOT), "Exact canonical project name")
    return p


def _read_small(name, expected_sha, cap=1048576):
    """Metadata/source/vocabulary only. Never used for checkpoint payloads."""
    p = _path(name)
    fd = os.open(p, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        before = os.fstat(f.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and before.st_size <= cap, "Bounded single-link source or metadata")
        raw = f.read(cap + 1)
        after = os.fstat(f.fileno())
        require(_stamp(before) == _stamp(after) == _stamp(p.lstat()),
                "Stable read-only source/metadata identity")
    require(len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == expected_sha,
            "Exact pinned source/metadata bytes")
    return raw


@dataclass(frozen=True)
class EligibilityV1:
    _bytes: bytes
    _seal: object

    def record(self):
        require(self._seal is _SEAL, "Verified artifact eligibility only")
        return json.loads(self._bytes)


def _facts(documents):
    w, m, outer, done, saved, milestone, corpus, counts = [documents[k] for k in
        ("wait", "minimal", "outer", "completed", "saved", "milestone", "corpus", "counts")]
    require(w["returncode"] == w["actual_OS_wait_status"] == 0
            and w["actual_direct_worker_wait"] is True
            and w["all_owned_processes_and_groups_drained"] is True
            and w["complete_owned_roster_known"] is True and w["owned_identity_absence"] is True
            and w["pipe_EOF_observed"] is True and w["cleanup_errors"] == [],
            "Genuine worker0, complete owned drain and EOF")
    own = w["ownership"]
    require(own["known_identity_count"] == len(own["owned_identities"]) == 50
            and own["complete_owned_roster_known"] is True
            and own["owned_roster_truncated"] is False and w["adopted_waits_truncated"] is False
            and w["adopted_wait_count"] == len(w["adopted_waits"]) == 1
            and all(v["returncode"] == 0 for v in w["adopted_waits"]),
            "Exact complete 50-identity current attempt")
    require(w["operational_failure"] == m["operational_failure"]
            == "ValueError: Actual native worker current identity"
            and w["no_future_stage_authorized"] is True
            and w["storage_close_or_credit_authorized"] is False,
            "Exact preserved postfit identity failure, no authority")
    for k in ("actual_OS_wait_status", "actual_direct_worker_wait", "returncode",
              "all_owned_processes_and_groups_drained", "complete_owned_roster_known",
              "pipe_EOF_observed", "adopted_wait_count"):
        require(m[k] == w[k], "Minimal/full wait join")
    require(m["known_identity_count"] == 50 and m["no_future_stage_authorized"] is True,
            "Minimal receipt is facts only")
    require(outer["tool_session_id"] == 70445 and outer["actual_tool_terminal_exit_code"] == 1
            and outer["terminal_consumed"] is True and outer["outer_failure_preserved"] is True
            and outer["actual_direct_worker_returncode_in_tool_output"] == 0
            and outer["all_owned_processes_and_groups_drained_in_tool_output"] is True
            and outer["successful_outer_terminal_claimed"] is False
            and outer["automatic_retry_authorized"] is False,
            "Genuine consumed outer1 remains1")
    require(done["status"] == "completed_C3_native_MLM_child" and done["stage"] == "mlm"
            and done["parent_id"] == PARENT and done["completed_epochs"] == 5
            and done["actual_optimizer_updates"] == 227990
            and done["observed_counters_verified"] is True
            and done["author_source_manifest_ref"] == AUTHOR_MANIFEST_REF
            and done["preparation_ref"] == PREPARATION_REF
            and done["checkpoint_ref"] == CHECKPOINT_REF
            and done["initializer_checkpoint_ref"] is None
            and done["query_ancestry_truth_opened"] is False
            and done["actual_original_full_state_resume"] is True
            and done["actual_Composer_restore"]["load_weights_only"] is False
            and done["actual_Composer_restore"]["actual_restored_optimizer_counters"]["unique_step_values"] == [22000],
            "Exact full saved MLM provenance and original full-state restore")
    for k in ("checkpoint_optimizer_counters", "trainer_optimizer_counters"):
        c = done[k]
        require(c["parameter_state_count"] == 77 and c["unique_step_values"] == [227990]
                and c["optimizer_names"] == ["DecoupledAdamW"], "Full live/saved optimizer counters")
    require(done["trainer_timestamp"] == {"epoch": 5, "batch": 227990, "batch_in_epoch": 0}
            and saved["saved_epoch"] == 5 and saved["saved_batch"] == 227990
            and saved["optimizer_parameter_state_count"] == 77
            and saved["optimizer_unique_step_values"] == [227990]
            and tuple(saved["stable_open_readonly_FD_content_identity"])
            == CHECKPOINT_STAMP[:4]
            and saved["metadata_before"] == saved["metadata_after"]
            == {"nlink": 1, "ctime_ns": CHECKPOINT_STAMP[4]}
            and milestone["native_saved_optimizer_updates"] == 227990
            and milestone["native_completed_epochs"] == 5,
            "Independent previous stable saved-FD observation")
    require(corpus["status"] == "authenticated_full_MLM_inputs"
            and corpus["inputs"]["vocab"]["ref"] == VOCAB_REF
            and corpus["inputs"]["vocab"]["sha256_verified"] is True
            and corpus["all_consumed_MLM_raw_files_authenticated"] is True
            and counts["MLM_optimizer_updates"] == 227990
            and counts["contrastive_optimizer_updates"] == 2135
            and counts["vocabulary_nodes"] + 4 == 107144
            and counts["parent_id"] == PARENT and counts["query_focal_count"] == 10000,
            "Original corpus, vocabulary and full budget")


def eligibility_from_bytes_v1(blobs):
    """Authenticate small immutable evidence in RAM; grants artifact eligibility only."""
    require(set(blobs) == set(EVIDENCE_PINS), "Exact evidence set")
    docs = {}
    refs = {}
    for role, pin in EVIDENCE_PINS.items():
        raw = blobs[role]
        require(type(raw) is bytes and len(raw) <= 1048576
                and _ref(pin["path"], raw) == pin, "Exact authenticated " + role)
        docs[role] = json.loads(raw)
        refs[role] = dict(pin)
    _facts(docs)
    require(docs["wait"]["completed_ref"] == refs["completed"]
            and docs["wait"]["actual_wait_minimal_ref"] == refs["minimal"]
            and docs["completed"]["corpus_authentication_ref"] == refs["corpus"],
            "Exact metadata joins")
    record = dict(schema="native_VAL000_saved_MLM_artifact_eligibility_v1",
        artifact_status="eligible_saved_MLM_initializer_only", parent_id=PARENT,
        checkpoint_ref=dict(CHECKPOINT_REF), evidence_refs=refs,
        completed_epochs=5, saved_optimizer_updates=227990, optimizer_state_count=77,
        historical_outer_tool_session_id=70445, historical_outer_exit_code=1,
        successful_wrapper_claimed=False, full_fit_completion_claimed=False,
        finite_tensors_or_actual_encoder_transfer_verified=False,
        launch_authority=False, retry_authority=False, close_or_credit_authority=False,
        requires_original_inline_initialization_checks=True)
    return EligibilityV1(_encoded(record), _SEAL)


def read_eligibility_v1():
    """Optional metadata-only reader; no data, checkpoint, guards or authority checks."""
    return eligibility_from_bytes_v1({
        role: _read_small(pin["path"], pin["sha256"]) for role, pin in EVIDENCE_PINS.items()})


class _PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Only primitive pinned vocabulary data")


def _runtime_reference_vocabulary():
    # Called only inside the first original downstream model initialization.
    # The original 2 GB checkpoint is never rehashed by this bridge.
    raw = _read_small(VOCAB_REF["path"], VOCAB_REF["sha256"], cap=2 * 1024 * 1024)
    value = _PrimitiveUnpickler(io.BytesIO(raw)).load()
    require(type(value) is dict and set(value) == {"special", "node_id_tokens"},
            "Original primitive observed vocabulary")
    return value


def _verify_tokenizer(reference, tokenizer):
    require(set(reference) == {"special", "node_id_tokens"}
            and reference["special"] == {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[SEP]": 3}
            and tokenizer.arg_tokenizer.vocab == reference,
            "Actual original tokenizer uses exact pinned MLM vocabulary")
    flat = dict(reference["special"])
    require(not set(flat).intersection(reference["node_id_tokens"]), "No vocabulary key collision")
    flat.update(reference["node_id_tokens"])
    require(len(flat) == 107144 and all(type(k) in (int, str) and type(v) is int for k, v in flat.items())
            and sorted(flat.values()) == list(range(107144))
            and tokenizer.get_vocab() == flat and len(tokenizer) == 107144
            and tokenizer.model_max_length == 128
            and tokenizer.pad_token_id == 0 and tokenizer.mask_token_id == 1
            and tokenizer.cls_token_id == 2 and tokenizer.sep_token_id == 3,
            "Complete same token-to-ID map, special IDs, length and vocabulary")
    typed = sorted((("i" if type(k) is int else "s", str(k), v) for k, v in flat.items()))
    return dict(vocab_size=107144, max_seq_len=128,
                typed_token_ID_mapping_sha256=hashlib.sha256(_encoded(typed)).hexdigest(),
                reference_vocab_ref=dict(VOCAB_REF), same_exact_mapping=True)


def _verify_transfer(source, model, torch):
    require(type(source) is dict and len(source) == 78, "Exact saved 78-key model state")
    mapped = {}
    for key, tensor in source.items():
        require(type(key) is str and key.startswith("model."), "Exact original Composer model prefix")
        name = key[len("model."):]
        require(name and name not in mapped, "No checkpoint prefix collision")
        mapped[name] = tensor
    target = model.state_dict()
    require(type(target) is dict or isinstance(target, dict), "Actual original model state mapping")
    core = {k for k in target if k.startswith("bert.")}
    source_core = {k for k in mapped if k.startswith("bert.")}
    require(core and core == source_core, "Every source and destination encoder key matches")
    require(all(k.startswith(("bert.", "head.", "decoder.")) for k in mapped)
            and all(k.startswith(("bert.", "head.", "classifier.")) for k in target),
            "Only original MLM/encoder/task-head namespaces")
    missing = sorted(set(target) - set(mapped))
    unexpected = sorted(set(mapped) - set(target))
    require(all(k.startswith(("head.", "classifier.")) for k in missing)
            and all(k.startswith(("head.", "decoder.")) for k in unexpected),
            "Only expected task-head differences")
    for name, tensor in list(mapped.items()) + list(target.items()):
        require(torch.is_tensor(tensor), "Actual model tensor " + name)
        if tensor.is_floating_point() or tensor.is_complex():
            require(bool(torch.isfinite(tensor).all().item()), "Nonfinite model tensor " + name)
    matched = sorted(set(mapped).intersection(target))
    for key in matched:
        a, b = mapped[key], target[key]
        require(tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype,
                "Exact loaded key shape/dtype " + key)
        # Per-key comparison only; no forward, optimizer, RNG or model mutation.
        require(bool(torch.equal(a.detach().to(device="cpu"), b.detach().to(device="cpu"))),
                "Exact original loaded weight/buffer value " + key)
    return dict(source_model_state_keys=78, complete_encoder_key_count=len(core),
                all_source_and_destination_floating_tensors_finite=True,
                complete_encoder_keys_shapes_dtypes_and_values_equal=True,
                matched_key_names=matched, missing_task_head_keys=missing,
                unexpected_MLM_head_keys=unexpected,
                no_forward_or_optimizer_probe=True)


class _AttrProxy:
    def __init__(self, original, replacements):
        self._original, self._replacements = original, replacements

    def __getattr__(self, name):
        return self._replacements[name] if name in self._replacements else getattr(self._original, name)


def _clone(function, replacements):
    result = types.FunctionType(function.__code__, dict(function.__globals__, **replacements),
                                function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def _code_shape(code):
    def constant(v):
        if isinstance(v, types.CodeType):
            return ("code", _code_shape(v))
        if isinstance(v, tuple):
            return ("tuple", tuple(constant(x) for x in v))
        if isinstance(v, frozenset):
            return ("frozenset", sorted(repr(x) for x in v))
        return (type(v).__name__, repr(v))
    return (code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_nlocals, code.co_stacksize, code.co_flags, code.co_code,
            tuple(constant(x) for x in code.co_consts), code.co_names,
            code.co_varnames, code.co_freevars, code.co_cellvars)


def _authenticate_function(function, source_name, code_path):
    raw = _read_small(source_name, AUTHOR_PINS[source_name], cap=2 * 1024 * 1024)
    code = compile(raw, str(ROOT / source_name), "exec", dont_inherit=True, optimize=0)
    for name in code_path:
        choices = [c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == name]
        require(len(choices) == 1, "Unique original author function")
        code = choices[0]
    require(_code_shape(function.__code__) == _code_shape(code),
            "Unchanged original author code object " + ".".join(code_path))


def _load_once_from_exact_FD(torch, path, *args, **kwargs):
    require(str(path) == str(ROOT / CHECKPOINT_REF["path"]) and not args and not kwargs,
            "Only original one-argument checkpoint initialization load")
    p = _path(CHECKPOINT_REF["path"])
    fd = os.open(p, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and _stamp(before) == CHECKPOINT_STAMP,
                "Exact previously qualified immutable saved checkpoint identity")
        value = torch.load(stream)  # Original load only, no extra load/hash/probe.
        require(_stamp(os.fstat(stream.fileno())) == _stamp(p.lstat()) == CHECKPOINT_STAMP,
                "Stable original saved checkpoint through actual loader")
    return value


def _instrument_training(train, checkpoint_loader, vocabulary_loader):
    """Private composition shared with RAM fixtures. Public entry authenticates it."""
    build = train.__globals__["build_contrastive_model"]
    flex = build.__globals__["flex_bert_module"]
    layers = flex.bert_layers_module
    cls = layers.FlexBertForSequenceClassification
    loader = cls.from_composer.__func__
    torch = loader.__globals__["torch"]
    state = {"builder_calls": 0, "checkpoint_loads": 0, "verified": None, "source": None}
    def load(path, *args, **kwargs):
        require(state["builder_calls"] == 1 and state["checkpoint_loads"] == 0,
                "One actual original initialization checkpoint load")
        state["checkpoint_loads"] += 1
        value = checkpoint_loader(torch, path, *args, **kwargs)
        require(type(value) is dict and type(value.get("state")) is dict
                and type(value["state"].get("model")) is dict, "Original Composer saved model state")
        state["source"] = dict(value["state"]["model"])  # References only, no tensor copy.
        return value
    checked_loader = _clone(loader, {"torch": _AttrProxy(torch, {"load": load})})
    class ClassProxy:
        @staticmethod
        def from_composer(*args, **kwargs):
            return checked_loader(cls, *args, **kwargs)
        def __new__(cls, *args, **kwargs):
            raise ValueError("No random encoder initialization fallback")
    flex_builder = _clone(flex.create_flex_bert_classification,
        {"bert_layers_module": _AttrProxy(layers, {"FlexBertForSequenceClassification": ClassProxy})})
    def checked_flex(*args, **kwargs):
        require(state["builder_calls"] == 0, "One original downstream model initialization")
        state["builder_calls"] = 1
        reference = vocabulary_loader()
        hf = flex_builder(*args, **kwargs)
        require(state["checkpoint_loads"] == 1 and state["source"] is not None,
                "Original model builder actually loaded accepted checkpoint")
        transfer = _verify_transfer(state["source"], hf.model, torch)
        tokenizer = _verify_tokenizer(reference, hf.tokenizer)
        state["source"] = None  # Release the extra tensor references before data/optimizer work.
        state["verified"] = dict(transfer, tokenizer=tokenizer,
            schema="native_VAL000_original_inline_saved_MLM_initialization_v1",
            checkpoint_ref=dict(CHECKPOINT_REF), parent_id=PARENT,
            original_function_code_objects_unchanged=True,
            installed_author_or_torch_modules_mutated=False,
            verified_before_optimizer_construction=True,
            independent_optimizer_or_forward_probe=False)
        return hf
    checked_build = _clone(build,
        {"flex_bert_module": _AttrProxy(flex, {"create_flex_bert_classification": checked_flex})})
    optimizer = train.__globals__["build_optimizer"]
    def checked_optimizer(*args, **kwargs):
        require(state["verified"] is not None, "Verified initialization before original optimizer construction")
        return optimizer(*args, **kwargs)
    checked_train = _clone(train, {"build_contrastive_model": checked_build,
                                   "build_optimizer": checked_optimizer})
    return checked_train, state


def run_original_contrastive_train_v1(cfg, eligibility, *, author_train):
    """Call only in the original admitted contrastive worker's author-call boundary.

    This executes the original complete training call. It has no standalone
    numerical validation mode and confers no permission to call it.
    """
    require(type(eligibility) is EligibilityV1 and eligibility._seal is _SEAL,
            "Exact authenticated saved-MLM eligibility object")
    eligibility.record()
    base = ".runtime/author_sources/argformer_f7545be7bca0/"
    _authenticate_function(author_train, base + "sequence_contrastive.py", ("train",))
    build = author_train.__globals__["build_contrastive_model"]
    _authenticate_function(build, base + "sequence_contrastive.py", ("build_contrastive_model",))
    flex = build.__globals__["flex_bert_module"]
    _authenticate_function(flex.create_flex_bert_classification, base + "src/flex_bert.py",
                           ("create_flex_bert_classification",))
    cls = flex.bert_layers_module.FlexBertForSequenceClassification
    _authenticate_function(cls.from_composer.__func__, base + "src/bert_layers/model.py",
                           ("FlexBertForSequenceClassification", "from_composer"))
    for name in (base + "src/arg_data.py", base + "yamls/contrastive.yaml"):
        _read_small(name, AUTHOR_PINS[name])
    om = author_train.__globals__["om"]
    actual = om.to_container(cfg, resolve=True)
    expected = om.load(str(ROOT / base / "yamls/contrastive.yaml"))
    expected.data_local = str(ROOT / PREPARATION_REF["path"]).rsplit("/", 1)[0] + "/contrastive"
    expected.run_name = "C3_native_contrastive"
    expected.save_folder = actual["save_folder"]  # Existing admitted worker owns this new output.
    expected.load_path = None
    expected.autoresume = False
    expected.loggers = {}
    expected.device = "gpu"
    expected.model.pretrained_model_name = str(ROOT / "research/reviews/C3_bert_base_configuration_root_v1")
    expected.model.pretrained_checkpoint = str(ROOT / CHECKPOINT_REF["path"])
    require(actual == om.to_container(expected, resolve=True),
            "Only original operational I/O substitutions; all author science unchanged")
    checked_train, state = _instrument_training(author_train, _load_once_from_exact_FD,
                                               _runtime_reference_vocabulary)
    checked_train(cfg)  # Same original complete epoch, data, optimizer and objective.
    require(state["builder_calls"] == state["checkpoint_loads"] == 1 and state["verified"] is not None,
            "Actual inline initialization occurred once")
    return dict(state["verified"], eligibility=eligibility.record(),
                training_completion_or_budget_not_asserted_by_initialization_receipt=True,
                no_launch_close_credit_or_successful_historical_wrapper_claim=True)


@dataclass(frozen=True)
class DocumentV1:
    ref_bytes: bytes
    json_bytes: bytes
    _seal: object

    @property
    def ref(self):
        require(self._seal is _SEAL, "Authenticated metadata document")
        return json.loads(self.ref_bytes)

    @property
    def value(self):
        require(self._seal is _SEAL and _ref(self.ref["path"], self.json_bytes) == self.ref,
                "Authenticated immutable raw document bytes")
        return json.loads(self.json_bytes)


def document_from_bytes_v1(ref, raw):
    """Future new-stage metadata only, with exact caller-bound content identity."""
    require(type(ref) is dict and set(ref) == {"path", "sha256"} and type(raw) is bytes
            and len(raw) <= 1048576 and _ref(ref["path"], raw) == ref, "Exact bounded new-stage document")
    _path(ref["path"])
    value = json.loads(raw)
    require(type(value) is dict, "New-stage JSON object")
    return DocumentV1(_encoded(ref), raw, _SEAL)


def qualify_for_native_embeddings_v1(eligibility, *, initialization, contrastive_completion,
                                     contrastive_wait, consumed_outer):
    """Separate qualification, never a synthetic legacy completed full-fit receipt.

    Caller must authenticate each DocumentV1 from its newly admitted stage. No
    old source is modified and original fit_completion_v1 remains unchanged.
    """
    require(type(eligibility) is EligibilityV1 and eligibility._seal is _SEAL,
            "Original exact saved artifact eligibility")
    require(all(type(x) is DocumentV1 for x in
        (initialization, contrastive_completion, contrastive_wait, consumed_outer)),
        "Bound metadata documents")
    init, done, wait, outer = [x.value for x in
        (initialization, contrastive_completion, contrastive_wait, consumed_outer)]
    require(init["schema"] == "native_VAL000_original_inline_saved_MLM_initialization_v1"
            and init["eligibility"] == eligibility.record() and init["checkpoint_ref"] == CHECKPOINT_REF
            and init["verified_before_optimizer_construction"] is True
            and init["all_source_and_destination_floating_tensors_finite"] is True
            and init["complete_encoder_keys_shapes_dtypes_and_values_equal"] is True
            and init["tokenizer"]["same_exact_mapping"] is True
            and init["tokenizer"]["vocab_size"] == 107144
            and init["tokenizer"]["reference_vocab_ref"] == VOCAB_REF,
            "Actual original inline initialization evidence")
    require(done["status"] == "completed_C3_native_child" and done["stage"] == "contrastive"
            and done["parent_id"] == PARENT and done["preparation_ref"] == PREPARATION_REF
            and done["counts_ref"] == COUNTS_REF and done["author_source_manifest_ref"] == AUTHOR_MANIFEST_REF
            and done["initializer_checkpoint_ref"] == CHECKPOINT_REF
            and done["saved_MLM_initialization_ref"] == initialization.ref
            and done["completed_epochs"] == 1 and done["actual_optimizer_updates"] == 2135
            and done["uniform_class_weight_fallback_preserved"] is True
            and done["training_config_plumbing_correction"] is False
            and done["query_ancestry_truth_opened"] is False,
            "Genuine full original same-parent contrastive completion")
    require(wait["returncode"] == 0 and wait["actual_direct_child_wait"] is True
            and wait["process_group_ESRCH"] is True
            and wait["child_completion_ref"] == contrastive_completion.ref
            and wait.get("operational_failure") is None,
            "Original genuine new contrastive worker wait0/drain")
    require(outer["schema"] == "native_saved_MLM_downstream_actual_outer_terminal_v1"
            and outer["terminal_consumed"] is True
            and outer["actual_tool_terminal_exit_code"] == 0
            and type(outer["tool_session_id"]) is int
            and outer["actual_wait_ref"] == contrastive_wait.ref
            and outer["child_completion_ref"] == contrastive_completion.ref,
            "Genuine consumed new contrastive outer0 bound to its real evidence")
    checkpoint = done["checkpoint_ref"]
    require(type(checkpoint) is dict and set(checkpoint) == {"path", "sha256"}
            and checkpoint != CHECKPOINT_REF and type(checkpoint["sha256"]) is str
            and len(checkpoint["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in checkpoint["sha256"]),
            "Actual separately completed contrastive checkpoint, never MLM for head_norm")
    _path(checkpoint["path"])
    return dict(schema="native_VAL000_saved_MLM_then_original_contrastive_qualification_v1",
        status="qualified_separate_saved_MLM_plus_completed_contrastive",
        parent_id=PARENT, preparation_ref=dict(PREPARATION_REF), counts_ref=dict(COUNTS_REF),
        author_source_manifest_ref=dict(AUTHOR_MANIFEST_REF),
        saved_MLM_eligibility=eligibility.record(), initialization_ref=initialization.ref,
        phases={"contrastive": dict(checkpoint_ref=checkpoint,
            child_completion_ref=contrastive_completion.ref, actual_wait_ref=contrastive_wait.ref,
            consumed_outer_ref=consumed_outer.ref)},
        historical_MLM_outer_exit_code=1, original_full_fit_qualifier_modified=False,
        completed_legacy_full_fit_wrapper_claimed=False,
        requires_all_four_original_single_split_embeddings=True,
        original_embedding_concat_retrieval_and_scoring_unchanged=True,
        query_ancestry_truth_opened=False, launch_authority=False, close_or_credit_authority=False)


if __name__ == "__main__":
    print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE", artifact_acceptance_activated=False,
                          numerical_execution=False, launch_authority=False), sort_keys=True))
