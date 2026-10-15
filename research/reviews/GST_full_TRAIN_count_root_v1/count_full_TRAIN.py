"""Count complete TRAIN using the unchanged native parser. No model or TEST."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, sys, time, traceback
ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = ROOT / "research/reviews/GST_full_TRAIN_count_root_v1"
AUTHOR = ROOT / ".runtime/gst_uspto_native_v1/source"
TRAIN = ROOT / "data/gst_uspto_native_payload_v2_r001/data/train.txt"
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert sys.version_info[:3] == (3, 14, 3)
    assert sha(AUTHOR / "scripts/eval_rxn_center.py") == "57fd803410d57c9b05d0719cdb5bd318d9d90b2cdebbc86a1260c37a3c3354b3"
    assert sha(AUTHOR / "src/graph_set_transformer/models/gst.py") == "bc08c15d55247b32bda98e4b03fcaee988ceb8f654423240a33ca632ce11ef72"
    sys.path[:0] = [str(AUTHOR / "scripts"), str(AUTHOR / "src")]
    import eval_rxn_center as native
    assert native.torch.__version__ == "2.10.0+cpu"
    assert Path(native.__file__).resolve() == AUTHOR / "scripts/eval_rxn_center.py"
    result = dict(schema="GST_full_TRAIN_native_parser_count_root_v1", status="running",
        started_utc=datetime.now(timezone.utc).isoformat(), author_commit="6469d56aa131463325058b897b1e435c6d2edcd2",
        parser_ref=dict(path=str((AUTHOR / "scripts/eval_rxn_center.py").relative_to(ROOT)), sha256=sha(AUTHOR / "scripts/eval_rxn_center.py")),
        TRAIN_ref=dict(path=str(TRAIN.relative_to(ROOT)), sha256=sha(TRAIN), bytes=TRAIN.stat().st_size),
        script_sha256=sha(__file__), runtime=dict(python=sys.version.split()[0], torch=native.torch.__version__),
        full_TRAIN=True, retained_limit=None, TEST_read=False, VALID_read=False, model_executed=False,
        labels_used_only_for_TRAIN_dataset_accounting=True, raw_rows=0, retained_rows=0, dropped_None_raw_ids=[],
        molecules=0, atoms=0, positive_atoms=0, directed_edges=0, max_atoms_per_reaction=0, max_molecules_per_reaction=0,
        malformed_bond_tokens_omitted_in_retained_reactions=0, native_fatal=None)
    ids, arrays = hashlib.sha256(), hashlib.sha256()
    began = time.monotonic()
    try:
        with TRAIN.open() as f:
            for i, line in enumerate(f):
                result["raw_rows"] += 1
                try:
                    ex = native._parse_line(i, line)
                except BaseException as error:
                    result["native_fatal"] = dict(raw_id=i, type=type(error).__name__, message=str(error)[:1024], line_sha256=hashlib.sha256(line.encode()).hexdigest())
                    raise
                if ex is None:
                    result["dropped_None_raw_ids"].append(i)
                    continue
                assert ex.rxn_id == i
                ids.update((str(i) + "\n").encode())
                result["retained_rows"] += 1
                atom_count = sum(len(x) for x in ex.labels)
                result["molecules"] += len(ex.mols)
                result["atoms"] += atom_count
                result["positive_atoms"] += int(sum(x.sum() for x in ex.labels))
                result["directed_edges"] += sum(x.shape[1] for x in ex.edges)
                result["max_atoms_per_reaction"] = max(result["max_atoms_per_reaction"], atom_count)
                result["max_molecules_per_reaction"] = max(result["max_molecules_per_reaction"], len(ex.mols))
                for maps, x, edge, y in zip(ex.atom_maps, ex.node_features, ex.edges, ex.labels):
                    assert x.dtype == native.np.float32 and x.shape == (len(maps), 19)
                    assert edge.dtype == native.np.int64 and y.dtype == native.np.float32
                    for value in (native.np.asarray(maps, dtype="<i8"), x, edge, y):
                        arrays.update(value.tobytes())
                edits = line.strip().split()[1].split(";")
                for component in edits[2:4]:
                    for token in component.split(","):
                        token = token.strip()
                        if not token:
                            continue
                        parts = token.split("-")
                        try:
                            if len(parts) != 3:
                                raise ValueError()
                            int(parts[0])
                            int(parts[1])
                            float(parts[2])
                        except ValueError:
                            result["malformed_bond_tokens_omitted_in_retained_reactions"] += 1
                del ex
        assert result["raw_rows"] == result["retained_rows"] + len(result["dropped_None_raw_ids"])
        n = result["retained_rows"]
        result.update(status="complete_all_TRAIN_no_fit", retained_raw_id_order_sha256=ids.hexdigest(),
            retained_native_array_content_sha256=arrays.hexdigest(),
            actual_updates_per_native20epoch_batch16_fit=20*((n+15)//16), native_scheduler_T_max=20*max(1,n//16))
    except BaseException:
        result["status"] = "failed_native_parser_partial_preserved"
        traceback.print_exc()
    result.update(seconds=time.monotonic()-began, finished_utc=datetime.now(timezone.utc).isoformat())
    raw = (json.dumps(result, sort_keys=True, indent=2)+"\n").encode()
    assert len(raw) <= 1000000
    with (HERE / "counts.json").open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "complete_all_TRAIN_no_fit" else 1
if __name__ == "__main__":
    raise SystemExit(main())
