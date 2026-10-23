"""Small, inspectable specifications. Scheduling and process supervision belong to ROOT."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
AUTHOR = ROOT / ".runtime/author_sources/argformer_f7545be7bca0"
NATIVE_PYTHON = ROOT / ".runtime/conda/envs/argformer_native_v1/bin/python"
CLASSES = ["EUR", "EAS", "AFR"]
VARIANTS = ("paper_aligned", "release_control")

def path(value):
    p = Path(value)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Every input and output must remain in the server repository")
    return p

def ref(value):
    p = path(value)
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 2**20), b""):
            h.update(block)
    return {"path": str(p.relative_to(ROOT)), "sha256": h.hexdigest()}

def read(record):
    if ref(record["path"]) != record:
        raise ValueError("Input identity changed: " + record["path"])
    return json.loads(path(record["path"]).read_text())

def save(value, destination):
    p = path(destination)
    with p.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return ref(p)

def build_spec(*, stage, preparation_ref, output_directory, variant="paper_aligned",
               checkpoint_completion_ref=None, num_workers=8, embedding_batch_size=256):
    if stage not in ("mlm", "contrastive", "predict") or variant not in VARIANTS:
        raise ValueError("Explicit supported phase and scientific variant required")
    prep = read(preparation_ref)
    if (prep.get("query_ancestry_truth_opened") is not False
            or prep.get("contrastive_labels_reference_only") is not True
            or prep.get("class_order") != CLASSES):
        raise ValueError("Observation-only prepared paths and reference-only supervision required")
    required = {v + "/" + s for v in ("mlm", "contrastive", "query_paths") for s in ("train", "val")}
    if set(prep["native_dataset_refs"]) != required:
        raise ValueError("Complete original native path corpus required")
    out = path(output_directory)
    if out.exists() or not out.is_relative_to(ROOT / "experiments/results"):
        raise ValueError("Fresh result directory required")
    if not 0 <= num_workers <= 16 or not 1 <= embedding_batch_size <= 4096:
        raise ValueError("Bounded worker and embedding batch settings required")
    predecessor = None
    if stage != "mlm":
        predecessor = read(checkpoint_completion_ref)
        expected = "mlm" if stage == "contrastive" else "contrastive"
        if (predecessor["stage"] != expected or predecessor["variant"] != variant
                or predecessor["preparation_ref"] != preparation_ref
                or predecessor.get("full_budget_completed") is not True):
            raise ValueError("Same-parent same-variant complete predecessor required")
    elif checkpoint_completion_ref is not None:
        raise ValueError("Corrected MLM starts fresh; no incompatible initializer")
    sources = [ref(p) for p in sorted(AUTHOR.rglob("*.py"))]
    sources += [ref(AUTHOR / ("yamls/" + p + ".yaml")) for p in ("mlm", "contrastive")]
    return dict(schema="paper_american_native_task_v1", stage=stage, variant=variant,
                parent_id=prep["parent_id"], preparation_ref=preparation_ref,
                checkpoint_completion_ref=checkpoint_completion_ref,
                output_directory=str(out.relative_to(ROOT)), num_workers=num_workers,
                embedding_batch_size=embedding_batch_size, author_source_refs=sources,
                adapter_source_refs=[ref(x) for x in sorted(Path(__file__).parent.glob("*.py"))],
                runtime_helper_ref=ref("research/reviews/American_C3_native_MLM_amendment_v12/abstract_unix_bootstrap_v1.py"),
                query_ancestry_truth_opened=False, expected_class_order=CLASSES,
                scientific_budget={"mlm_epochs": 5, "mlm_batch": 512,
                                   "contrastive_epochs": 1, "contrastive_batch": 4096},
                evaluator="reference_PCA10_nearest_class_centroid",
                evaluator_is_disclosed_study_choice=True)
