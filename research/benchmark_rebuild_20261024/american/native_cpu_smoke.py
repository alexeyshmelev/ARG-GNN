"""Exercise the actual author ARG loader and literal-mask hook on tiny CPU data."""
import json
import os
import pickle
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from arggnn.benchmarking.paper_american import protocol as p
from arggnn.benchmarking.paper_american.native import adaptations

torch.set_num_threads(1)
sys.path.insert(0, str(p.AUTHOR))
import main
from src.arg_data import ARGTokenizer, ARGPreTrainedTokenizer

out = p.path("research/benchmark_rebuild_20261024/american/native_cpu_smoke")
(out / "data/train").mkdir(parents=True, exist_ok=False)
vocab = {"special": {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[SEP]": 3},
         "node_id_tokens": {"node_0": 4, "node_1": 5, "node_2": 6}}
with (out / "data/train/raw_train_sequences_and_vocab.pkl").open("xb") as stream:
    pickle.dump({"raw_sequences": [[0, 1], [0, 2]], "vocab": vocab}, stream)
tokenizer = ARGPreTrainedTokenizer(ARGTokenizer(vocab))
cfg = OmegaConf.create({"name": "arg", "dataset": {
    "streaming": False, "local": str(out / "data"), "split": "train", "remote": None,
    "max_seq_len": 128, "skip_extant_tokens": False, "mlm_probability": 1.0},
    "drop_last": False, "num_workers": 0})
with adaptations(main, cfg, phase="mlm"):
    loader = main.arg_data_module.build_arg_dataloader(
        cfg=cfg, tokenizer=tokenizer, device_batch_size=2, device_microbatch_size=2)
    batch = next(iter(loader))
assert batch["input_ids"].shape == (2, 4)
assert batch["input_ids"].tolist() == [[2, 1, 1, 3], [2, 1, 1, 3]]
assert batch["labels"][:, 1].tolist() == [4, 4]
assert (batch["labels"][:, 0] == -100).all() and (batch["labels"][:, -1] == -100).all()
result = {"schema": "paper_american_actual_author_CPU_loader_smoke_v1",
          "passed": True, "CPU_only": True, "synthetic_rows": 2,
          "actual_author_loader_used": True, "retained_extant_token_target_verified": True,
          "every_selected_node_replaced_by_MASK": True, "special_tokens_preserved": True,
          "cropped_width": 4, "original_max_seq_len": 128, "training_updates": 0,
          "CUDA_context_created": False}
p.save(result, out / "result.json")
print(json.dumps(result))
