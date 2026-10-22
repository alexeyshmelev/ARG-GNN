"""Small CPU checks for exact source, factories, resume and full-device policy.

No datasets, GPU kernels, scientific preparations or campaign workers are run.
The real AM checkpoint loader restores model and AdamW state on CPU; the test
intercepts only random-state restoration because its CUDA state needs admission.
"""
import ast
import copy
import json
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import ap_fit_v1 as ap
from . import fit_v1 as am
from . import full_vram
from research.repairs.benchmark_recovery_20261020.training.candidate_common import ROOT, file_ref, read_ref

HERE = Path(__file__).resolve().parent


def function_source(path, name):
    text = path.read_text()
    lines = text.splitlines()
    node = next(item for item in ast.parse(text).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    return "\n".join(lines[node.lineno-1:node.end_lineno])


class FullDeviceChecks(unittest.TestCase):
    def test_scientific_source_preserved(self):
        old = function_source(ROOT/am.ORIGINAL_FIT_REF["path"], "run_fit")
        new = function_source(Path(am.__file__), "run_fit")
        boundary = "    check('TreeLoom_upfront_full_RAM_admission')"
        self.assertEqual(old[old.index(boundary):], new[new.index(boundary):])
        old_ap = function_source(ROOT/ap.ORIGINAL_STAGE_REF["path"], "_run_scientific_fit")
        new_ap = function_source(Path(ap.__file__), "_run_scientific_fit")
        new_ap = new_ap.replace(
            "def _run_scientific_fit(spec,*,resource_check,acquire_gpu0,full_device_vram_permission):",
            "def _run_scientific_fit(spec,*,resource_check,acquire_gpu0):")
        new_ap = new_ap.replace(
            "    full_vram.enable_gpu(torch, dict(**spec, gpu_memory_policy=full_vram.POLICY,\n"
            "        full_device_vram_permission=full_device_vram_permission), writer.path, check)",
            "    common.finish_gpu_setup(torch, spec)")
        self.assertEqual(old_ap, new_ap)
        for module in (am, ap, full_vram):
            tree = ast.parse(Path(module.__file__).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr,
                        ("set_per_process_memory_fraction", "memory_reserved", "get_device_properties"))
        self.assertIs(ap.fixed_training_loop, ap.original.fixed_training_loop)
        self.assertTrue(all(file_ref(ref["path"]) == ref for ref in am.source_refs()+ap.source_refs()))

    def test_am_factory_keeps_inputs_and_requires_six_update_resume(self):
        job = read_ref(am.JOB)
        historical = job["call"]["spec"]
        inputs = historical["training_spec"]["train_inputs"]
        preparation_refs = {parent:item["native_preparation_ref"] for parent,item in inputs.items()}
        generation_refs = {parent:file_ref(Path(item["truth_ref"]["path"]).parents[2]/"observation_ready.json")
            for parent,item in inputs.items()}
        arguments = dict(preparation_refs=preparation_refs, generation_refs=generation_refs,
            output_directory="experiments/results/benchmark_full_vram_CPU_factory_check_AM/native",
            max_rss_bytes=512*2**30, qualification_ref=historical["qualification_ref"],
            physical_gpu=2, gpu_uuid=full_vram.gpu_binding.GPU_UUIDS[2])
        call = am.fit_factory(**arguments)
        self.assertEqual(call["module"], am.MODULE)
        self.assertEqual(call["spec"]["resume_checkpoint_ref"], am.RESUME_CHECKPOINT_REF)
        self.assertEqual(call["spec"]["observation_cache_bytes"], 352*2**30)
        self.assertEqual(call["spec"]["training_headroom_bytes"], 128*2**30)
        self.assertNotIn("max_cuda_reserved_bytes", call["spec"])
        full_vram.validate_permission(call["spec"])
        bad = dict(am.RESUME_CHECKPOINT_REF, sha256="0"*64)
        with self.assertRaises(ValueError):
            am.fit_factory(**arguments, resume_checkpoint_ref=bad)
        self.assertFalse((ROOT/arguments["output_directory"]).exists())

    def test_real_am_checkpoint_restores_all_49_cpu_optimizer_states(self):
        import torch
        from research.repairs.benchmark_recovery_20261020.runtime.binding import ReadableTreeLoom
        from research.repairs.benchmark_recovery_20261020.training.candidate_gru import install_chunked_gru
        from arggnn.benchmarking.treeloom_training import _optimizer_counts
        self.assertFalse(torch.cuda.is_initialized())
        fresh, original, registry, plans = am.metadata_bindings(read_ref(am.JOB),read_ref(am.BARRIER))
        schedule = [(epoch,parent) for epoch in range(1,31) for parent in registry.epoch_order(epoch)]
        with patch.object(torch.cuda, "init", side_effect=AssertionError("CPU check must not initialize CUDA")):
            model = install_chunked_gru(ReadableTreeLoom(registry).build_model(),chunk_holds=1024)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,
                betas=(.9,.999),eps=1e-8,maximize=False,amsgrad=False,foreach=False,fused=False,
                capturable=False,differentiable=False)
            with patch.object(am.checkpoints_io, "restore_random_state") as restore:
                start, checkpoints = am.checkpoints_io.load_resume(am.RESUME_CHECKPOINT_REF,
                    model=model,optimizer=optimizer,schedule=schedule,registry_ref=registry.reference,
                    barrier_ref=am.BARRIER,settings=original["settings"])
            self.assertEqual(start,6)
            self.assertEqual(len(_optimizer_counts(optimizer)),49)
            self.assertEqual(set(_optimizer_counts(optimizer)),{6})
            self.assertEqual(schedule[start],(1,"argformer_american_train_013"))
            self.assertEqual(len(schedule[start:]),594)
            self.assertEqual(checkpoints,{})
            restored_random = restore.call_args.args[0]
            self.assertEqual(len(restored_random["torch_CUDA"]),1)
            self.assertTrue(all(p.device.type=="cpu" for p in model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_ap_factory_preserves_all_science_and_rejects_resume(self):
        historical = read_ref(ap.ORIGINAL_JOB_REF)["call"]["spec"]
        scientific = historical["scientific_spec"]
        keys = ("protocol_id", "sampling_plan_ref", "train_inputs", "observation_cache_bytes",
            "max_sites", "max_metadata_bytes", "preparation_limits", "max_output_bytes", "physical_gpu", "gpu_uuid")
        arguments = {key:copy.deepcopy(scientific[key]) for key in keys}
        arguments.update(scientific["memory_envelope"])
        for key in ("analytical_parent_peak_RSS_proven", "no_parent_reduction_or_partial_update", "reserve_basis"):
            arguments.pop(key)
        arguments["output_directory"] = "experiments/results/benchmark_full_vram_CPU_factory_check_AP/native"
        arguments.update(physical_gpu=5,gpu_uuid=full_vram.gpu_binding.GPU_UUIDS[5])
        call = ap.fit_factory(qualification_ref=historical["qualification_ref"], **arguments)
        expected = copy.deepcopy(scientific)
        for key in ("output_directory", "physical_gpu", "gpu_uuid"):
            expected[key] = arguments[key]
        self.assertEqual(call["spec"]["scientific_spec"], expected)
        self.assertEqual(call["spec"]["qualification_ref"], historical["qualification_ref"])
        self.assertEqual(expected["optimizer_updates"],90)
        self.assertEqual(call["module"],ap.MODULE)
        self.assertNotIn("resume_checkpoint_ref",call["spec"])
        self.assertFalse((ROOT/arguments["output_directory"]).exists())
        with self.assertRaises(ValueError):
            ap.fit_factory(qualification_ref=historical["qualification_ref"],resume_checkpoint_ref=None,**arguments)
        changed = copy.deepcopy(arguments)
        changed["training_headroom_bytes"] = 129*2**30
        with self.assertRaises(ValueError):
            ap.fit_factory(qualification_ref=historical["qualification_ref"],**changed)

    def test_full_device_setup_uses_identity_and_title_without_allocator_cap(self):
        for physical,uuid in full_vram.gpu_binding.GPU_UUIDS.items():
            spec = dict(physical_gpu=physical,gpu_uuid=uuid,gpu_memory_policy="full_device",
                full_device_vram_permission=full_vram.permission(physical,uuid))
            full_vram.validate_permission(spec)
        with self.assertRaises(ValueError):
            full_vram.permission(0,full_vram.gpu_binding.GPU_UUIDS[2])
        calls=[]
        fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1,
            reset_peak_memory_stats=lambda device:calls.append(("reset",device))),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(allow_tf32=True)))
        spec = dict(physical_gpu=2,gpu_uuid=full_vram.gpu_binding.GPU_UUIDS[2],
            gpu_memory_policy="full_device",full_device_vram_permission=full_vram.permission(2,full_vram.gpu_binding.GPU_UUIDS[2]))
        environment = {key:value for key,value in os.environ.items() if key!="ARGGNN_MAX_CUDA_RESERVED_BYTES"}
        environment["ARGGNN_GPU_MEMORY_POLICY"]="full_device"
        with tempfile.TemporaryDirectory(prefix="CPU_policy_check_",dir=HERE) as directory:
            with patch.dict(os.environ,environment,clear=True), patch.object(
                    full_vram.gpu_binding,"verify_before_cuda") as identity, patch(
                    "setproctitle.getproctitle",return_value=full_vram.PROCESS_TITLE):
                full_vram.enable_gpu(fake,spec,Path(directory),calls.append)
                identity.assert_called_once_with(fake,spec)
                metadata=json.loads((Path(directory)/"gpu_binding.json").read_text())
                self.assertEqual(metadata["full_device_vram_permission"],spec["full_device_vram_permission"])
                self.assertEqual(calls,["TreeLoom_before_full_device_CUDA",("reset",0)])
                os.environ["ARGGNN_MAX_CUDA_RESERVED_BYTES"]="1"
                with self.assertRaises(ValueError):
                    full_vram.enable_gpu(fake,spec,Path(directory),calls.append)


if __name__ == "__main__":
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    raise SystemExit(0 if result.wasSuccessful() else 1)
