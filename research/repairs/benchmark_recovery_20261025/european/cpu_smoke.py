"""Authenticate completed AP fit and test only small CPU recurrent execution."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
from pathlib import Path
import copy
import hashlib
import json
import torch
from . import ap_prediction_chunked_gru as repair

OUT = Path("research/repairs/benchmark_recovery_20261025/european")
JOB = Path(".runtime/benchmarking/paper_campaign_20261024_v2/jobs/paper_ap_treeloom__all_test__predict.json")


def ref(path):
    path=Path(path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    task=json.loads(JOB.read_text())
    kwargs=task["call"]["factory"]["kwargs"]
    complete=repair.original._fit_completion(kwargs["fit_completion_ref"],
        protocol_id=repair.PROTOCOL,sampling_plan_ref=kwargs["sampling_plan_ref"],
        train_count=3)
    checkpoint=Path(complete["checkpoint_ref"]["path"])
    assert ref(checkpoint)==complete["checkpoint_ref"]
    assert checkpoint.stat().st_size<=8*2**20
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True)
    model=repair.european.build_model(repair.PROTOCOL)
    repair.original._validate_saved(saved,model,protocol_id=repair.PROTOCOL,
        sampling_plan_ref=kwargs["sampling_plan_ref"],updates=90,
        barrier_ref=complete["observation_barrier_ref"])
    assert saved["epoch"]==30 and saved["optimizer_updates"]==90
    assert len(saved["optimizer_state"]["state"])==49
    validation=dict(schema="AP_final_fit_checkpoint_CPU_validation_v1",
        status="usable_final_checkpoint",fit_completion_ref=kwargs["fit_completion_ref"],
        checkpoint_ref=complete["checkpoint_ref"],checkpoint_bytes=checkpoint.stat().st_size,
        completed_epochs=30,optimizer_updates=90,all49_Adam_counters_equal90=True,
        exact_model_and_optimizer_shapes_finite=True,
        model_parameter_tensors=len(list(model.parameters())),
        model_state_keys=len(model.state_dict()),class_order=list(model.output_classes),
        training_repeated=False,TEST_truth_read=False,CUDA_initialized=False)
    reference=copy.deepcopy(model).eval()
    before=[(name,id(value),tuple(value.shape)) for name,value in model.named_parameters()]
    keys=tuple(model.state_dict())
    rng=torch.get_rng_state().clone()
    repair.install_model_execution(model).eval()
    assert before==[(name,id(value),tuple(value.shape)) for name,value in model.named_parameters()]
    assert keys==tuple(model.state_dict())
    assert torch.equal(rng,torch.get_rng_state())
    generator=torch.Generator().manual_seed(20261025)
    sequence=torch.randn(2,2051,35,generator=generator).transpose(0,1)
    assert not sequence.is_contiguous()
    incoming=torch.randn(2,2,32,generator=generator)
    with torch.no_grad():
        expected,expected_final=reference.genomic_gru(sequence,incoming)
        actual,actual_final=model.genomic_gru(sequence,incoming)
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(actual_final,expected_final,atol=1e-5,rtol=1e-5)
        holds=torch.tensor([0,1023,1024,2047,2050])
        columns=torch.tensor([0,1,0,1,1])
        old_logits=reference.classifier(torch.cat(
            (sequence[holds,columns,:32],expected[holds,columns]),-1))
        new_logits=model.classifier(torch.cat(
            (sequence[holds,columns,:32],actual[holds,columns]),-1))
        torch.testing.assert_close(old_logits,new_logits,atol=1e-5,rtol=1e-5)
    assert not torch.cuda.is_initialized()
    validation["source_refs"]=[ref(p) for p in (
        "src/arggnn/benchmarking/european_training.py",
        "src/arggnn/benchmarking/paper_argmix/paper_argmix_ap_prediction.py")]
    (OUT/"checkpoint_validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    result=dict(schema="AP_prediction_chunked_GRU_CPU_smoke_v1",passed=True,
        checkpoint_validation_ref=ref(OUT/"checkpoint_validation.json"),
        recurrent_input_shape=list(sequence.shape),input_noncontiguous=True,
        nonzero_incoming_state=True,chunk_holds=repair.GRU_CHUNK_HOLDS,
        chunks_per_direction=3,full_forward_reverse_carry=True,
        output_max_abs_error=float((expected-actual).abs().max()),
        final_state_max_abs_error=float((expected_final-actual_final).abs().max()),
        sampled_logits_max_abs_error=float((old_logits-new_logits).abs().max()),
        parameter_identities_and_state_keys_preserved=True,RNG_unchanged=True,
        source_refs=[ref(p) for p in (
            OUT/"ap_prediction_chunked_gru.py",Path(__file__),
            Path(repair.european_gru.__file__))],
        CUDA_initialized=False,optimizer_created=False,truth_read=False,
        limitation="Small CPU parity and checkpoint authentication only; no CUDA production reproduction or full-parent prediction claim.")
    (OUT/"cpu_smoke.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"checkpoint":validation,"smoke":result},indent=2))


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(passed=False,error_type=type(error).__name__,error=str(error))),flush=True)
        raise
