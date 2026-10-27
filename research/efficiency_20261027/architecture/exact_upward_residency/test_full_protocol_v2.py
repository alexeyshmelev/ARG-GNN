"""Metadata-only checks for full qualification prerequisites and finite budgets."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import torch
from . import compare_full_v2 as runner
from .full_release_v2 import validate_numerical_release, require_comparison


def main():
    def rejected(function):
        try:
            function()
        except (ValueError, AssertionError):
            return
        raise AssertionError('Expected prerequisite rejection')
    comparison = dict(passed=True,argmax_equal=True,argmax_disagreements=0,categories={})
    for name,count in dict(logits=1,loss=1,gradients=49,parameters=49,moments=147).items():
        comparison['categories'][name]=dict(passed=True,tensors=count,mismatched_elements=0,
            mismatched_paths=[],structural_errors=[],atol=1e-6,
            rtol=1e-5 if name in ('logits','loss') else 1e-4)
    checkpoint_ref=dict(path='fixture_checkpoint',sha256='1'*64)
    manifest_ref=dict(path='fixture_manifest',sha256='2'*64)
    common=dict(status='completed',all_strict_comparisons_passed=True,
        checkpoint_ref=checkpoint_ref,source_manifest_ref=manifest_ref)
    scope=dict(parent_id='argformer_american_train_000',haplotypes=1600,heldout_truth_opened=False)
    prefix=dict(common,schema='TreeLoom_upward_direct_residency_prefix_GPU_qualification_v1',
        actual_49gradient_and_Adam_checks=True,
        scope=dict(scope,native_holds=1153,full_parent_claim=False),records=[])
    for phase,repeat,kinds in [('warmup',0,runner.KINDS),('timed',1,runner.KINDS),
                               ('timed',2,tuple(reversed(runner.KINDS)))]:
        for kind in kinds:
            prefix['records'].append(dict(kind=kind,phase=phase,repeat=repeat,optimizer_step=242,
                parameter_count=49,profiled=False,CPU_context=True,hotpack=False,
                comparison=copy.deepcopy(comparison)))
    full=dict(common,schema='TreeLoom_boundary_full_v1',
        scope=dict(scope,mode='full',native_holds=77472,sites=227474,
            supervised_original_focals=10000,full_parent_claim=True,physical_support_bp=[0,30000000]),
        records=[dict(kind=kind,CPU_context=True,hotpack=False,comparison=copy.deepcopy(comparison))
                 for kind in ('selected_explicit_CPU512','boundary_fused_CPU512')])
    def validate(value,kind):
        return validate_numerical_release(value,kind,checkpoint_ref=checkpoint_ref,
                                          source_manifest_ref=manifest_ref)
    assert validate(prefix,'prefix')['numerical_records']==6
    assert validate(full,'boundary_full')['numerical_records']==2
    failed=copy.deepcopy(prefix);failed['records'][0]['comparison']['passed']=False
    rejected(lambda: validate(failed,'prefix'))
    wrong=copy.deepcopy(comparison);wrong['categories']['moments']['tensors']=146
    rejected(lambda: require_comparison(wrong))
    tolerance=copy.deepcopy(comparison);tolerance['categories']['gradients']['rtol']=1e-3
    rejected(lambda: require_comparison(tolerance))
    argmax=copy.deepcopy(comparison);argmax['argmax_disagreements']=1
    rejected(lambda: require_comparison(argmax))
    shuffled=copy.deepcopy(prefix);shuffled['records'].reverse()
    rejected(lambda: validate(shuffled,'prefix'))
    missing=copy.deepcopy(full);missing['scope']['supervised_original_focals']=9999
    rejected(lambda: validate(missing,'boundary_full'))
    partial=copy.deepcopy(full);partial['scope']['physical_support_bp']=[0,29999999]
    rejected(lambda: validate(partial,'boundary_full'))
    wrongpin=copy.deepcopy(prefix);wrongpin['source_manifest_ref']['sha256']='3'*64
    rejected(lambda: validate(wrongpin,'prefix'))
    assert runner.EXPECTED_NATIVE_NODES==205369272
    assert runner.KINDS==('boundary_fused_CPU512','upward_direct_resident_CPUcontext512')
    assert runner.DEVICE_BUDGET_BYTES==74*2**30
    payload=2*runner.EXPECTED_NATIVE_NODES*32*4
    assert payload==52574533632
    assert payload+runner.DEFAULT_SCRATCH_BYTES==69754402816<runner.DEVICE_BUDGET_BYTES
    historical_total=85094825984
    assumed_peer=1800000000
    assert runner.DEVICE_BUDGET_BYTES+2*2**30+assumed_peer<historical_total
    tree=ast.parse(Path(runner.__file__).read_text())
    assert sum(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
               and node.func.id=='run_update' for node in ast.walk(tree))==1
    assert 'torch.profiler' not in Path(runner.__file__).read_text()
    manifest=Path(runner.MANIFEST)
    pinned=runner.verify_pins(hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert not torch.cuda.is_initialized()
    report=dict(schema='TreeLoom_full_residency_protocol_CPU_metadata_v1',passed=True,
        valid_prefix_and_full_metadata_accepted=True,failed_or_reordered_prefix_rejected=True,
        missing_Adam_tensor_rejected=True,looser_tolerance_rejected=True,argmax_disagreement_rejected=True,
        missing_focal_or_partial_support_rejected=True,wrong_source_pin_rejected=True,
        actual_full_node_arithmetic_verified=True,one_full_call_per_arm=True,
        device_budget_bytes=runner.DEVICE_BUDGET_BYTES,two_bank_bytes=payload,
        scratch_bytes=runner.DEFAULT_SCRATCH_BYTES,
        historical_device_bytes=historical_total,illustrative_GST_peer_bytes=assumed_peer,
        gate_margin_after_assumed_peer_bytes=historical_total-runner.DEVICE_BUDGET_BYTES-2*2**30-assumed_peer,
        live_admission_still_required=True,GPU_initialized=False,scientific_payload_read=False,
        GPU_run_launched=False,source_manifest_ref=pinned['source_manifest_ref'])
    path=Path(__file__).with_name('FULL_PROTOCOL_CPU_v2.json')
    with path.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
