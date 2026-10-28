"""Two-step AdamW connectivity qualification without labels or model fitting."""
import hashlib
import json
from pathlib import Path
import torch
from . import window_model_frozen_v1 as old
from . import window_model_focal_executor_v2 as new
from . import test_focal_executor_v2 as fixtures


def main():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    first = old.WindowTreeLoom(width=32)
    second = new.WindowTreeLoom(width=32)
    second.load_state_dict(first.state_dict())
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.001, eps=1e-8, weight_decay=.01)
                  for model in (first, second)]
    empty = fixtures.edgeless_focal_with_edged_sources()
    selected = empty['site_tree_index'] != 1
    for key in ('site_tree_index', 'site_position_local_bp', 'site_genotypes', 'site_allele_count'):
        empty[key] = empty[key][selected]
    empty['focal_genotypes'][:] = -1
    edged = fixtures.select_focals(fixtures.toy(), [1])
    report = dict(schema='focal_executor_v2_sequential_AdamW_connectivity', tests=[], labels_used=False,
                  ancestry_fitting=False, optimizer_steps=2, mode='no_cross', source_sha256={}, all_passed=True)
    for name in ('window_model_frozen_v1.py', 'window_model_focal_executor_v2.py', 'qualify_focal_executor_sequence_v2.py'):
        report['source_sha256'][name] = hashlib.sha256((fixtures.ROOT / name).read_bytes()).hexdigest()
    for index, arrays in enumerate((empty, edged)):
        source = old.prepare_observations(arrays)
        batch = new.prepare_executor(source)
        outputs = []
        norms = []
        for model, model_batch, optimizer in zip((first, second), (source, batch), optimizers):
            optimizer.zero_grad(set_to_none=True)
            output = model(model_batch, mode='no_cross')
            coefficient = torch.linspace(-.7, 1.1, output.numel()).reshape_as(output)
            loss = (output * coefficient).sum() * 10 + output.square().sum()
            loss.backward()
            outputs.append(output.detach())
        max_gradient = 0.
        none_names = []
        for (name, parameter), (_, candidate) in zip(first.named_parameters(), second.named_parameters()):
            assert (parameter.grad is None) == (candidate.grad is None), name
            if parameter.grad is None:
                none_names.append(name)
            else:
                torch.testing.assert_close(candidate.grad, parameter.grad, atol=2e-5, rtol=2e-4)
                max_gradient = max(max_gradient, fixtures.max_error(candidate.grad, parameter.grad))
        for model, optimizer in zip((first, second), optimizers):
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), .5)))
            optimizer.step()
        max_parameter = 0.
        max_state = 0.
        for (name, parameter), (_, candidate) in zip(first.named_parameters(), second.named_parameters()):
            torch.testing.assert_close(candidate, parameter, atol=3e-6, rtol=2e-5)
            max_parameter = max(max_parameter, fixtures.max_error(candidate, parameter))
            a, b = optimizers[0].state[parameter], optimizers[1].state[candidate]
            assert a.keys() == b.keys(), name
            for key in a:
                torch.testing.assert_close(b[key], a[key], atol=2e-7, rtol=2e-4)
                max_state = max(max_state, fixtures.max_error(b[key], a[key]))
        report['tests'].append(dict(step=index + 1, description='edgeless focal, no pooled SNPs, missing focal genotype' if index == 0 else 'edged focal with SNPs',
            logits_max_abs_error=fixtures.max_error(outputs[1], outputs[0]), gradient_max_abs_error=max_gradient,
            parameter_max_abs_error=max_parameter, optimizer_state_max_abs_error=max_state,
            gradient_norms_before_clip=norms, none_gradient_names=none_names, passed=True))
    report['CUDA_initialized'] = torch.cuda.is_initialized()
    assert not report['CUDA_initialized']
    (fixtures.ROOT / 'FOCAL_EXECUTOR_V2_SEQUENCE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
