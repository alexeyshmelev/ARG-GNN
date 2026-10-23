"""Small CPU functional checks with synthetic tensors and public metadata only."""
def run_smoke():
    import time
    import numpy as np
    import torch
    from torch.nn import functional as F
    from .native import inverse_frequency, weighted_supcon, LiteralMaskCollator, install_extant_head
    from .evaluation import ReferencePCA, seal_predictions
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    weights = inverse_frequency(labels.tolist())
    z = torch.randn(8, 12, requires_grad=True)
    loss, _ = weighted_supcon(z, labels, .05, weights)
    unit = F.normalize(z, dim=-1)
    sim = unit @ unit.T / .05
    eye = torch.eye(8, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    logprob = sim - torch.logsumexp(sim.masked_fill(eye, -torch.inf), 1, keepdim=True)
    anchor = -(logprob * positive).sum(1) / positive.sum(1)
    per = torch.tensor([weights[int(c)] for c in labels])
    expected = (anchor * (per / per.mean())).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    actual_grad = torch.autograd.grad(loss, z, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, z)[0]
    assert torch.allclose(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)
    def base(rows):
        return dict(input_ids=torch.tensor([[2, 9, 8, 3, 0, 0]]),
                    labels=torch.tensor([[-100, 5, 6, -100, -100, -100]]),
                    attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0]]))
    batch = LiteralMaskCollator(base, 1)([])
    assert batch["input_ids"].tolist() == [[2, 1, 1, 3]]
    assert batch["labels"].tolist() == [[-100, 5, 6, -100]]
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Identity()
            self.act = torch.nn.Identity()
            self.norm = torch.nn.Identity()
            self.drop = torch.nn.Identity()
    class Backbone:
        head = Head()
    backbone = Backbone()
    install_extant_head(backbone)
    hidden = torch.randn(2, 5, 12)
    assert torch.equal(backbone.head(hidden), hidden[:, 1])
    rng = np.random.default_rng(83)
    vectors = rng.normal(size=(60, 12))
    y = np.repeat(np.arange(3), 20)
    vectors += y[:, None] * np.arange(12)[None, :] / 4
    model = ReferencePCA(12, 10)
    for start in range(0, 60, 7):
        model.update(vectors[start:start+7], y[start:start+7])
    model.finish()
    assert np.allclose(model.mean, vectors.mean(0), atol=1e-12)
    assert np.allclose(model.m2 / 59, np.cov(vectors, rowvar=False), atol=1e-12)
    assert model.predict(vectors).shape == (60,)
    r = dict(record_ids=np.array([10, 11], dtype=np.int64),
             sample_nodes=np.array([101, 102], dtype=np.int64),
             sample_indices=np.array([601, 602], dtype=np.int64),
             positions_bp0=np.array([20., 30.]), parent_id=np.asarray("synthetic"),
             class_order=np.asarray(["EUR", "EAS", "AFR"]))
    rows = {k: r[k][::-1] for k in ("record_ids", "sample_nodes", "sample_indices")}
    sealed = seal_predictions(r, rows, np.array([2, 1]))
    assert sealed["probabilities"].argmax(1).tolist() == [1, 2]
    return dict(schema="paper_american_CPU_smoke_v1", passed=True, cpu_only=True,
                synthetic_inputs_only=True, training_updates=0, CUDA_context_created=False,
                checks=["inverse-frequency values", "weighted loss matches native formula",
                        "weighted loss gradients match native formula",
                        "all selected tokens become MASK", "padding crop preserves loss targets",
                        "extant representation is position one",
                        "streamed PCA covariance equals full covariance",
                        "full path identity join survives row reordering"],
                loss_absolute_error=float(abs(loss.detach() - expected.detach())),
                gradient_max_absolute_error=float(abs(actual_grad - expected_grad).max()),
                elapsed_seconds=time.monotonic() - started)
