"""Reference-only exact covariance PCA and deterministic centroid study evaluator."""
from __future__ import annotations

class ReferencePCA:
    """O(d squared) memory. Every reference path contributes, no query fitting."""
    def __init__(self, dimensions=768, components=10):
        import numpy as np
        self.dimensions, self.components = dimensions, components
        self.count = 0
        self.mean = np.zeros(dimensions, dtype=np.float64)
        self.m2 = np.zeros((dimensions, dimensions), dtype=np.float64)
        self.class_sum = np.zeros((3, dimensions), dtype=np.float64)
        self.class_count = np.zeros(3, dtype=np.int64)
    def update(self, vectors, labels):
        import numpy as np
        x = np.asarray(vectors, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim != 2 or x.shape[1] != self.dimensions or len(x) != len(y) or not len(x):
            raise ValueError("Nonempty complete embedding batch and reference labels required")
        if not np.isfinite(x).all() or not ((y >= 0) & (y < 3)).all():
            raise ValueError("Finite reference vectors and source labels only")
        n = len(x)
        mu = x.mean(axis=0)
        centered = x - mu
        delta = mu - self.mean
        self.m2 += centered.T @ centered + np.outer(delta, delta) * self.count * n / (self.count + n)
        self.mean += delta * n / (self.count + n)
        self.count += n
        for c in range(3):
            take = x[y == c]
            self.class_count[c] += len(take)
            self.class_sum[c] += take.sum(axis=0)
    def finish(self):
        import numpy as np
        if self.count < self.components + 1 or not (self.class_count > 0).all():
            raise ValueError("At least eleven reference paths and every class required for PCA10")
        eigenvalues, axes = np.linalg.eigh(self.m2 / (self.count - 1))
        self.axes = axes[:, np.argsort(eigenvalues, kind="stable")[::-1][:self.components]]
        # Signs are fixed for a reproducible saved projection; distances are sign invariant.
        for j in range(self.components):
            k = abs(self.axes[:, j]).argmax()
            if self.axes[k, j] < 0:
                self.axes[:, j] *= -1
        self.centroids = (self.class_sum / self.class_count[:, None] - self.mean) @ self.axes
        self.eigenvalues = np.sort(eigenvalues)[::-1][:self.components]
        return self
    def predict(self, vectors):
        import numpy as np
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dimensions or not np.isfinite(x).all():
            raise ValueError("Finite query embeddings with the reference dimension required")
        z = (x - self.mean) @ self.axes
        squared = ((z[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        return squared.argmin(axis=1).astype(np.int64)
    def arrays(self):
        return dict(mean=self.mean, components=self.axes, centroids=self.centroids,
                    class_count=self.class_count, eigenvalues=self.eigenvalues)

def seal_predictions(registry, query_rows, calls):
    """Join full path identity before scoring. One-hot values carry hard calls."""
    import numpy as np
    identity = ("record_ids", "sample_nodes", "sample_indices", "positions_bp0", "parent_id", "class_order")
    ids = query_rows["record_ids"]
    if len(ids) != len(set(ids.tolist())) or set(ids.tolist()) != set(registry["record_ids"].tolist()):
        raise ValueError("Exactly one prediction for every prespecified full path required")
    lookup = {int(r): i for i, r in enumerate(ids)}
    join = np.asarray([lookup[int(r)] for r in registry["record_ids"]])
    for key in ("sample_nodes", "sample_indices"):
        if not np.array_equal(query_rows[key][join], registry[key]):
            raise ValueError("Sample identity differs despite record ID match")
    labels = np.asarray(calls)
    if labels.shape != ids.shape or not ((labels >= 0) & (labels < 3)).all():
        raise ValueError("One known class call for each original query path required")
    return {**{k: registry[k] for k in identity},
            "probabilities": np.eye(3, dtype=np.float32)[labels[join]]}
