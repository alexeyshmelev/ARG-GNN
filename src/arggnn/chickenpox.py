"""CC-BY-4.0 Chickenpox data preparation with chronological forecast boundaries.

Data are observed county-level weekly counts on a fixed geographic graph.
No model assessment or test-set scoring is performed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import urllib.request
import zipfile

import numpy as np
import torch

from .data import GraphSequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_URL = "https://archive.ics.uci.edu/static/public/580/hungarian+chickenpox+cases.zip"
DATA_PAGE = "https://archive.ics.uci.edu/dataset/580/hungarian+chickenpox+cases"
DATA_DOI = "10.24432/C5103B"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ARCHIVE_SHA256 = "dc306b1845c86616d2a72c202deeefac8fd1fcf0a34b3067cf6721e377d9da44"
ARCHIVE_NAME = "hungarian-chickenpox-cases.zip"


def _repository_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("All persistent Chickenpox files must remain inside the repository")
    return path


@dataclass(frozen=True)
class ChickenpoxData:
    counts: np.ndarray               # [T,20], edge-ID county order, raw cases
    dates: tuple[str, ...]            # ISO dates
    counties: tuple[str, ...]         # ID 0 through 19
    edge_index: np.ndarray           # [2,102], includes supplied self loops
    archive_sha256: str


@dataclass(frozen=True)
class ForecastProtocol:
    history: int = 52
    train_end: int = 365              # exclusive raw/target boundary
    validation_end: int = 443         # exclusive target boundary
    observations: int = 522

    def targets(self, split: str) -> np.ndarray:
        if not (1 <= self.history < self.train_end < self.validation_end < self.observations):
            raise ValueError("Invalid chronological forecast boundaries")
        if split == "train":
            return np.arange(self.history, self.train_end, dtype=np.int64)
        if split == "validation":
            return np.arange(self.train_end, self.validation_end, dtype=np.int64)
        if split == "test":
            return np.arange(self.validation_end, self.observations, dtype=np.int64)
        raise ValueError("split must be train, validation or test")


@dataclass(frozen=True)
class TrainingScaler:
    mean: np.ndarray
    scale: np.ndarray
    fit_end: int

    @classmethod
    def fit(cls, counts: np.ndarray, train_end: int) -> TrainingScaler:
        if counts.ndim != 2 or not 1 <= train_end <= counts.shape[0]:
            raise ValueError("Expected [T,N] counts and valid train_end")
        training = counts[:train_end].astype(np.float64)
        if not np.isfinite(training).all():
            raise ValueError("Training observations must be finite")
        mean = training.mean(axis=0)
        scale = training.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        return cls(mean, scale, train_end)

    def transform(self, counts: np.ndarray) -> np.ndarray:
        return (counts - self.mean) / self.scale

    def inverse(self, standardized: np.ndarray) -> np.ndarray:
        return standardized * self.scale + self.mean


def load_chickenpox_archive(path: str | Path) -> ChickenpoxData:
    """Read bounded public CSV assets without executing or extracting members."""
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Archive exceeds one MB bound")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Dataset archive hash differs from the frozen source version")
    archive = zipfile.ZipFile(io.BytesIO(data))
    members = archive.infolist()
    if sum(m.file_size for m in members) > 1024 * 1024:
        raise ValueError("Archive expansion exceeds one MB bound")
    expected = {"hungary_chickenpox.csv", "hungary_county_edges.csv"}
    if {m.filename for m in members} != expected or len(members) != 2:
        raise ValueError("Unexpected archive members")
    for member in members:
        parts = PurePosixPath(member.filename)
        if parts.is_absolute() or ".." in parts.parts or "\\" in member.filename:
            raise ValueError("Unsafe archive member")
        if stat.S_ISLNK((member.external_attr >> 16) & 0xffff):
            raise ValueError("Archive contains a symbolic link")
    rows = list(csv.reader(io.StringIO(archive.read("hungary_chickenpox.csv").decode("utf-8-sig"))))
    edges = list(csv.DictReader(io.StringIO(archive.read("hungary_county_edges.csv").decode("utf-8-sig"))))
    name_to_id = {}
    for edge in edges:
        for name, identity in [(edge["name_1"], int(edge["id_1"])),
                               (edge["name_2"], int(edge["id_2"]))]:
            if name in name_to_id and name_to_id[name] != identity:
                raise ValueError("Inconsistent county identity mapping")
            name_to_id[name] = identity
    if set(name_to_id.values()) != set(range(20)) or len(name_to_id) != 20:
        raise ValueError("Expected exactly 20 county identities")
    counties = tuple(name for name, _ in sorted(name_to_id.items(), key=lambda item: item[1]))
    column = {name: i for i, name in enumerate(rows[0])}
    if set(rows[0][1:]) != set(counties):
        raise ValueError("Case columns do not match the graph counties")
    dates = tuple(datetime.strptime(row[0], "%d/%m/%Y").date().isoformat() for row in rows[1:])
    parsed_dates = [datetime.fromisoformat(day) for day in dates]
    if len(dates) != 522 or len(set(dates)) != 522:
        raise ValueError("Expected 522 distinct observations in the pinned archive")
    if any((b - a).days != 7 for a, b in zip(parsed_dates, parsed_dates[1:])):
        raise ValueError("Expected weekly chronological observations")
    counts = np.asarray([[float(row[column[name]]) for name in counties]
                         for row in rows[1:]], dtype=np.float64)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Case counts must be finite and nonnegative")
    edge_index = np.asarray([[int(edge["id_1"]), int(edge["id_2"])]
                             for edge in edges], dtype=np.int64).T
    if edge_index.shape != (2, 102):
        raise ValueError("Expected 102 supplied adjacency entries")
    return ChickenpoxData(counts, dates, counties, edge_index, digest)


def download_chickenpox(raw_dir: str | Path = "data/raw/chickenpox") -> Path:
    """Acquire the explicitly CC-BY-4.0 archive only inside this repository."""
    raw_dir = _repository_path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / ARCHIVE_NAME
    if destination.exists():
        loaded = load_chickenpox_archive(destination)
        content = destination.read_bytes()
    else:
        with urllib.request.urlopen(DATA_URL, timeout=30) as response:
            content = response.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            raise ValueError("Download exceeds one MB bound")
        if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Downloaded source differs from the frozen hash")
        destination.write_bytes(content)
        loaded = load_chickenpox_archive(destination)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assets = {name: {"bytes": len(archive.read(name)),
                         "sha256": hashlib.sha256(archive.read(name)).hexdigest()}
                  for name in archive.namelist()}
    manifest = {
        "dataset": "Hungarian Chickenpox Cases", "doi": DATA_DOI,
        "source_page": DATA_PAGE, "archive_url": DATA_URL,
        "license": "CC BY 4.0", "license_url": LICENSE_URL,
        "archive_bytes": len(content), "archive_sha256": loaded.archive_sha256,
        "assets": assets,
        "observations": len(loaded.dates), "counties": list(loaded.counties),
        "first_date": loaded.dates[0], "last_date": loaded.dates[-1],
        "metadata_discrepancy": "UCI page reports 521 instances. The pinned CSV has 522 unique weekly observations.",
        "citation": "Hungarian Chickenpox Cases [Dataset]. (2021). UCI Machine Learning Repository. https://doi.org/10.24432/C5103B.",
    }
    (raw_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def history_permutation(history: int, seed: int, fix_last: bool = True) -> np.ndarray:
    if history < 1:
        raise ValueError("history must be positive")
    rng = np.random.default_rng(seed)
    if fix_last:
        return np.concatenate((rng.permutation(history - 1), np.asarray([history - 1])))
    return rng.permutation(history)


def forecast_sequence(data: ChickenpoxData, scaler: TrainingScaler, target: int,
                      history: int = 52, permutation: np.ndarray | None = None) -> GraphSequence:
    """Observed history only. The following-week label is not part of the input.

    The single scoring position is a readout anchor in the last observed graph,
    not a claim that the forecast target lies inside the historical domain.
    Calendar intervals are equally spaced weekly holds normalized to [0,1].
    No tree-clade meaning is assigned to county graphs. Clade metadata is zero.
    """
    if not history <= target < data.counts.shape[0]:
        raise ValueError("Target must follow a complete observed history")
    values = scaler.transform(data.counts[target - history:target])
    if permutation is not None:
        permutation = np.asarray(permutation)
        if (permutation.shape != (history,) or not np.issubdtype(permutation.dtype, np.integer)
                or not np.array_equal(np.sort(permutation), np.arange(history))):
            raise ValueError("permutation must reorder each history graph once")
        values = values[permutation]
    n = data.counts.shape[1]
    bounds = torch.linspace(0., 1., history + 1)
    intervals = torch.stack((bounds[:-1], bounds[1:]), dim=1)
    edge = torch.as_tensor(data.edge_index.copy(), dtype=torch.long)
    packed_edges = torch.cat([edge + graph * n for graph in range(history)], dim=1)
    seq = GraphSequence(
        x=torch.as_tensor(values.reshape(history * n, 1), dtype=torch.float32),
        edge_index=packed_edges,
        node_graph=torch.arange(history).repeat_interleave(n),
        node_ids=torch.arange(n).repeat(history),
        clades=torch.zeros(history * n, 1),
        intervals=intervals, query_node_ids=torch.arange(n),
        positions=intervals[-1].mean().reshape(1),
        target_graph=torch.tensor([history - 1]),
    )
    seq.validate()
    return seq


def last_value_forecast(raw_history: np.ndarray) -> np.ndarray:
    if raw_history.ndim != 2 or raw_history.shape[0] < 1:
        raise ValueError("Expected nonempty [history,N] values")
    return raw_history[-1].copy()


def seasonal_forecast(raw_history: np.ndarray, period: int = 52) -> np.ndarray:
    if raw_history.ndim != 2 or raw_history.shape[0] < period or period < 1:
        raise ValueError("Seasonal control requires at least period observations")
    return raw_history[-period].copy()


@dataclass(frozen=True)
class LinearRidgeForecaster:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    target_mean: np.ndarray
    scaler: TrainingScaler
    history: int
    fit_targets: tuple[int, ...]

    @classmethod
    def fit_training(cls, data: ChickenpoxData, protocol: ForecastProtocol,
                     scaler: TrainingScaler, ridge: float = 1.0) -> LinearRidgeForecaster:
        if not ridge > 0 or scaler.fit_end != protocol.train_end:
            raise ValueError("Positive ridge and matching training-only scaler required")
        targets = protocol.targets("train")
        standardized = scaler.transform(data.counts)
        x = np.stack([standardized[t - protocol.history:t].reshape(-1) for t in targets])
        y = standardized[targets]
        x_mean, y_mean = x.mean(0), y.mean(0)
        x, y = x - x_mean, y - y_mean
        # Dual ridge avoids a large feature-by-feature solve when history*N > T.
        gram = x @ x.T + ridge * np.eye(x.shape[0])
        coefficients = x.T @ np.linalg.solve(gram, y)
        return cls(coefficients, x_mean, y_mean, scaler, protocol.history,
                   tuple(int(t) for t in targets))

    def predict(self, raw_history: np.ndarray) -> np.ndarray:
        if raw_history.shape != (self.history, self.scaler.mean.size):
            raise ValueError("History shape does not match the fitted linear control")
        x = self.scaler.transform(raw_history).reshape(-1)
        prediction = (x - self.feature_mean) @ self.coefficients + self.target_mean
        return self.scaler.inverse(prediction)


def prepare_chickenpox(raw_dir: str | Path = "data/raw/chickenpox",
                       output_dir: str | Path = "data/processed/chickenpox",
                       protocol: ForecastProtocol | None = None) -> dict:
    """Save reusable arrays and a frozen protocol, without fitting or scoring models."""
    protocol = protocol or ForecastProtocol()
    output_dir = _repository_path(output_dir)
    source = download_chickenpox(raw_dir)
    data = load_chickenpox_archive(source)
    if data.counts.shape[0] != protocol.observations:
        raise ValueError("Protocol length differs from source observations")
    scaler = TrainingScaler.fit(data.counts, protocol.train_end)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = output_dir / "chickenpox.npz"
    np.savez_compressed(
        arrays, counts=data.counts.astype(np.float32),
        dates=np.asarray(data.dates), counties=np.asarray(data.counties),
        edge_index=data.edge_index, mean=scaler.mean, scale=scaler.scale,
        train_targets=protocol.targets("train"),
        validation_targets=protocol.targets("validation"),
        test_targets=protocol.targets("test"),
    )
    metadata = {
        "task": "One-week-ahead county-level count regression",
        "topology": "Fixed geographic adjacency, including original self loops",
        "observations": protocol.observations, "history_weeks": protocol.history,
        "train_end_exclusive": protocol.train_end,
        "validation_end_exclusive": protocol.validation_end,
        "scaler_fit_range": [0, protocol.train_end],
        "scaler": "Per-county mean/std from training observations only",
        "archive_sha256": data.archive_sha256,
        "arrays_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
        "county_order": list(data.counties), "license": "CC BY 4.0",
        "dataset_doi": DATA_DOI, "model_assessment_performed": False,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        targets = protocol.targets(split)
        metadata["splits"][split] = {
            "target_count": int(targets.size), "first_index": int(targets[0]),
            "last_index": int(targets[-1]), "first_date": data.dates[targets[0]],
            "last_date": data.dates[targets[-1]],
        }
    (output_dir / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
