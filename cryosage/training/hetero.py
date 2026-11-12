# Adapted from CryoSTAR under Apache-2.0. See NOTICE and provenance.json.
"""Template-free factorized Gaussian VAE for heterogeneous reconstruction.

The particle encoder supplies the latent coordinate used by the position and
occupancy prediction networks:

    centered particle -> (optional CTF phase flip) -> Hartley image
                      -> encoder -> (z_mu, z_logvar) -> sampled z
                      -> Gaussian deformation field -> projection

The RELION rotation/translation stored in the STAR file remain the known imaging
pose.  The VAE latent represents particle conformation; it is not an SO(3) pose.
The model separates position dynamics from a low-rank compositional occupancy
head.  It contains no molecule-specific class names, labels, or state ordering.
Optional external labels are used only for post-hoc validation and never enter
the training objective.
With ``model.hetero=False``, neither the VAE encoder nor the deformation field is
constructed; all particles share one canonical Gaussian volume.

Factorized composition models use ``deform_attributes='xyz_occupancy'`` and
``deform_decoder='factorized'``. Position-only models use ``'xyz'`` and
``'joint'``, respectively. Other settings include:

    model = dict(
        ...,
        hetero=True,               # False: canonical homo reconstruction
        gaussian_parameterization="density",  # "density" (peak) or "amplitude" (integral)
        init_density=9e-4,          # also defines equivalent amplitude initialization
        # init_amplitude=...,       # optional explicit override in amplitude mode
        renderer_gain=1.0,          # >1 lowers R2's effective absolute alpha cutoff
        deform_attributes="xyz",    # "xyz", "xyz_occupancy", or "all"
        occupancy_init=0.95,          # only used by xyz_occupancy; state-dependent multiplier
        occupancy_temperature=1.0,    # sigmoid temperature for occupancy logits
        deform_pe_freqs=4,          # 0 disables xyz Fourier features
        deform_pe_include_xyz=True,
    )

    vae = dict(
        encoder_type="resid",       # "resid" (cryoDRGN-like) or "conv"
        encoder_hidden=256,
        encoder_layers=3,
        encoder_mask_radius=1.0,    # fraction of the Fourier/Hartley half-width
        phase_flip=True,
        beta=None,                  # None -> 1 / model.z_dim, as in cryoDRGN
        beta_warmup_epochs=5,
        logvar_min=-10.0,
        logvar_max=10.0,
        export_batch_size=32,
    )

    optim = dict(..., encoder_lr=1e-4)

    loss = dict(
        type="fourier_mse",         # or "real_mse"
        mask_radius=1.0,            # frequency mask used by fourier_mse
        real_mse_scale="parseval", # H*W; keeps real/Fourier gradient scales comparable
    )

    # The optimizer can split every physical Gaussian attribute and decay all
    # groups by the same multiplier while preserving their LR ratios:
    optim = dict(
        position_lr=1e-4,
        scale_lr=5e-4,
        rotation_lr=1e-4,
        strength_lr=1e-3,
        deform_lr=1e-4,
        encoder_lr=1e-4,
        scheduler="cosine",        # "none", "cosine", or "exponential"
        min_lr_ratio=0.1,
        eps=1e-8,                   # use 1e-15 for GEM-style real-space training
        weight_decay=0.0,
    )

    eval = dict(
        ...,
        umap_n_neighbors=50,        # UMAP visualization of the learned z_mu manifold
        umap_min_dist=0.08,
        umap_metric="euclidean",
        umap_random_state=42,
        umap_num_representatives=10, # >=10; k-means medoids selected in 2D UMAP
        export_umap_volumes=True,
        pca_dims=2,                 # independently export maps along PCA axes
        pca_steps=4,
        export_pca_volumes=True,
        # class_label_file=...,     # optional post-hoc validation only
    )

At every configured export epoch, and unconditionally after the final epoch,
heterogeneous mode saves posterior means/log-variances and a two-dimensional UMAP
plot of z_mu.  UMAP-space k-means chooses at least ten representative real
particles, marked U000/U001/... and exported as ``vol_umap_*.mrc``.  PCA paths
are sampled and exported separately as ``vol_pca_*``.  Homogeneous mode exports
only its checkpoint and consensus volume.
"""
from __future__ import annotations

import csv
import math
import os
import os.path as osp
import pickle
import shutil
import tempfile
import zlib
from typing import Any

# UMAP imports numba kernels. Some non-interactive launchers do not provide a
# writable/default cache locator, so give numba a stable cache path up front.
os.environ.setdefault("NUMBA_CACHE_DIR", osp.join(tempfile.gettempdir(), "cryosage_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", osp.join(tempfile.gettempdir(), "cryosage_matplotlib_cache"))

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from mmengine import mkdir_or_exist
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from cryosage.utils.ctf_utils import CTFParticle, CTFRelion
from cryosage.utils.fft_utils import fourier_to_primal_2d, primal_to_fourier_2d
from cryosage.utils.latent import sample_along_pca
from cryosage.utils.runtime import create_circular_mask, log_to_current, pl_init_exp, pretty_dict
from cryosage.utils.mrc_tools import save_mrc
from cryosage.utils.transforms import FourierGridTranslate, SpatialGridTranslate

from cryosage.dataio import StarDataset, StarDatasetConfig
from cryosage.utils.optics import infer_ctf_params_from_config
from cryosage import dataio as _dataio

from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)


TASK_NAME = ""
log_to_current = rank_zero_only(log_to_current)

DEFAULT_CLASS_LABEL_NAMES: dict[Any, str] = {}
DEFAULT_CLASS_LABEL_COLORS: dict[Any, str] = {}


def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read an optional config field."""
    if obj is None:
        return default
    try:
        if key in obj:
            return obj[key]
        return default
    except TypeError:
        pass
    return getattr(obj, key, default)


def normalize_deform_attributes(value: Any) -> str:
    """Normalize the per-particle hetero Gaussian attribute mode."""
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "xyz": "xyz",
        "position": "xyz",
        "positions": "xyz",
        "pos": "xyz",
        "xyz_occupancy": "xyz_occupancy",
        "xyz_occ": "xyz_occupancy",
        "position_occupancy": "xyz_occupancy",
        "position_occ": "xyz_occupancy",
        "occupancy": "xyz_occupancy",
        "all": "all",
        "all_attributes": "all",
        "full": "all",
    }
    if text not in aliases:
        raise ValueError(
            "model.deform_attributes must be 'xyz'/'position', 'xyz_occupancy', or 'all', "
            f"got {value!r}."
        )
    return aliases[text]


def normalize_occupancy_temperature_schedule(
    value: Any, initial_temperature: float
) -> tuple[tuple[int, float], ...]:
    """Validate a 1-based, piecewise-linear occupancy-temperature schedule."""
    if value is None:
        return ((1, float(initial_temperature)),)

    points: list[tuple[int, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                "model.occupancy_temperature_schedule entries must be "
                "(epoch, temperature) pairs."
            )
        raw_epoch, raw_temperature = item
        epoch = int(raw_epoch)
        temperature = float(raw_temperature)
        if epoch < 1 or float(raw_epoch) != epoch:
            raise ValueError("Occupancy-temperature schedule epochs must be positive integers.")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Occupancy-temperature schedule values must be positive and finite.")
        if points and epoch <= points[-1][0]:
            raise ValueError("Occupancy-temperature schedule epochs must increase strictly.")
        points.append((epoch, temperature))

    if not points:
        raise ValueError("model.occupancy_temperature_schedule must not be empty.")
    if points[0][0] > 1:
        points.insert(0, (1, float(initial_temperature)))
    return tuple(points)


def interpolate_epoch_schedule(
    schedule: tuple[tuple[int, float], ...], epoch: int
) -> float:
    """Evaluate a piecewise-linear schedule at a 1-based epoch."""
    epoch = max(1, int(epoch))
    if epoch <= schedule[0][0]:
        return schedule[0][1]
    for (left_epoch, left_value), (right_epoch, right_value) in zip(
        schedule, schedule[1:]
    ):
        if epoch <= right_epoch:
            fraction = (epoch - left_epoch) / (right_epoch - left_epoch)
            return left_value + fraction * (right_value - left_value)
    return schedule[-1][1]


def ensure_ctf_compatibility(cfg):
    """Create the legacy aliases required by the existing CTF utilities."""
    if not hasattr(cfg, "dataset_attr"):
        cfg.dataset_attr = dict(
            dataset_dir=cfg.dataset.root,
            starfile_path=cfg.dataset.star,
            apix=cfg_get(cfg.dataset, "apix", None),
            side_shape=cfg_get(cfg.dataset, "size", None),
            f_mu=cfg_get(cfg.dataset, "f_mu", None),
            f_std=cfg_get(cfg.dataset, "f_std", None),
        )
    if not hasattr(cfg, "data_process"):
        cfg.data_process = dict(
            down_side_shape=cfg_get(cfg.dataset, "down_size", cfg_get(cfg.dataset, "size", None)),
            down_method=cfg_get(cfg.dataset, "down_method", "interp"),
            mask_rad=cfg_get(cfg.dataset, "mask_radius", 1.0),
        )
    cfg.model.shift_method = str(cfg_get(cfg.model, "shift_method", "interp"))
    cfg.model.ctf = str(cfg_get(cfg.model, "ctf", "v1"))
    return cfg


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    # ``eps`` (~1.19e-7 for float32) is far too large for integrated Gaussian
    # amplitudes: a density around 9e-4 at scale 0.012 corresponds to roughly
    # 2.45e-8.  ``expm1`` remains accurate there, so only guard against zero
    # using the smallest positive normal value.
    tiny = torch.finfo(x.dtype).tiny
    x = x.clamp_min(tiny)
    return x + torch.log(-torch.expm1(-x))


def compute_latent_pca(latents: np.ndarray) -> dict:
    """Compute particle-latent PCA using NumPy only."""
    z = np.asarray(latents, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
        raise ValueError(f"Expected latents shaped (N,z_dim), got {z.shape}.")
    mean = z.mean(axis=0, keepdims=True)
    u, singular_values, components = np.linalg.svd(z - mean, full_matrices=False)
    scores = u * singular_values[None, :]
    explained_variance = np.square(singular_values) / max(z.shape[0] - 1, 1)
    total = float(explained_variance.sum())
    ratio = explained_variance / total if total > 0 else np.zeros_like(explained_variance)
    return dict(
        mean=mean.squeeze(0),
        components=components,
        scores=scores,
        explained_variance_ratio=ratio,
    )


def _smooth_histogram(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0:
        return values
    radius = max(1, int(np.ceil(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def _matplotlib_pyplot():
    """Return a non-interactive pyplot with editable publication SVG text."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def plot_latent_pca_scatter(
    pca_scores: np.ndarray,
    explained_variance_ratio: np.ndarray,
    output_path: str,
    title: str = "",
    max_points: int = 100_000,
    bins: int = 60,
    alpha: float = 0.08,
    selected_indices: np.ndarray | None = None,
    selected_labels: list[str] | None = None,
) -> bool:
    if pca_scores.ndim != 2 or pca_scores.shape[1] < 2:
        return False
    plt = _matplotlib_pyplot()

    x, y = pca_scores[:, 0], pca_scores[:, 1]
    max_points = int(max_points or 0)
    if max_points > 0 and len(x) > max_points:
        keep = np.random.default_rng(42).choice(len(x), max_points, replace=False)
        xs, ys = x[keep], y[keep]
    else:
        xs, ys = x, y
    fig = plt.figure(figsize=(5.2, 5.0))
    grid = fig.add_gridspec(4, 4, left=0.15, right=0.92, bottom=0.14, top=0.92,
                            hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(grid[1:, :3])
    ax_x = fig.add_subplot(grid[0, :3], sharex=ax)
    ax_y = fig.add_subplot(grid[1:, 3], sharey=ax)
    ax.scatter(xs, ys, s=1.5, alpha=float(alpha), linewidths=0, rasterized=True)
    if selected_indices is not None:
        selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        valid = (selected_indices >= 0) & (selected_indices < len(x))
        selected_indices = selected_indices[valid]
        labels = selected_labels or [f"C{i:03d}" for i in range(len(valid))]
        labels = [label for label, keep in zip(labels, valid) if keep]
        selected_x = x[selected_indices]
        selected_y = y[selected_indices]
        ax.scatter(
            selected_x,
            selected_y,
            s=42,
            facecolors="white",
            edgecolors="#d62728",
            linewidths=1.2,
            zorder=5,
            label="selected conformations",
        )
        for label, px, py in zip(labels, selected_x, selected_y):
            ax.annotate(
                label,
                xy=(px, py),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                fontweight="bold",
                color="#b22222",
                zorder=6,
            )
        if len(selected_indices):
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                fontsize=7,
                frameon=False,
            )
    ax_x.hist(x, bins=int(bins), density=True, alpha=0.45)
    ax_y.hist(y, bins=int(bins), density=True, orientation="horizontal", alpha=0.45)
    ev1 = float(explained_variance_ratio[0]) if len(explained_variance_ratio) else 0.0
    ev2 = float(explained_variance_ratio[1]) if len(explained_variance_ratio) > 1 else 0.0
    ax.set_xlabel(f"PC1 (EV, {ev1:.2f})")
    ax.set_ylabel(f"PC2 (EV, {ev2:.2f})")
    ax.text(0.03, 0.04, f"N = {len(x):,}", transform=ax.transAxes)
    if title:
        ax.set_title(title, pad=8)
    ax_x.tick_params(axis="x", labelbottom=False)
    ax_x.tick_params(axis="y", left=False, labelleft=False)
    ax_y.tick_params(axis="y", labelleft=False)
    ax_y.tick_params(axis="x", bottom=False, labelbottom=False)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def compute_latent_umap(
    latents: np.ndarray,
    n_neighbors: int = 50,
    min_dist: float = 0.08,
    metric: str = "euclidean",
    random_state: int | None = 42,
    fit_max_points: int = 0,
) -> dict:
    """Compute a 2D UMAP embedding for particle latents."""
    try:
        import umap
    except Exception as exc:
        raise RuntimeError(
            "UMAP export requires the umap-learn package in the training environment."
        ) from exc

    z = np.asarray(latents, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError(f"Expected latents shaped (N,z_dim), got {z.shape}.")

    n_neighbors = max(2, min(int(n_neighbors), z.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=random_state,
        low_memory=True,
        verbose=False,
    )

    fit_max_points = int(fit_max_points or 0)
    if 0 < fit_max_points < z.shape[0]:
        rng = np.random.default_rng(random_state)
        fit_indices = np.sort(rng.choice(z.shape[0], fit_max_points, replace=False))
        reducer.fit(z[fit_indices])
        embedding = reducer.transform(z)
    else:
        fit_indices = None
        embedding = reducer.fit_transform(z)

    return dict(
        embedding=np.asarray(embedding, dtype=np.float32),
        fit_indices=fit_indices,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=random_state,
    )


def select_umap_representatives(
    embedding: np.ndarray,
    num_representatives: int,
    random_state: int | None = 42,
) -> dict[str, np.ndarray]:
    """Select real particles nearest to k-means centers in 2D UMAP space."""
    coords = np.asarray(embedding, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected a two-dimensional UMAP embedding, got {coords.shape}.")
    requested = int(num_representatives)
    if requested <= 0:
        raise ValueError("eval.umap_num_representatives must be positive.")
    count = max(10, requested)
    if len(coords) < count:
        raise ValueError(
            f"Cannot select {count} UMAP representatives from only {len(coords)} particles."
        )

    estimator = KMeans(
        n_clusters=count,
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    raw_cluster_labels = estimator.fit_predict(coords)
    raw_centers = np.asarray(estimator.cluster_centers_, dtype=np.float32)

    # Stable left-to-right ordering makes U000/U001/... reproducible and easy to
    # compare between export epochs without assigning biological semantics.
    order = np.lexsort((raw_centers[:, 1], raw_centers[:, 0]))
    remap = np.empty(count, dtype=np.int64)
    remap[order] = np.arange(count, dtype=np.int64)
    cluster_labels = remap[raw_cluster_labels]
    centers = raw_centers[order]

    selected_indices = np.empty(count, dtype=np.int64)
    cluster_sizes = np.empty(count, dtype=np.int64)
    for cluster_index, center in enumerate(centers):
        members = np.flatnonzero(cluster_labels == cluster_index)
        if not len(members):
            raise RuntimeError(f"UMAP cluster {cluster_index} unexpectedly has no particles.")
        distance2 = np.square(coords[members] - center[None]).sum(axis=1)
        selected_indices[cluster_index] = members[int(np.argmin(distance2))]
        cluster_sizes[cluster_index] = len(members)

    return {
        "selected_indices": selected_indices,
        "cluster_labels": cluster_labels,
        "cluster_centers": centers,
        "cluster_sizes": cluster_sizes,
    }


def plot_latent_umap_scatter(
    umap_embedding: np.ndarray,
    output_path: str,
    title: str = "",
    max_points: int = 100_000,
    bins: int = 60,
    alpha: float = 0.08,
    point_size: float = 0.45,
    selected_indices: np.ndarray | None = None,
    selected_labels: list[str] | None = None,
) -> bool:
    coords = np.asarray(umap_embedding)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return False

    plt = _matplotlib_pyplot()

    x, y = coords[:, 0], coords[:, 1]
    max_points = int(max_points or 0)
    if max_points > 0 and len(x) > max_points:
        keep = np.random.default_rng(42).choice(len(x), max_points, replace=False)
        xs, ys = x[keep], y[keep]
    else:
        xs, ys = x, y
    fig = plt.figure(figsize=(5.2, 5.0))
    grid = fig.add_gridspec(4, 4, left=0.15, right=0.92, bottom=0.14, top=0.92,
                            hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(grid[1:, :3])
    ax_x = fig.add_subplot(grid[0, :3], sharex=ax)
    ax_y = fig.add_subplot(grid[1:, 3], sharey=ax)
    ax.scatter(
        xs,
        ys,
        s=float(point_size),
        alpha=float(alpha),
        linewidths=0,
        rasterized=True,
    )
    if selected_indices is not None:
        selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        valid = (selected_indices >= 0) & (selected_indices < len(x))
        selected_indices = selected_indices[valid]
        labels = selected_labels or [f"C{i:03d}" for i in range(len(valid))]
        labels = [label for label, keep in zip(labels, valid) if keep]
        selected_x = x[selected_indices]
        selected_y = y[selected_indices]
        ax.scatter(
            selected_x,
            selected_y,
            s=42,
            facecolors="white",
            edgecolors="#d62728",
            linewidths=1.2,
            zorder=5,
            label="selected conformations",
        )
        for label, px, py in zip(labels, selected_x, selected_y):
            ax.annotate(
                label,
                xy=(px, py),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                fontweight="bold",
                color="#b22222",
                zorder=6,
            )
        if len(selected_indices):
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                fontsize=7,
                frameon=False,
            )
    ax_x.hist(x, bins=int(bins), density=True, alpha=0.45)
    ax_y.hist(y, bins=int(bins), density=True, orientation="horizontal", alpha=0.45)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.text(0.03, 0.04, f"N = {len(x):,}", transform=ax.transAxes)
    if title:
        ax.set_title(title, pad=8)
    ax_x.tick_params(axis="x", labelbottom=False)
    ax_x.tick_params(axis="y", left=False, labelleft=False)
    ax_y.tick_params(axis="y", labelleft=False)
    ax_y.tick_params(axis="x", bottom=False, labelbottom=False)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def _plain_dict(mapping: Any) -> dict:
    if mapping is None:
        return {}
    if isinstance(mapping, dict):
        return dict(mapping)
    if hasattr(mapping, "to_dict"):
        return dict(mapping.to_dict())
    if hasattr(mapping, "keys"):
        return {key: mapping[key] for key in mapping.keys()}
    return {}


def _label_key(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            return text
    try:
        as_float = float(value)
    except Exception:
        return value
    if np.isfinite(as_float) and as_float.is_integer():
        return int(as_float)
    return as_float


def _label_name(value: Any, names: dict | None = None) -> str:
    key = _label_key(value)
    mapping = dict(DEFAULT_CLASS_LABEL_NAMES)
    mapping.update(_plain_dict(names))
    if key in mapping:
        return str(mapping[key])
    if str(key) in mapping:
        return str(mapping[str(key)])
    return str(key)


def _label_color(value: Any, colors: dict | None = None) -> str:
    key = _label_key(value)
    mapping = dict(DEFAULT_CLASS_LABEL_COLORS)
    mapping.update(_plain_dict(colors))
    if key in mapping:
        return str(mapping[key])
    if str(key) in mapping:
        return str(mapping[str(key)])
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    stable_index = zlib.crc32(str(key).encode("utf-8")) % len(palette)
    return palette[stable_index]


def _sanitize_stem(text: str) -> str:
    safe = []
    for char in str(text):
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        elif char in (" ", ".", "/", "\\", "(", ")"):
            safe.append("_")
    stem = "".join(safe).strip("_")
    return stem or "class"


def _label_sort_key(value: Any) -> tuple[int, Any]:
    key = _label_key(value)
    if isinstance(key, int):
        return (0, key)
    if isinstance(key, float):
        return (1, key)
    return (2, str(key))


def _parse_label_values(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",") if part.strip()]
    parsed = []
    for value in values:
        parsed.append(_label_key(value))
    return parsed


def load_class_label_vector(path: str, column: str | None = None) -> np.ndarray:
    """Load per-particle labels from pkl/npy/txt/csv without touching training."""
    path = osp.expanduser(str(path))
    if not osp.exists(path):
        raise FileNotFoundError(path)
    suffix = osp.splitext(path)[1].lower()
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            labels = pickle.load(handle)
        if isinstance(labels, dict):
            for key in ("labels", "label", "class", "classes", "y"):
                if key in labels:
                    labels = labels[key]
                    break
        return np.asarray(labels).reshape(-1)
    if suffix == ".npy":
        return np.load(path, allow_pickle=True).reshape(-1)
    if suffix in {".txt", ".dat"}:
        return np.loadtxt(path, dtype=str).reshape(-1)
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV label file has no header: {path}")
            if column is None:
                candidates = ("label", "class", "state", "class_id", "particle_label")
                column = next((name for name in candidates if name in reader.fieldnames), None)
            if column is None or column not in reader.fieldnames:
                raise ValueError(
                    f"Set eval.class_label_column for {path}. Available columns: {reader.fieldnames}"
                )
            return np.asarray([row[column] for row in reader]).reshape(-1)
    raise ValueError(f"Unsupported class label file extension: {path}")


def labels_for_dataset(labels: np.ndarray, dataset: StarDataset) -> np.ndarray:
    """Align full-dataset or selected-subset labels to the current dataset order."""
    labels = np.asarray(labels).reshape(-1)
    if len(labels) == len(dataset):
        return labels
    total = int(getattr(dataset, "n_particles_total", len(dataset)))
    if len(labels) == total:
        orig_indices = dataset.df["__orig_index"].to_numpy(dtype=np.int64)
        return labels[orig_indices]
    raise ValueError(
        f"Label vector length {len(labels)} does not match current particles {len(dataset)} "
        f"or full STAR particles {total}."
    )


def plot_latent_umap_by_label(
    umap_embedding: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "",
    class_order: list[Any] | None = None,
    class_names: dict | None = None,
    class_colors: dict | None = None,
    max_points: int = 131_899,
    alpha: float = 0.18,
    point_size: float = 0.45,
    selected_indices: np.ndarray | None = None,
    selected_labels: list[str] | None = None,
) -> bool:
    coords = np.asarray(umap_embedding)
    labels = np.asarray(labels).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(labels) != len(coords):
        return False

    plt = _matplotlib_pyplot()
    from matplotlib.patches import Patch

    if class_order is None:
        class_order = sorted({_label_key(value) for value in labels}, key=_label_sort_key)
    max_points = int(max_points or 0)
    if max_points > 0 and len(coords) > max_points:
        keep = np.random.default_rng(42).choice(len(coords), max_points, replace=False)
    else:
        keep = np.arange(len(coords), dtype=np.int64)
    keep_labels = np.asarray([_label_key(labels[index]) for index in keep], dtype=object)

    fig, ax = plt.subplots(figsize=(4.1, 3.8))
    legend_handles = []
    legend_labels = []
    for label_value in class_order:
        key = _label_key(label_value)
        mask = keep_labels == key
        if not np.any(mask):
            continue
        pts = coords[keep[mask]]
        color = _label_color(key, class_colors)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=float(point_size),
            alpha=float(alpha),
            linewidths=0,
            color=color,
            rasterized=True,
        )
        legend_handles.append(Patch(facecolor=color, edgecolor="none"))
        legend_labels.append(_label_name(key, class_names))
    if selected_indices is not None:
        selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        valid = (selected_indices >= 0) & (selected_indices < len(coords))
        selected_indices = selected_indices[valid]
        labels_to_show = selected_labels or [str(i) for i in range(len(valid))]
        labels_to_show = [label for label, keep_value in zip(labels_to_show, valid) if keep_value]
        pts = coords[selected_indices]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=30,
            color="black",
            linewidths=0,
            zorder=5,
        )
        for label, px, py in zip(labels_to_show, pts[:, 0], pts[:, 1]):
            ax.annotate(label, xy=(px, py), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    if title:
        ax.set_title(title)
    legend = ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        frameon=True,
        facecolor="white",
        edgecolor="#333333",
        framealpha=1.0,
        fancybox=False,
        handlelength=1.05,
        handleheight=0.8,
        borderpad=0.45,
        labelspacing=0.45,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    legend.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_latent_pc1_distribution(
    pc1_scores: np.ndarray,
    explained_variance_ratio: np.ndarray,
    output_path: str,
    title: str = "",
    z_norm_mean: float | None = None,
    bins: int = 80,
    smooth_sigma_bins: float = 1.5,
) -> None:
    plt = _matplotlib_pyplot()

    values = np.asarray(pc1_scores, dtype=np.float64)
    density, edges = np.histogram(values, bins=int(bins), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.hist(values, bins=int(bins), density=True, alpha=0.45)
    ax.plot(centers, _smooth_histogram(density, smooth_sigma_bins), linewidth=1.8)
    ev1 = float(explained_variance_ratio[0]) if len(explained_variance_ratio) else 0.0
    ax.set_xlabel(f"PC1 (EV, {ev1:.2f})")
    ax.set_ylabel("Particle density")
    if title:
        ax.set_title(title)
    annotation = f"N = {values.size:,}"
    if z_norm_mean is not None and np.isfinite(z_norm_mean):
        annotation += f"\nmean ||z|| = {z_norm_mean:.2f}"
    ax.text(0.04, 0.93, annotation, transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


class ResidualLinearBlock(torch.nn.Module):
    """Linear residual layer used by cryoDRGN's ResidLinearMLP."""

    def __init__(self, width: int):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x)


class CryoDRGNVAEEncoder(torch.nn.Module):
    """Encode a scalar Hartley image into diagonal-Gaussian posterior parameters."""

    def __init__(
        self,
        image_size: int,
        z_dim: int,
        hidden: int = 256,
        layers: int = 3,
        encoder_type: str = "resid",
        mask_radius: float = 1.0,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.z_dim = int(z_dim)
        self.encoder_type = str(encoder_type).lower()
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        if self.logvar_min >= self.logvar_max:
            raise ValueError("vae.logvar_min must be smaller than vae.logvar_max.")

        yy = torch.linspace(-1.0, 1.0, self.image_size)[:, None]
        xx = torch.linspace(-1.0, 1.0, self.image_size)[None, :]
        radius = float(mask_radius)
        if radius <= 0:
            mask = torch.ones(self.image_size, self.image_size, dtype=torch.bool)
        else:
            mask = (xx.square() + yy.square()) <= radius * radius
        self.register_buffer("encoder_mask", mask, persistent=True)

        hidden = int(hidden)
        layers = max(1, int(layers))
        if self.encoder_type == "resid":
            in_dim = int(mask.sum().item())
            blocks: list[torch.nn.Module] = [
                torch.nn.Linear(in_dim, hidden),
                torch.nn.ReLU(inplace=True),
            ]
            for _ in range(layers):
                blocks.extend(
                    [ResidualLinearBlock(hidden), torch.nn.ReLU(inplace=True)]
                )
            blocks.append(torch.nn.Linear(hidden, 2 * self.z_dim))
            self.network = torch.nn.Sequential(*blocks)
        elif self.encoder_type == "conv":
            channels = max(16, hidden // 8)
            conv: list[torch.nn.Module] = []
            in_channels = 1
            for multiplier in (1, 2, 4, 8):
                out_channels = channels * multiplier
                conv.extend(
                    [
                        torch.nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                        torch.nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
                in_channels = out_channels
            conv.extend(
                [
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(in_channels, 2 * self.z_dim),
                ]
            )
            self.network = torch.nn.Sequential(*conv)
        else:
            raise ValueError("vae.encoder_type must be 'resid' or 'conv'.")

    def forward(self, hartley_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hartley_image.ndim == 4:
            if hartley_image.shape[1] != 1:
                raise ValueError(f"Expected one Hartley channel, got {hartley_image.shape}.")
            hartley_image = hartley_image[:, 0]
        if hartley_image.ndim != 3:
            raise ValueError(f"Expected Hartley images shaped (B,H,W), got {hartley_image.shape}.")

        # Zeroing outside the mask is useful for both encoders.  The linear encoder
        # then drops those pixels, exactly like cryoDRGN's enc_mask path.
        masked = hartley_image * self.encoder_mask.to(dtype=hartley_image.dtype)
        if self.encoder_type == "resid":
            features = masked[:, self.encoder_mask]
        else:
            features = masked[:, None]

        posterior = self.network(features)
        z_mu, z_logvar = posterior.chunk(2, dim=-1)
        z_logvar = z_logvar.clamp(self.logvar_min, self.logvar_max)
        return z_mu, z_logvar

    @staticmethod
    def reparameterize(z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * z_logvar)
        return z_mu + torch.randn_like(std) * std


class GaussianParameters(torch.nn.Module):
    """Canonical learnable Gaussian cloud.

    ``density`` is the value at a Gaussian's center. ``amplitude`` is its
    integral over R^3.  The selected quantity is optimized directly; the other
    is derived from the current scale, so amplitude mode preserves integrated
    mass when a Gaussian changes size.
    """

    def __init__(
        self,
        num_points: int,
        scale_min: float,
        scale_max: float,
        init_scale: float,
        parameterization: str,
        init_value: float,
        xyz_init_std: float,
    ):
        super().__init__()
        self.num_points = int(num_points)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        if not (0.0 < self.scale_min < self.scale_max):
            raise ValueError("Require 0 < scale_min < scale_max.")
        if not (self.scale_min < float(init_scale) < self.scale_max):
            raise ValueError("model.init_scale must lie inside (scale_min, scale_max).")
        self.parameterization = str(parameterization).lower()
        if self.parameterization not in {"density", "amplitude"}:
            raise ValueError(
                "model.gaussian_parameterization must be 'density' or 'amplitude', "
                f"got {parameterization!r}."
            )
        if float(init_value) <= 0:
            field = f"model.init_{self.parameterization}"
            raise ValueError(f"{field} must be positive.")

        self.xyz = torch.nn.Parameter(torch.randn(self.num_points, 3) * float(xyz_init_std))
        init_ratio = torch.full(
            (self.num_points, 3),
            (float(init_scale) - self.scale_min) / (self.scale_max - self.scale_min),
        )
        self.scale_raw = torch.nn.Parameter(inverse_sigmoid(init_ratio))
        self.rotation_raw = torch.nn.Parameter(torch.zeros(self.num_points, 4))
        self.rotation_raw.data[:, 0] = 1.0
        strength_value = torch.full((self.num_points, 1), float(init_value))
        self.strength_raw = torch.nn.Parameter(inverse_softplus(strength_value))

    @property
    def scaling(self) -> torch.Tensor:
        return torch.sigmoid(self.scale_raw) * (self.scale_max - self.scale_min) + self.scale_min

    @property
    def rotation(self) -> torch.Tensor:
        return self.rotation_raw / self.rotation_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @property
    def strength(self) -> torch.Tensor:
        """Positive value optimized in the configured parameterization."""
        return torch.nn.functional.softplus(self.strength_raw)

    @property
    def gaussian_integral_factor(self) -> torch.Tensor:
        """Integral of an unnormalized 3D Gaussian with the current scales."""
        return (2.0 * torch.pi) ** 1.5 * self.scaling.prod(dim=-1, keepdim=True)

    @property
    def density(self) -> torch.Tensor:
        """Peak density expected by the xray rasterizer and voxelizer."""
        if self.parameterization == "density":
            return self.strength
        return self.strength / self.gaussian_integral_factor

    @property
    def amplitude(self) -> torch.Tensor:
        """Integrated density (mass) of each normalized 3D Gaussian."""
        if self.parameterization == "amplitude":
            return self.strength
        return self.strength * self.gaussian_integral_factor


class DeformationMLP(torch.nn.Module):
    """Per-particle field: [PE(xyz), z] -> Gaussian attribute deltas."""

    def __init__(
        self,
        z_dim: int,
        hidden: int,
        layers: int,
        pe_freqs: int = 0,
        pe_include_xyz: bool = True,
        output_dim: int = 3,
    ):
        super().__init__()
        if z_dim <= 0:
            raise ValueError("VAE heterogeneous reconstruction requires model.z_dim > 0.")
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError("DeformationMLP output_dim must be positive.")
        self.pe_freqs = int(pe_freqs)
        self.pe_include_xyz = bool(pe_include_xyz)
        if self.pe_freqs < 0:
            raise ValueError("model.deform_pe_freqs must be non-negative.")
        frequency_bands = torch.pi * torch.pow(2.0, torch.arange(self.pe_freqs))
        self.register_buffer("frequency_bands", frequency_bands, persistent=True)

        # pe_freqs=0 is exactly the original raw-xyz deformation network.
        if self.pe_freqs == 0:
            position_dim = 3
        else:
            position_dim = 6 * self.pe_freqs + (3 if self.pe_include_xyz else 0)
        self.position_feature_dim = position_dim

        modules: list[torch.nn.Module] = []
        in_dim = self.position_feature_dim + int(z_dim)
        self.input_dim = in_dim
        for _ in range(max(1, int(layers) - 1)):
            modules.extend([torch.nn.Linear(in_dim, int(hidden)), torch.nn.ReLU(inplace=True)])
            in_dim = int(hidden)
        modules.append(torch.nn.Linear(in_dim, self.output_dim))
        self.net = torch.nn.Sequential(*modules)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def encode_position(self, xyz: torch.Tensor) -> torch.Tensor:
        if self.pe_freqs == 0:
            return xyz
        angles = xyz[..., None] * self.frequency_bands.to(dtype=xyz.dtype)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        fourier = fourier.reshape(*xyz.shape[:-1], -1)
        return torch.cat([xyz, fourier], dim=-1) if self.pe_include_xyz else fourier

    def forward(self, xyz: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            z = z[None]
        if z.shape[0] != 1:
            raise ValueError(f"DeformationMLP expects one latent, got {tuple(z.shape)}.")
        z = z.expand(xyz.shape[0], -1)
        return self.net(torch.cat([self.encode_position(xyz), z], dim=-1))


class FactorizedAttributeDecoder(torch.nn.Module):
    """Separate xyz residuals from low-rank state-dependent occupancy logits.

    The occupancy branch implements

        logit(o_i(z)) = b0 + b_i(x_i) + a_i(x_i)^T h(z) / sqrt(K),

    so independently learned compositional factors can be combined at one latent
    state instead of competing in a single joint output head.
    """

    def __init__(
        self,
        z_dim: int,
        hidden: int,
        layers: int,
        pe_freqs: int,
        pe_include_xyz: bool,
        factor_dim: int,
        factor_hidden: int,
        factor_layers: int,
        initial_occupancy_logit: float,
        factor_tanh: bool,
    ):
        super().__init__()
        self.position_head = DeformationMLP(
            z_dim=z_dim,
            hidden=hidden,
            layers=layers,
            pe_freqs=pe_freqs,
            pe_include_xyz=pe_include_xyz,
            output_dim=3,
        )
        self.pe_freqs = self.position_head.pe_freqs
        self.input_dim = self.position_head.input_dim
        self.output_dim = 4
        self.factor_dim = int(factor_dim)
        self.factor_tanh = bool(factor_tanh)
        if self.factor_dim <= 0:
            raise ValueError("model.occupancy_factor_dim must be positive.")
        if int(factor_hidden) <= 0:
            raise ValueError("model.occupancy_factor_hidden must be positive.")

        spatial_modules: list[torch.nn.Module] = []
        in_dim = self.position_head.position_feature_dim
        for _ in range(max(1, int(layers) - 1)):
            spatial_modules.extend(
                [torch.nn.Linear(in_dim, int(factor_hidden)), torch.nn.ReLU(inplace=True)]
            )
            in_dim = int(factor_hidden)
        spatial_modules.append(torch.nn.Linear(in_dim, self.factor_dim + 1))
        self.spatial_factors = torch.nn.Sequential(*spatial_modules)

        latent_modules: list[torch.nn.Module] = []
        in_dim = int(z_dim)
        for _ in range(max(1, int(factor_layers) - 1)):
            latent_modules.extend(
                [torch.nn.Linear(in_dim, int(factor_hidden)), torch.nn.ReLU(inplace=True)]
            )
            in_dim = int(factor_hidden)
        latent_modules.append(torch.nn.Linear(in_dim, self.factor_dim))
        self.latent_factors = torch.nn.Sequential(*latent_modules)

        # Preserve the exact occupancy initialization. Zero spatial loadings let
        # the latent factors start learning without perturbing the shared scaffold.
        torch.nn.init.zeros_(self.spatial_factors[-1].weight)
        torch.nn.init.zeros_(self.spatial_factors[-1].bias)
        self.register_buffer(
            "base_occupancy_logit",
            torch.tensor(float(initial_occupancy_logit)),
            persistent=True,
        )

    def forward(self, xyz: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        displacement = self.position_head(xyz, z)
        if z.ndim == 1:
            z = z[None]
        if z.shape[0] != 1:
            raise ValueError(
                f"FactorizedAttributeDecoder expects one latent, got {tuple(z.shape)}."
            )
        factors = self.latent_factors(z)[0]
        if self.factor_tanh:
            factors = torch.tanh(factors)
        spatial = self.spatial_factors(self.position_head.encode_position(xyz))
        loadings, bias = spatial[:, : self.factor_dim], spatial[:, self.factor_dim :]
        logits = bias + (
            (loadings * factors[None]).sum(dim=-1, keepdim=True)
            / math.sqrt(float(self.factor_dim))
        )
        logits = logits + self.base_occupancy_logit.to(dtype=logits.dtype)
        return torch.cat([displacement, logits], dim=-1)


class GaussianVolume(torch.nn.Module):
    """Projection and voxelization of canonical or VAE-deformed Gaussians."""

    def __init__(self, model_cfg, image_size: int):
        super().__init__()
        self.image_size = int(image_size)
        self.camera_z = float(cfg_get(model_cfg, "camera_z", 5.0))
        self.renderer_gain = float(cfg_get(model_cfg, "renderer_gain", 1.0))
        if not math.isfinite(self.renderer_gain) or self.renderer_gain <= 0.0:
            raise ValueError("model.renderer_gain must be a finite positive number.")
        self.hetero = bool(cfg_get(model_cfg, "hetero", True))
        self.deform_scale = float(cfg_get(model_cfg, "deform_scale", 0.03))
        self.deform_tanh = bool(cfg_get(model_cfg, "deform_tanh", True))
        self.deform_attributes = normalize_deform_attributes(
            cfg_get(model_cfg, "deform_attributes", "xyz")
        )
        self.deform_decoder = str(
            cfg_get(model_cfg, "deform_decoder", "factorized")
        ).strip().lower()
        if self.deform_decoder not in {"joint", "factorized"}:
            raise ValueError("model.deform_decoder must be 'joint' or 'factorized'.")
        if self.deform_decoder == "factorized" and self.deform_attributes != "xyz_occupancy":
            raise ValueError(
                "The factorized decoder currently requires "
                "model.deform_attributes='xyz_occupancy'."
            )
        self.deform_output_dim = {"xyz": 3, "xyz_occupancy": 4, "all": 11}[self.deform_attributes]
        self.occupancy_init = float(cfg_get(model_cfg, "occupancy_init", 0.95))
        self.occupancy_temperature = float(cfg_get(model_cfg, "occupancy_temperature", 1.0))
        if not 0.0 < self.occupancy_init < 1.0:
            raise ValueError("model.occupancy_init must lie strictly inside (0, 1).")
        if not math.isfinite(self.occupancy_temperature) or self.occupancy_temperature <= 0.0:
            raise ValueError("model.occupancy_temperature must be a finite positive number.")
        raw_attribute_scale = float(cfg_get(model_cfg, "deform_raw_attribute_scale", 1.0))
        self.deform_scale_raw_scale = float(
            cfg_get(model_cfg, "deform_scale_raw_scale", raw_attribute_scale)
        )
        self.deform_rotation_raw_scale = float(
            cfg_get(model_cfg, "deform_rotation_raw_scale", raw_attribute_scale)
        )
        self.deform_strength_raw_scale = float(
            cfg_get(model_cfg, "deform_strength_raw_scale", raw_attribute_scale)
        )
        for name, value in (
            ("model.deform_scale_raw_scale", self.deform_scale_raw_scale),
            ("model.deform_rotation_raw_scale", self.deform_rotation_raw_scale),
            ("model.deform_strength_raw_scale", self.deform_strength_raw_scale),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number.")
        parameterization = str(
            cfg_get(model_cfg, "gaussian_parameterization", "density")
        ).lower()
        num_points = int(model_cfg.num_points)
        init_scale = float(cfg_get(model_cfg, "init_scale", 0.012))
        init_density = float(cfg_get(model_cfg, "init_density", 9e-4))
        if parameterization == "density":
            init_value = init_density
        elif parameterization == "amplitude":
            init_amplitude = cfg_get(model_cfg, "init_amplitude", None)
            if init_amplitude is None:
                # Match the initial peak density of density mode.  CryoSplat uses
                # 0.5/P with its own renderer and data normalization; copying that
                # absolute value here makes this xray renderer start about two
                # orders of magnitude too strong for the existing VAE configs.
                init_integral_factor = (2.0 * torch.pi) ** 1.5 * init_scale**3
                init_value = init_density * init_integral_factor
            else:
                init_value = float(init_amplitude)
        else:
            raise ValueError(
                "model.gaussian_parameterization must be 'density' or 'amplitude', "
                f"got {parameterization!r}."
            )
        self.gaussians = GaussianParameters(
            num_points=num_points,
            scale_min=float(cfg_get(model_cfg, "scale_min", 0.001)),
            scale_max=float(cfg_get(model_cfg, "scale_max", 0.30)),
            init_scale=init_scale,
            parameterization=parameterization,
            init_value=init_value,
            xyz_init_std=float(cfg_get(model_cfg, "xyz_init_std", 0.25)),
        )
        if self.hetero:
            z_dim = int(model_cfg.z_dim)
            hidden = int(cfg_get(model_cfg, "deform_hidden", 64))
            layers = int(cfg_get(model_cfg, "deform_layers", 3))
            pe_freqs = int(cfg_get(model_cfg, "deform_pe_freqs", 0))
            pe_include_xyz = bool(cfg_get(model_cfg, "deform_pe_include_xyz", True))
            if self.deform_decoder == "factorized":
                initial_logit = float(
                    inverse_sigmoid(torch.tensor(self.occupancy_init)).item()
                ) * self.occupancy_temperature
                self.deform: torch.nn.Module | None = FactorizedAttributeDecoder(
                    z_dim=z_dim,
                    hidden=hidden,
                    layers=layers,
                    pe_freqs=pe_freqs,
                    pe_include_xyz=pe_include_xyz,
                    factor_dim=int(cfg_get(model_cfg, "occupancy_factor_dim", 16)),
                    factor_hidden=int(cfg_get(model_cfg, "occupancy_factor_hidden", 128)),
                    factor_layers=int(cfg_get(model_cfg, "occupancy_factor_layers", 2)),
                    initial_occupancy_logit=initial_logit,
                    factor_tanh=bool(cfg_get(model_cfg, "occupancy_factor_tanh", True)),
                )
            else:
                self.deform = DeformationMLP(
                    z_dim=z_dim,
                    hidden=hidden,
                    layers=layers,
                    pe_freqs=pe_freqs,
                    pe_include_xyz=pe_include_xyz,
                    output_dim=self.deform_output_dim,
                )
            if self.deform_attributes == "xyz_occupancy" and self.deform_decoder == "joint":
                # Start from an almost fully occupied shared scaffold while retaining
                # useful sigmoid gradients.  The network then learns state-dependent
                # occupancy residuals together with xyz displacements.
                assert isinstance(self.deform, DeformationMLP)
                final = self.deform.net[-1]
                assert isinstance(final, torch.nn.Linear) and final.bias is not None
                with torch.no_grad():
                    init_occ = torch.tensor(
                        self.occupancy_init, dtype=final.bias.dtype, device=final.bias.device
                    )
                    final.bias[3] = inverse_sigmoid(init_occ) * self.occupancy_temperature
        else:
            self.deform = None

    def _rasterizer(self, rotation: torch.Tensor) -> GaussianRasterizer:
        device = rotation.device
        viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
        viewmatrix[:3, :3] = rotation.mT
        viewmatrix[3, 2] = self.camera_z
        settings = GaussianRasterizationSettings(
            image_height=self.image_size,
            image_width=self.image_size,
            tanfovx=1.0,
            tanfovy=1.0,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=viewmatrix,
            campos=torch.tensor([0.0, 0.0, 1.0], device=device),
            prefiltered=False,
            mode=0,
            debug=False,
        )
        return GaussianRasterizer(raster_settings=settings)

    def _scaling_from_raw(self, scale_raw: torch.Tensor) -> torch.Tensor:
        g = self.gaussians
        return torch.sigmoid(scale_raw) * (g.scale_max - g.scale_min) + g.scale_min

    @staticmethod
    def _rotation_from_raw(rotation_raw: torch.Tensor) -> torch.Tensor:
        return rotation_raw / rotation_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _density_from_strength_and_scaling(
        self, strength: torch.Tensor, scaling: torch.Tensor
    ) -> torch.Tensor:
        if self.gaussians.parameterization == "density":
            return strength
        integral_factor = (2.0 * torch.pi) ** 1.5 * scaling.prod(dim=-1, keepdim=True)
        return strength / integral_factor

    def attributes(self, z: torch.Tensor | None) -> dict[str, torch.Tensor]:
        g = self.gaussians
        unit_occupancy = torch.ones_like(g.density)
        if not self.hetero:
            return dict(
                xyz=g.xyz,
                density=g.density,
                base_density=g.density,
                occupancy=unit_occupancy,
                scaling=g.scaling,
                rotation=g.rotation,
            )
        if z is None:
            raise ValueError("Heterogeneous GaussianVolume requires a latent z.")
        assert self.deform is not None
        deltas = self.deform(g.xyz, z)
        displacement = deltas[:, :3]
        if self.deform_tanh:
            displacement = torch.tanh(displacement)
        xyz = g.xyz + self.deform_scale * displacement

        if self.deform_attributes == "xyz":
            return dict(
                xyz=xyz,
                density=g.density,
                base_density=g.density,
                occupancy=unit_occupancy,
                scaling=g.scaling,
                rotation=g.rotation,
            )

        if self.deform_attributes == "xyz_occupancy":
            # rho_eff_i(z) = rho_i * o_i(z).  Base density/scale/rotation are
            # state invariant; xyz and occupancy are conditioned on the VAE latent.
            occupancy_logits = deltas[:, 3:4]
            occupancy = torch.sigmoid(occupancy_logits / self.occupancy_temperature)
            return dict(
                xyz=xyz,
                density=g.density * occupancy,
                base_density=g.density,
                occupancy=occupancy,
                scaling=g.scaling,
                rotation=g.rotation,
            )

        scale_delta = deltas[:, 3:6]
        rotation_delta = deltas[:, 6:10]
        strength_delta = deltas[:, 10:11]
        if self.deform_tanh:
            scale_delta = torch.tanh(scale_delta)
            rotation_delta = torch.tanh(rotation_delta)
            strength_delta = torch.tanh(strength_delta)

        scaling = self._scaling_from_raw(g.scale_raw + self.deform_scale_raw_scale * scale_delta)
        rotation = self._rotation_from_raw(
            g.rotation_raw + self.deform_rotation_raw_scale * rotation_delta
        )
        strength = torch.nn.functional.softplus(
            g.strength_raw + self.deform_strength_raw_scale * strength_delta
        )
        return dict(
            xyz=xyz,
            density=self._density_from_strength_and_scaling(strength, scaling),
            base_density=g.density,
            occupancy=unit_occupancy,
            scaling=scaling,
            rotation=rotation,
        )

    def positions(self, z: torch.Tensor | None) -> torch.Tensor:
        return self.attributes(z)["xyz"]

    def occupancy(self, z: torch.Tensor | None) -> torch.Tensor:
        return self.attributes(z)["occupancy"]

    def forward(self, z: torch.Tensor | None, rotations: torch.Tensor) -> torch.Tensor:
        if self.hetero and z is None:
            raise ValueError("Heterogeneous GaussianVolume requires batched latents.")
        # R2 discards projected contributions with alpha < 1e-5.  Scaling the
        # opacity before its additive X-ray rasterizer and dividing the image by
        # the same factor preserves the mathematical projection and gradient
        # scale, while lowering that effective cutoff to 1e-5 / renderer_gain.
        outputs = []
        for particle_idx in range(rotations.shape[0]):
            particle_z = None if not self.hetero else z[particle_idx]
            attrs = self.attributes(particle_z)
            xyz = attrs["xyz"]
            means2d = torch.zeros_like(xyz, requires_grad=True)
            image, _ = self._rasterizer(rotations[particle_idx])(
                means3D=xyz,
                means2D=means2d,
                opacities=attrs["density"] * self.renderer_gain,
                scales=attrs["scaling"],
                rotations=attrs["rotation"],
                cov3D_precomp=None,
            )
            image = image / self.renderer_gain
            if image.ndim == 3:
                image = image.unsqueeze(0)
            outputs.append(image)
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def voxelize(
        self,
        z: torch.Tensor | None,
        scale_modifier: float = 1.0,
    ) -> torch.Tensor:
        scale_modifier = float(scale_modifier)
        if not math.isfinite(scale_modifier) or scale_modifier <= 0.0:
            raise ValueError("scale_modifier must be a finite positive number.")
        size = self.image_size
        settings = GaussianVoxelizationSettings(
            scale_modifier=scale_modifier,
            nVoxel_x=size,
            nVoxel_y=size,
            nVoxel_z=size,
            sVoxel_x=2.0,
            sVoxel_y=2.0,
            sVoxel_z=2.0,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
            prefiltered=False,
            debug=False,
        )
        voxelizer = GaussianVoxelizer(voxel_settings=settings)
        attrs = self.attributes(z)
        volume, _ = voxelizer(
            means3D=attrs["xyz"],
            opacities=attrs["density"],
            scales=attrs["scaling"],
            rotations=attrs["rotation"],
            cov3D_precomp=None,
        )
        return volume.permute(2, 1, 0).contiguous()


class VAEGaussianRecon(pl.LightningModule):
    """Homo reconstruction or hetero reconstruction with amortized VAE latents."""

    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.hetero = bool(cfg_get(cfg.model, "hetero", True))
        self.z_dim = int(cfg_get(cfg.model, "z_dim", 0))
        if self.hetero and self.z_dim <= 0:
            raise ValueError("Set model.z_dim > 0 for VAE heterogeneous reconstruction.")
        self.volume = GaussianVolume(cfg.model, cfg.data_process.down_side_shape)

        if cfg.model.shift_method == "interp":
            self.translate = SpatialGridTranslate(cfg.data_process.down_side_shape)
            self.f_translate = None
        elif cfg.model.shift_method == "fft":
            self.translate = None
            self.f_translate = FourierGridTranslate(cfg.data_process.down_side_shape)
        else:
            raise ValueError(f"Unknown model.shift_method={cfg.model.shift_method!r}.")

        ctf_params = infer_ctf_params_from_config(cfg)
        if cfg.model.ctf == "v1":
            self.ctf = CTFRelion(**ctf_params, num_particles=len(dataset))
        elif cfg.model.ctf == "v2":
            self.ctf = CTFParticle(**ctf_params, num_particles=len(dataset))
        else:
            raise ValueError(f"Unknown model.ctf={cfg.model.ctf!r}.")

        mask = create_circular_mask(
            cfg.data_process.down_side_shape,
            cfg.data_process.down_side_shape,
            None,
            cfg.data_process.down_side_shape // 2 * float(cfg_get(cfg.loss, "mask_radius", 1.0)),
        )
        self.register_buffer("fourier_mask", torch.from_numpy(mask).bool())

        vae_cfg = getattr(cfg, "vae", None)
        self.vae_cfg = vae_cfg
        if self.hetero:
            self.phase_flip = bool(cfg_get(vae_cfg, "phase_flip", True))
            self.encoder: CryoDRGNVAEEncoder | None = CryoDRGNVAEEncoder(
                image_size=int(cfg.data_process.down_side_shape),
                z_dim=self.z_dim,
                hidden=int(cfg_get(vae_cfg, "encoder_hidden", 256)),
                layers=int(cfg_get(vae_cfg, "encoder_layers", 3)),
                encoder_type=str(cfg_get(vae_cfg, "encoder_type", "resid")),
                mask_radius=float(cfg_get(vae_cfg, "encoder_mask_radius", 1.0)),
                logvar_min=float(cfg_get(vae_cfg, "logvar_min", -10.0)),
                logvar_max=float(cfg_get(vae_cfg, "logvar_max", 10.0)),
            )
            beta_cfg = cfg_get(vae_cfg, "beta", None)
            self.beta = 1.0 / self.z_dim if beta_cfg is None else float(beta_cfg)
            if self.beta < 0:
                raise ValueError("vae.beta must be non-negative.")
            self.beta_warmup_epochs = max(
                0, int(cfg_get(vae_cfg, "beta_warmup_epochs", 5))
            )
        else:
            self.phase_flip = False
            self.encoder = None
            self.beta = 0.0
            self.beta_warmup_epochs = 0
        self._last_export_epoch = -1
        self._last_completed_epoch = 0
        self.occupancy_temperature_schedule = normalize_occupancy_temperature_schedule(
            cfg_get(cfg.model, "occupancy_temperature_schedule", None),
            self.volume.occupancy_temperature,
        )
        epoch_one_temperature = interpolate_epoch_schedule(
            self.occupancy_temperature_schedule, 1
        )
        if not math.isclose(
            epoch_one_temperature,
            self.volume.occupancy_temperature,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "The epoch-1 occupancy temperature must match "
                "model.occupancy_temperature so occupancy_init remains exact."
            )

        g = self.volume.gaussians
        log_to_current(
            f"[init] standalone {'VAE hetero' if self.hetero else 'homo'} Gaussian | "
            f"N={g.num_points} | xyz_std={g.xyz.detach().std().item():.5f} | "
            f"scale_mean={g.scaling.detach().mean().item():.6f} | "
            f"parameterization={g.parameterization} | "
            f"renderer_gain={self.volume.renderer_gain:g} | "
            f"density_mean={g.density.detach().mean().item():.6g} | "
            f"amplitude_mean={g.amplitude.detach().mean().item():.6g} | "
            f"amplitude_sum={g.amplitude.detach().sum().item():.6g}"
        )

        if self.hetero:
            assert self.volume.deform is not None and self.encoder is not None
            log_to_current(
                "[deform] "
                f"attributes={self.volume.deform_attributes} | "
                f"decoder={self.volume.deform_decoder} | "
                f"scale={self.volume.deform_scale:g} | "
                f"tanh={self.volume.deform_tanh} | "
                f"pe_freqs={self.volume.deform.pe_freqs} | "
                f"input_dim={self.volume.deform.input_dim} | "
                f"output_dim={self.volume.deform.output_dim}"
            )
            if isinstance(self.volume.deform, FactorizedAttributeDecoder):
                log_to_current(
                    "[factorized-occupancy] "
                    f"factors={self.volume.deform.factor_dim} | "
                    f"factor_tanh={self.volume.deform.factor_tanh} | "
                    f"parameters={sum(p.numel() for p in self.volume.deform.parameters()):,}"
                )
            if self.volume.deform_attributes == "all":
                log_to_current(
                    "[deform-attributes] "
                    f"scale_raw_scale={self.volume.deform_scale_raw_scale:g} | "
                    f"rotation_raw_scale={self.volume.deform_rotation_raw_scale:g} | "
                    f"strength_raw_scale={self.volume.deform_strength_raw_scale:g}"
                )
            if self.volume.deform_attributes == "xyz_occupancy":
                log_to_current(
                    "[occupancy] "
                    f"init={self.volume.occupancy_init:g} | "
                    f"temperature={self.volume.occupancy_temperature:g} | "
                    "schedule="
                    + ",".join(
                        f"{epoch}:{temperature:g}"
                        for epoch, temperature in self.occupancy_temperature_schedule
                    )
                )
            log_to_current(
                "[vae] "
                f"encoder={self.encoder.encoder_type} | z_dim={self.z_dim} | "
                f"encoder_pixels={int(self.encoder.encoder_mask.sum())} | "
                f"phase_flip={self.phase_flip} | beta={self.beta:g} | "
                f"warmup_epochs={self.beta_warmup_epochs}"
            )

    def _prepare_target(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the raw image and the RELION-shift-corrected training target."""
        if "shifted_target" in batch:
            target = batch["shifted_target"].to(self.device)
            image = batch["proj"].to(self.device) if "proj" in batch else target
            return image, target
        image = batch["proj"].to(self.device)
        batch_size = image.shape[0]
        shifts = torch.cat(
            [
                batch["shiftY"].float().reshape(batch_size, 1, 1) / self.cfg.data_process.down_apix,
                batch["shiftX"].float().reshape(batch_size, 1, 1) / self.cfg.data_process.down_apix,
            ],
            dim=2,
        ).to(self.device)
        if self.cfg.model.shift_method == "interp":
            assert self.translate is not None
            shifted = self.translate.transform(image.squeeze(1), shifts)
        else:
            assert self.f_translate is not None
            shifted_fourier = self.f_translate.transform(
                primal_to_fourier_2d(image).squeeze(1), shifts
            )
            # The inverse FFT has a complex dtype even when the shifted image is
            # physically real.  Discard its numerical imaginary residual before
            # feeding it to the encoder or a real-space loss; otherwise real_mse
            # becomes a complex scalar and autograd cannot create its gradient.
            shifted = fourier_to_primal_2d(shifted_fourier).real
        return image, shifted

    def _ctf_prediction(self, batch, latent: torch.Tensor | None) -> torch.Tensor:
        predicted_real = self.volume(latent, batch["rotmat"])
        predicted_fourier = primal_to_fourier_2d(predicted_real)
        return self.ctf(
            predicted_fourier,
            batch["idx"].long(),
            ctf_params=self._ctf_params(batch),
            mode="gt",
            frequency_marcher=None,
        )

    @torch.no_grad()
    def _export_volume(
        self,
        z: torch.Tensor | None,
        path: str,
        scale_modifier: float = 1.0,
    ) -> None:
        volume = (
            self.volume.voxelize(z, scale_modifier=scale_modifier)
            .detach()
            .cpu()
            .numpy()
        )
        pixel = float(self.cfg.data_process.down_apix)
        save_mrc(volume, path, pixel, -pixel * (volume.shape[0] // 2))

    @torch.no_grad()
    def _export_gaussian_state(self, z: torch.Tensor | None, path: str) -> None:
        """Save per-Gaussian state attributes for occupancy/displacement analysis."""
        attrs = self.volume.attributes(z)
        np.savez_compressed(
            path,
            xyz=attrs["xyz"].detach().float().cpu().numpy(),
            occupancy=attrs["occupancy"].detach().float().cpu().numpy(),
            density=attrs["density"].detach().float().cpu().numpy(),
            base_density=attrs["base_density"].detach().float().cpu().numpy(),
            scaling=attrs["scaling"].detach().float().cpu().numpy(),
            rotation=attrs["rotation"].detach().float().cpu().numpy(),
        )

    @torch.no_grad()
    def _export_fine_class_representatives(
        self,
        z_mu: np.ndarray,
        output_dir: str,
        eval_cfg: Any,
        dtype: torch.dtype,
        export_gaussian_states: bool,
    ) -> None:
        """Export one decoded state per post-hoc fine class without changing plots."""
        label_file = str(
            cfg_get(eval_cfg, "fine_class_label_file", "") or ""
        ).strip()
        if not label_file or not bool(
            cfg_get(eval_cfg, "export_fine_class_volumes", True)
        ):
            return

        raw_labels = load_class_label_vector(
            label_file,
            column=cfg_get(eval_cfg, "fine_class_label_column", None),
        )
        labels = labels_for_dataset(raw_labels, self.dataset)
        label_keys = np.asarray([_label_key(value) for value in labels], dtype=object)
        np.save(osp.join(output_dir, "particle_fine_class_labels.npy"), labels)

        class_order = _parse_label_values(
            cfg_get(eval_cfg, "fine_class_label_order", None)
        )
        if not class_order:
            class_order = sorted(set(label_keys.tolist()), key=_label_sort_key)
        class_names = _plain_dict(
            cfg_get(eval_cfg, "fine_class_label_names", None)
        )
        ignore_values = set(
            _parse_label_values(
                cfg_get(
                    eval_cfg,
                    "fine_class_label_ignore_values",
                    [-1, "nan", ""],
                )
            )
        )
        representative = str(
            cfg_get(eval_cfg, "fine_class_representative", "mean")
        ).strip().lower()
        if representative not in {"mean", "medoid"}:
            raise ValueError(
                "eval.fine_class_representative must be 'mean' or 'medoid', "
                f"got {representative!r}."
            )

        rows = []
        representative_latents = []
        representative_names = []
        for label_value in class_order:
            key = _label_key(label_value)
            if key in ignore_values:
                continue
            mask = label_keys == key
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue

            subset_indices = np.flatnonzero(mask)
            center_latent = z_mu[subset_indices].mean(axis=0)
            local_index = -1
            original_index = -1
            if representative == "medoid":
                distances = np.square(
                    z_mu[subset_indices] - center_latent[None]
                ).sum(axis=1)
                local_index = int(subset_indices[int(np.argmin(distances))])
                original_index = int(
                    self.dataset.df.iloc[local_index]["__orig_index"]
                )
                representative_latent = z_mu[local_index]
            else:
                representative_latent = center_latent

            class_name = _label_name(key, class_names)
            stem = _sanitize_stem(class_name)
            volume_file = f"vol_fine_class_{stem}.mrc"
            gaussian_file = f"gaussians_fine_class_{stem}.npz"
            z = torch.from_numpy(
                np.asarray(representative_latent, dtype=np.float32)
            ).to(self.device, dtype=dtype)
            self._export_volume(z, osp.join(output_dir, volume_file))
            export_fine_gaussians = export_gaussian_states and bool(
                cfg_get(eval_cfg, "export_fine_gaussian_states", False)
            )
            if export_fine_gaussians:
                self._export_gaussian_state(
                    z, osp.join(output_dir, gaussian_file)
                )
            else:
                gaussian_file = ""

            representative_latents.append(representative_latent)
            representative_names.append(class_name)
            rows.append(
                [
                    key,
                    class_name,
                    count,
                    representative,
                    local_index,
                    original_index,
                    volume_file,
                    gaussian_file,
                ]
                + [float(value) for value in representative_latent]
            )

        if representative_latents:
            np.savetxt(
                osp.join(output_dir, "z_fine_class_representatives.txt"),
                np.asarray(representative_latents, dtype=np.float32),
                fmt="%.7f",
            )
            with open(
                osp.join(output_dir, "fine_class_representative_names.txt"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("\n".join(representative_names) + "\n")

        with open(
            osp.join(output_dir, "fine_class_label_summary.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "fine_class_label",
                    "fine_class_name",
                    "particle_count",
                    "representative_mode",
                    "representative_local_particle_index",
                    "representative_original_star_index",
                    "volume_file",
                    "gaussian_file",
                ]
                + [f"z{i}" for i in range(self.z_dim)]
            )
            writer.writerows(rows)
        log_to_current(
            f"[export] fine-class representatives={len(rows)} "
            f"mode={representative} -> {output_dir}"
        )

    def _ctf_params(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            key: batch[key].to(self.device)
            for key in ("defocusU", "defocusV", "angleAstigmatism")
            if key in batch
        }

    def _encoder_input(self, centered_image: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        """Build the shifted, optionally phase-flipped Hartley encoder image."""
        image_fourier = primal_to_fourier_2d(centered_image)
        if self.phase_flip:
            # Applying the CTF to ones gives its real transfer function while reusing
            # exactly the same per-particle CTF convention as the reconstruction path.
            transfer = self.ctf(
                torch.ones_like(image_fourier),
                batch["idx"].to(self.device).long(),
                ctf_params=self._ctf_params(batch),
                mode="gt",
                frequency_marcher=None,
            ).real
            image_fourier = image_fourier * transfer.sign()
        return image_fourier.real - image_fourier.imag

    def encode_particles(
        self, centered_image: torch.Tensor, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.hetero or self.encoder is None:
            raise RuntimeError("The VAE encoder is unavailable in homogeneous mode.")
        return self.encoder(self._encoder_input(centered_image, batch))

    def _kl_scale(self) -> float:
        if not self.hetero:
            return 0.0
        if self.beta_warmup_epochs <= 0:
            warmup = 1.0
        else:
            warmup = min(1.0, float(self.current_epoch + 1) / self.beta_warmup_epochs)
        # cryoDRGN scales KL relative to the number of reconstruction pixels.
        n_pixels = max(1, int(self.fourier_mask.sum().item()))
        return self.beta * warmup / n_pixels

    def training_step(self, batch, batch_idx):
        _, target = self._prepare_target(batch)
        if self.hetero:
            z_mu, z_logvar = self.encode_particles(target, batch)
            assert self.encoder is not None
            latent: torch.Tensor | None = self.encoder.reparameterize(z_mu, z_logvar)
        else:
            z_mu = None
            z_logvar = None
            latent = None
        prediction_fourier = self._ctf_prediction(batch, latent)

        loss_type = str(cfg_get(self.cfg.loss, "type", "fourier_mse")).lower()
        unscaled_data_loss = None
        data_loss_scale = 1.0
        if loss_type == "real_mse":
            prediction = fourier_to_primal_2d(prediction_fourier).real
            unscaled_data_loss = (prediction - target).square().mean()
            scale_cfg = cfg_get(self.cfg.loss, "real_mse_scale", "parseval")
            if isinstance(scale_cfg, str):
                scale_name = scale_cfg.strip().lower()
                if scale_name in {"parseval", "pixels", "auto"}:
                    # torch.fft uses an unnormalized forward FFT.  Consequently,
                    # mean(|FFT(error)|^2) = H*W*mean(error^2).  Applying this
                    # factor prevents real-space gradients from being ~65k times
                    # smaller for 256x256 images while leaving the optimum intact.
                    data_loss_scale = float(prediction.shape[-2] * prediction.shape[-1])
                elif scale_name in {"none", "off", "unit"}:
                    data_loss_scale = 1.0
                else:
                    raise ValueError(
                        "loss.real_mse_scale must be 'parseval', 'none', or a "
                        f"positive number, got {scale_cfg!r}."
                    )
            else:
                data_loss_scale = float(scale_cfg)
                if data_loss_scale <= 0:
                    raise ValueError("loss.real_mse_scale must be positive.")
            data_loss = unscaled_data_loss * data_loss_scale
        elif loss_type == "fourier_mse":
            target_fourier = primal_to_fourier_2d(target)
            error2 = (target_fourier - prediction_fourier).abs().square()
            mask = self.fourier_mask[None, None].expand_as(error2)
            data_loss = error2.masked_select(mask).mean()
        else:
            raise ValueError("loss.type must be 'real_mse' or 'fourier_mse'.")

        if self.hetero:
            assert z_mu is not None and z_logvar is not None
            kld = -0.5 * torch.sum(
                1.0 + z_logvar - z_mu.square() - z_logvar.exp(), dim=-1
            ).mean()
            kl_scale = self._kl_scale()
            kl_loss = kld * kl_scale
        else:
            kld = data_loss.new_zeros(())
            kl_scale = 0.0
            kl_loss = data_loss.new_zeros(())
        loss = data_loss + kl_loss

        occupancy_off = data_loss.new_zeros(())
        occupancy_reg = data_loss.new_zeros(())
        occupancies = None
        occupancy_off_penalty = float(cfg_get(self.cfg.loss, "occupancy_off_penalty", 0.0))
        if (
            self.hetero
            and self.volume.deform_attributes == "xyz_occupancy"
            and occupancy_off_penalty > 0.0
        ):
            assert latent is not None
            # Mildly prefer the shared scaffold to stay occupied.  This breaks the
            # otherwise weak scale ambiguity rho_i * o_i(z) while still allowing
            # compositionally absent regions to switch off when supported by data.
            occupancies = torch.stack(
                [self.volume.occupancy(latent[i]) for i in range(latent.shape[0])], dim=0
            )
            occupancy_off = (1.0 - occupancies).mean()
            occupancy_reg = occupancy_off_penalty * occupancy_off
            loss = loss + occupancy_reg

        if self.global_step % int(cfg_get(self.cfg.train, "log_every", 100)) == 0:
            logs = {"data": data_loss.detach(), "total": loss.detach()}
            if unscaled_data_loss is not None:
                logs.update(
                    {
                        "real_raw": unscaled_data_loss.detach(),
                        "real_scale": data_loss_scale,
                    }
                )
            if self.hetero:
                assert z_mu is not None and z_logvar is not None
                logs.update(
                    {
                        "kld": kld.detach(),
                        "kl_scale": kl_scale,
                        "kl_loss": kl_loss.detach(),
                        "z_mu_norm": z_mu.detach().norm(dim=-1).mean(),
                        "z_std": torch.exp(0.5 * z_logvar.detach()).mean(),
                    }
                )
                if self.volume.deform_attributes == "xyz_occupancy":
                    assert latent is not None
                    if occupancies is None:
                        with torch.no_grad():
                            occupancies = torch.stack(
                                [
                                    self.volume.occupancy(latent[i]).detach()
                                    for i in range(latent.shape[0])
                                ],
                                dim=0,
                            )
                    logs.update(
                        {
                            "occ_temperature": data_loss.new_tensor(
                                self.volume.occupancy_temperature
                            ),
                            "occ_mean": occupancies.mean(),
                            "occ_min": occupancies.min(),
                            "occ_max": occupancies.max(),
                            "occ_lt_0.1": (occupancies < 0.1).float().mean(),
                            "occ_off": (1.0 - occupancies).mean(),
                            "occ_reg": occupancy_reg.detach(),
                        }
                    )
            log_to_current(
                f"epoch {self.current_epoch + 1} "
                f"[{batch_idx}/{self.trainer.num_training_batches}] | "
                + pretty_dict(logs, 6)
            )
        return loss

    def configure_optimizers(self):
        base_lr = float(cfg_get(self.cfg.optim, "lr", 1e-4))
        weight_decay = float(cfg_get(self.cfg.optim, "weight_decay", 0.0))
        optimizer_eps = float(cfg_get(self.cfg.optim, "eps", 1e-8))
        if optimizer_eps <= 0:
            raise ValueError("optim.eps must be positive.")
        gaussians = self.volume.gaussians
        parameter_groups = [
            {
                "params": [gaussians.xyz],
                "lr": float(cfg_get(self.cfg.optim, "position_lr", base_lr)),
                "name": "xyz",
            },
            {
                "params": [gaussians.scale_raw],
                "lr": float(cfg_get(self.cfg.optim, "scale_lr", base_lr)),
                "name": "scale",
            },
            {
                "params": [gaussians.rotation_raw],
                "lr": float(cfg_get(self.cfg.optim, "rotation_lr", base_lr)),
                "name": "rotation",
            },
            {
                "params": [gaussians.strength_raw],
                "lr": float(cfg_get(self.cfg.optim, "strength_lr", base_lr)),
                "name": "strength",
            },
        ]

        if self.hetero:
            assert self.volume.deform is not None and self.encoder is not None
            parameter_groups.extend(
                [
                    {
                        "params": self.volume.deform.parameters(),
                        "lr": float(cfg_get(self.cfg.optim, "deform_lr", base_lr)),
                        "name": "deform",
                    },
                    {
                        "params": self.encoder.parameters(),
                        "lr": float(cfg_get(self.cfg.optim, "encoder_lr", base_lr)),
                        "name": "encoder",
                    },
                ]
            )

        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            eps=optimizer_eps,
        )
        scheduler_name = str(cfg_get(self.cfg.optim, "scheduler", "none")).lower()
        if scheduler_name in {"none", "constant", "off"}:
            return optimizer
        if scheduler_name == "cosine":
            min_lr_ratio = float(cfg_get(self.cfg.optim, "min_lr_ratio", 0.1))
            if not 0.0 <= min_lr_ratio <= 1.0:
                raise ValueError("optim.min_lr_ratio must be in [0, 1].")
            total_epochs = max(1, int(cfg_get(self.cfg.trainer, "max_epochs", 1)))

            def cosine_multiplier(epoch: int) -> float:
                progress = min(max(float(epoch) / total_epochs, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=cosine_multiplier
            )
        elif scheduler_name == "exponential":
            gamma = float(cfg_get(self.cfg.optim, "gamma", 0.95))
            if not 0.0 < gamma <= 1.0:
                raise ValueError("optim.gamma must be in (0, 1].")
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        else:
            raise ValueError(
                "optim.scheduler must be 'none', 'cosine', or 'exponential'."
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": f"{scheduler_name}_lr",
            },
        }

    def on_train_epoch_start(self):
        if self.hetero and self.volume.deform_attributes == "xyz_occupancy":
            epoch = int(self.current_epoch) + 1
            self.volume.occupancy_temperature = interpolate_epoch_schedule(
                self.occupancy_temperature_schedule, epoch
            )
        if not getattr(self.trainer, "is_global_zero", True):
            return
        optimizer = self.trainer.optimizers[0]
        current_lrs = {
            group.get("name", f"group{index}"): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        log_to_current(
            f"[lr] epoch {self.current_epoch + 1} | " + pretty_dict(current_lrs, 8)
        )
        if self.hetero and self.volume.deform_attributes == "xyz_occupancy":
            log_to_current(
                f"[occupancy-temperature] epoch {self.current_epoch + 1} | "
                f"temperature={self.volume.occupancy_temperature:g}"
            )

    def on_train_epoch_end(self):
        self._last_completed_epoch = int(self.current_epoch) + 1
        self.export_epoch(force=False, epoch=self._last_completed_epoch)

    def on_fit_end(self):
        # Guarantee a final volume (and posterior/PCA in hetero mode) even when the
        # final epoch is not divisible by eval.every_n_epochs.
        if self._last_completed_epoch > 0:
            self.export_epoch(force=True, epoch=self._last_completed_epoch)

    @torch.no_grad()
    def _encode_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.hetero:
            raise RuntimeError("Homogeneous mode has no VAE posterior to encode.")
        batch_size = int(
            cfg_get(self.vae_cfg, "export_batch_size", cfg_get(self.cfg.loader, "batch_size", 8))
        )
        num_workers = int(cfg_get(self.vae_cfg, "export_num_workers", cfg_get(self.cfg.loader, "num_workers", 0)))
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=bool(cfg_get(self.cfg.loader, "pin_memory", torch.cuda.is_available())),
            persistent_workers=False,
        )

        means = np.empty((len(self.dataset), self.z_dim), dtype=np.float32)
        logvars = np.empty_like(means)
        was_training = self.training
        self.eval()
        for batch in loader:
            _, centered = self._prepare_target(batch)
            z_mu, z_logvar = self.encode_particles(centered, batch)
            indices = batch["idx"].numpy().astype(np.int64, copy=False)
            means[indices] = z_mu.float().cpu().numpy()
            logvars[indices] = z_logvar.float().cpu().numpy()
        self.train(was_training)
        return means, logvars

    @torch.no_grad()
    def export_epoch(self, force: bool = False, epoch: int | None = None):
        if not getattr(self.trainer, "is_global_zero", True):
            return

        epoch = int(self.current_epoch) + 1 if epoch is None else int(epoch)
        every = int(cfg_get(getattr(self.cfg, "eval", None), "every_n_epochs", 1))
        if not force and every > 1 and epoch % every != 0:
            return
        if self._last_export_epoch == epoch:
            return
        self._last_export_epoch = epoch

        output_dir = osp.join(self.cfg.work_dir, f"epoch{epoch:04d}_step{self.global_step:07d}")
        mkdir_or_exist(output_dir)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "epoch": epoch,
                "step": int(self.global_step),
                "occupancy_temperature": float(self.volume.occupancy_temperature),
            },
            osp.join(output_dir, "model_state.pt"),
        )

        if not self.hetero:
            self._export_volume(None, osp.join(output_dir, "vol_consensus.mrc"))
            log_to_current(f"[export] homo volume -> {output_dir}")
            return

        z_mu, z_logvar = self._encode_dataset()
        # z.npy remains a compatibility alias for existing CryoSTAR analysis tools.
        np.save(osp.join(output_dir, "z.npy"), z_mu)
        np.save(osp.join(output_dir, "z_mu.npy"), z_mu)
        np.save(osp.join(output_dir, "z_logvar.npy"), z_logvar)

        dtype = self.volume.gaussians.xyz.dtype
        consensus_z = torch.from_numpy(z_mu.mean(axis=0)).to(self.device, dtype=dtype)
        self._export_volume(consensus_z, osp.join(output_dir, "vol_consensus.mrc"))
        export_gaussian_states = bool(
            cfg_get(getattr(self.cfg, "eval", None), "export_gaussian_states", False)
        )
        if export_gaussian_states:
            self._export_gaussian_state(
                consensus_z, osp.join(output_dir, "gaussians_consensus.npz")
            )

        eval_cfg = getattr(self.cfg, "eval", None)
        try:
            self._export_fine_class_representatives(
                z_mu=z_mu,
                output_dir=output_dir,
                eval_cfg=eval_cfg,
                dtype=dtype,
                export_gaussian_states=export_gaussian_states,
            )
        except Exception as exc:
            log_to_current(f"[export] fine-class volume export skipped: {exc}")

        selected_latents = None
        selected_indices = None
        selected_labels = None
        umap_cluster_sizes = None

        try:
            pca = compute_latent_pca(z_mu)
            scores = pca["scores"]
            ev_ratio = pca["explained_variance_ratio"]
            np.save(osp.join(output_dir, "z_pca_scores.npy"), scores.astype(np.float32))
            np.save(osp.join(output_dir, "z_pca_components.npy"), pca["components"].astype(np.float32))
            np.save(
                osp.join(output_dir, "z_pca_explained_variance_ratio.npy"),
                ev_ratio.astype(np.float32),
            )

            title = str(cfg_get(eval_cfg, "latent_plot_title", "")).strip()
            if not title:
                title = str(cfg_get(self.cfg, "exp_name", "")).strip()
            umap_embedding = None
            class_selected_latents = None
            class_selected_indices = None
            class_selected_labels = None
            if bool(cfg_get(eval_cfg, "export_umap", True)):
                try:
                    random_state_cfg = cfg_get(eval_cfg, "umap_random_state", 42)
                    random_state = None if random_state_cfg is None else int(random_state_cfg)
                    umap_result = compute_latent_umap(
                        z_mu,
                        n_neighbors=int(cfg_get(eval_cfg, "umap_n_neighbors", 50)),
                        min_dist=float(cfg_get(eval_cfg, "umap_min_dist", 0.08)),
                        metric=str(cfg_get(eval_cfg, "umap_metric", "euclidean")),
                        random_state=random_state,
                        fit_max_points=int(cfg_get(eval_cfg, "umap_fit_max_points", 0) or 0),
                    )
                    umap_embedding = umap_result["embedding"]
                    np.save(osp.join(output_dir, "z_umap.npy"), umap_embedding)
                    if umap_result["fit_indices"] is not None:
                        np.savetxt(
                            osp.join(output_dir, "z_umap_fit_indices.txt"),
                            umap_result["fit_indices"],
                            fmt="%d",
                        )
                    umap_meta_path = osp.join(output_dir, "umap_metadata.txt")
                    with open(umap_meta_path, "w", encoding="utf-8") as handle:
                        for key in ("n_neighbors", "min_dist", "metric", "random_state"):
                            handle.write(f"{key}\\t{umap_result[key]}\\n")
                        handle.write(f"fit_max_points\\t{int(cfg_get(eval_cfg, 'umap_fit_max_points', 0) or 0)}\\n")

                    requested_representatives = int(
                        cfg_get(eval_cfg, "umap_num_representatives", 10)
                    )
                    if requested_representatives > 0:
                        representative_result = select_umap_representatives(
                            umap_embedding,
                            num_representatives=requested_representatives,
                            random_state=random_state,
                        )
                        selected_indices = representative_result["selected_indices"]
                        selected_latents = z_mu[selected_indices]
                        selected_labels = [
                            f"U{index:03d}" for index in range(len(selected_indices))
                        ]
                        umap_cluster_sizes = representative_result["cluster_sizes"]
                        np.save(
                            osp.join(output_dir, "z_umap_cluster_labels.npy"),
                            representative_result["cluster_labels"],
                        )
                        np.save(
                            osp.join(output_dir, "z_umap_cluster_centers.npy"),
                            representative_result["cluster_centers"],
                        )
                        np.savetxt(
                            osp.join(output_dir, "z_umap_representatives.txt"),
                            selected_latents,
                            fmt="%.7f",
                        )
                        np.savetxt(
                            osp.join(output_dir, "z_umap_representative_indices.txt"),
                            selected_indices,
                            fmt="%d",
                        )
                        original_indices = self.dataset.df.iloc[selected_indices][
                            "__orig_index"
                        ].to_numpy(dtype=np.int64)
                        np.savetxt(
                            osp.join(
                                output_dir,
                                "z_umap_representative_original_indices.txt",
                            ),
                            original_indices,
                            fmt="%d",
                        )
                        log_to_current(
                            f"[export] selected {len(selected_indices)} representatives "
                            "by k-means in 2D UMAP space"
                        )
                    for ext in ("png", "svg"):
                        scatter_path = osp.join(output_dir, f"latent_umap_scatter.{ext}")
                        plotted = plot_latent_umap_scatter(
                            umap_embedding=umap_embedding,
                            output_path=scatter_path,
                            title=title,
                            max_points=int(cfg_get(eval_cfg, "latent_scatter_max_points", 100_000)),
                            bins=int(cfg_get(eval_cfg, "latent_scatter_bins", 60)),
                            alpha=float(cfg_get(eval_cfg, "latent_scatter_alpha", 0.08)),
                            point_size=float(cfg_get(eval_cfg, "latent_scatter_size", 0.45)),
                            selected_indices=selected_indices,
                            selected_labels=selected_labels,
                        )
                        if plotted and ext == "png":
                            shutil.copyfile(scatter_path, osp.join(output_dir, "pose_umap_2d.png"))
                            shutil.copyfile(
                                scatter_path,
                                osp.join(output_dir, "vae_umap_2d_selected.png"),
                            )
                except Exception as exc:
                    log_to_current(f"[export] UMAP plot skipped: {exc}")
            else:
                log_to_current("[export] UMAP disabled by eval.export_umap=False.")

            class_label_file = str(cfg_get(eval_cfg, "class_label_file", "") or "").strip()
            if class_label_file:
                try:
                    raw_labels = load_class_label_vector(
                        class_label_file,
                        column=cfg_get(eval_cfg, "class_label_column", None),
                    )
                    class_labels = labels_for_dataset(raw_labels, self.dataset)
                    class_keys = np.asarray([_label_key(value) for value in class_labels], dtype=object)
                    np.save(osp.join(output_dir, "particle_class_labels.npy"), class_labels)

                    class_order = _parse_label_values(cfg_get(eval_cfg, "class_label_order", None))
                    if not class_order:
                        class_order = sorted(set(class_keys.tolist()), key=_label_sort_key)
                    class_names = _plain_dict(cfg_get(eval_cfg, "class_label_names", None))
                    class_colors = _plain_dict(cfg_get(eval_cfg, "class_label_colors", None))
                    center_ignore = set(
                        _parse_label_values(
                            cfg_get(
                                eval_cfg,
                                "class_label_center_ignore_values",
                                [-1, "nan", ""],
                            )
                        )
                    )

                    summary_rows = []
                    medoid_latents = []
                    medoid_indices = []
                    medoid_labels = []
                    for label_value in class_order:
                        key = _label_key(label_value)
                        mask = class_keys == key
                        count = int(np.count_nonzero(mask))
                        if count == 0:
                            continue
                        medoid_index = -1
                        original_index = -1
                        volume_file = ""
                        gaussian_file = ""
                        center_latent = z_mu[mask].mean(axis=0)
                        if key not in center_ignore:
                            subset_indices = np.flatnonzero(mask)
                            distances = np.square(z_mu[subset_indices] - center_latent[None]).sum(axis=1)
                            medoid_index = int(subset_indices[int(np.argmin(distances))])
                            original_index = int(self.dataset.df.iloc[medoid_index]["__orig_index"])
                            label_name = _label_name(key, class_names)
                            stem = _sanitize_stem(label_name)
                            volume_file = f"vol_class_{stem}.mrc"
                            gaussian_file = f"gaussians_class_{stem}.npz"
                            medoid_latents.append(z_mu[medoid_index])
                            medoid_indices.append(medoid_index)
                            medoid_labels.append(label_name)
                            if bool(cfg_get(eval_cfg, "export_class_volumes", True)):
                                z = torch.from_numpy(z_mu[medoid_index]).to(self.device, dtype=dtype)
                                self._export_volume(z, osp.join(output_dir, volume_file))
                                if export_gaussian_states:
                                    self._export_gaussian_state(z, osp.join(output_dir, gaussian_file))
                        umap1 = float("nan")
                        umap2 = float("nan")
                        if umap_embedding is not None and medoid_index >= 0:
                            umap1 = float(umap_embedding[medoid_index, 0])
                            umap2 = float(umap_embedding[medoid_index, 1])
                        pc1 = float("nan")
                        pc2 = float("nan")
                        if medoid_index >= 0:
                            pc1 = float(scores[medoid_index, 0])
                            pc2 = float(scores[medoid_index, 1]) if scores.shape[1] >= 2 else float("nan")
                        summary_rows.append(
                            [
                                key,
                                _label_name(key, class_names),
                                count,
                                medoid_index,
                                original_index,
                                volume_file,
                                gaussian_file,
                                umap1,
                                umap2,
                                pc1,
                                pc2,
                            ]
                            + [float(value) for value in center_latent]
                        )

                    if medoid_latents:
                        class_selected_latents = np.asarray(medoid_latents, dtype=np.float32)
                        class_selected_indices = np.asarray(medoid_indices, dtype=np.int64)
                        class_selected_labels = medoid_labels
                        np.savetxt(
                            osp.join(output_dir, "z_class_medoids.txt"),
                            class_selected_latents,
                            fmt="%.7f",
                        )
                        np.savetxt(
                            osp.join(output_dir, "z_class_medoid_indices.txt"),
                            class_selected_indices,
                            fmt="%d",
                        )
                        original_indices = self.dataset.df.iloc[class_selected_indices][
                            "__orig_index"
                        ].to_numpy(dtype=np.int64)
                        np.savetxt(
                            osp.join(output_dir, "z_class_medoid_original_indices.txt"),
                            original_indices,
                            fmt="%d",
                        )

                    with open(
                        osp.join(output_dir, "class_label_summary.csv"),
                        "w",
                        newline="",
                        encoding="utf-8",
                    ) as handle:
                        writer = csv.writer(handle)
                        writer.writerow(
                            [
                                "class_label",
                                "class_name",
                                "particle_count",
                                "medoid_local_particle_index",
                                "medoid_original_star_index",
                                "volume_file",
                                "gaussian_file",
                                "umap1",
                                "umap2",
                                "pc1",
                                "pc2",
                            ]
                            + [f"z_center_{i}" for i in range(self.z_dim)]
                        )
                        writer.writerows(summary_rows)

                    if bool(cfg_get(eval_cfg, "write_particle_label_table", True)):
                        with open(
                            osp.join(output_dir, "particle_class_labels.csv"),
                            "w",
                            newline="",
                            encoding="utf-8",
                        ) as handle:
                            writer = csv.writer(handle)
                            writer.writerow(
                                [
                                    "local_particle_index",
                                    "original_star_index",
                                    "image_name",
                                    "class_label",
                                    "class_name",
                                    "umap1",
                                    "umap2",
                                    "pc1",
                                    "pc2",
                                ]
                            )
                            for local_index, key in enumerate(class_keys):
                                row = self.dataset.df.iloc[local_index]
                                if umap_embedding is not None:
                                    umap1 = float(umap_embedding[local_index, 0])
                                    umap2 = float(umap_embedding[local_index, 1])
                                else:
                                    umap1 = float("nan")
                                    umap2 = float("nan")
                                writer.writerow(
                                    [
                                        local_index,
                                        int(row["__orig_index"]),
                                        str(row.get("rlnImageName", "")),
                                        key,
                                        _label_name(key, class_names),
                                        umap1,
                                        umap2,
                                        float(scores[local_index, 0]),
                                        float(scores[local_index, 1]) if scores.shape[1] >= 2 else float("nan"),
                                    ]
                                )

                    if umap_embedding is not None:
                        for ext in ("png", "svg"):
                            plot_latent_umap_by_label(
                                umap_embedding=umap_embedding,
                                labels=class_labels,
                                output_path=osp.join(output_dir, f"latent_umap_by_class.{ext}"),
                                title=title,
                                class_order=class_order,
                                class_names=class_names,
                                class_colors=class_colors,
                                max_points=int(cfg_get(eval_cfg, "latent_scatter_max_points", 100_000)),
                                alpha=float(cfg_get(eval_cfg, "class_scatter_alpha", 0.18)),
                                point_size=float(cfg_get(eval_cfg, "class_scatter_size", 0.45)),
                                selected_indices=class_selected_indices,
                                selected_labels=class_selected_labels,
                            )
                except Exception as exc:
                    log_to_current(f"[export] class-label validation skipped: {exc}")

            if selected_indices is not None:
                selected_rows = self.dataset.df.iloc[selected_indices]
                mapping_path = osp.join(output_dir, "umap_representatives.csv")
                with open(mapping_path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "selection_id",
                            "volume_file",
                            "local_particle_index",
                            "original_star_index",
                            "image_name",
                            "umap1",
                            "umap2",
                            "pc1",
                            "pc2",
                            "umap_cluster_particle_count",
                        ]
                        + [f"z{i}" for i in range(self.z_dim)]
                    )
                    selections = zip(selected_labels, selected_indices, selected_latents)
                    for row_number, (selection_id, local_index, latent) in enumerate(selections):
                        row = selected_rows.iloc[row_number]
                        if umap_embedding is not None:
                            umap1 = float(umap_embedding[local_index, 0])
                            umap2 = float(umap_embedding[local_index, 1])
                        else:
                            umap1 = float("nan")
                            umap2 = float("nan")
                        writer.writerow(
                            [
                                selection_id,
                                f"vol_umap_{int(selection_id[1:]):03d}.mrc",
                                int(local_index),
                                int(row["__orig_index"]),
                                str(row.get("rlnImageName", "")),
                                umap1,
                                umap2,
                                float(scores[local_index, 0]),
                                float(scores[local_index, 1]) if scores.shape[1] >= 2 else float("nan"),
                                int(umap_cluster_sizes[row_number])
                                if umap_cluster_sizes is not None
                                else -1,
                            ]
                            + [float(value) for value in latent]
                        )

            for ext in ("png", "svg"):
                plot_latent_pc1_distribution(
                    pc1_scores=scores[:, 0],
                    explained_variance_ratio=ev_ratio,
                    output_path=osp.join(output_dir, f"latent_pc1_distribution.{ext}"),
                    title=title,
                    z_norm_mean=float(np.linalg.norm(z_mu, axis=1).mean()),
                    bins=int(cfg_get(eval_cfg, "latent_hist_bins", 80)),
                    smooth_sigma_bins=float(cfg_get(eval_cfg, "latent_hist_smooth_sigma", 1.5)),
                )
        except Exception as exc:
            log_to_current(f"[export] posterior latent plots skipped: {exc}")

        if selected_latents is not None:
            try:
                for index, center in enumerate(selected_latents):
                    z = torch.from_numpy(center).to(self.device, dtype=dtype)
                    if bool(cfg_get(eval_cfg, "export_umap_volumes", True)):
                        self._export_volume(
                            z, osp.join(output_dir, f"vol_umap_{index:03d}.mrc")
                        )
                    if export_gaussian_states and bool(
                        cfg_get(eval_cfg, "export_umap_volumes", True)
                    ):
                        self._export_gaussian_state(
                            z, osp.join(output_dir, f"gaussians_umap_{index:03d}.npz")
                        )
            except Exception as exc:
                log_to_current(f"[export] UMAP representative volume export skipped: {exc}")

        pca_dims = min(int(cfg_get(eval_cfg, "pca_dims", 2)), self.z_dim)
        pca_steps = int(cfg_get(eval_cfg, "pca_steps", 4))
        export_pca_volumes = bool(cfg_get(eval_cfg, "export_pca_volumes", True))
        if export_pca_volumes and pca_dims > 0 and pca_steps > 1:
            try:
                for component in range(1, pca_dims + 1):
                    z_path, nearest_indices = sample_along_pca(z_mu, component, pca_steps)
                    np.savetxt(osp.join(output_dir, f"z_pca_{component}.txt"), z_path, fmt="%.7f")
                    np.savetxt(
                        osp.join(output_dir, f"z_pca_{component}_indices.txt"),
                        nearest_indices,
                        fmt="%d",
                    )
                    for index, point in enumerate(z_path):
                        z = torch.from_numpy(point).to(self.device, dtype=dtype)
                        self._export_volume(z, osp.join(output_dir, f"vol_pca_{component}_{index:03d}.mrc"))
                        if export_gaussian_states:
                            self._export_gaussian_state(
                                z,
                                osp.join(
                                    output_dir,
                                    f"gaussians_pca_{component}_{index:03d}.npz",
                                ),
                            )
            except Exception as exc:
                log_to_current(f"[export] PCA volume path skipped: {exc}")

        log_to_current(f"[export] VAE posterior / UMAP / class validation -> {output_dir}")


def build_dataset(cfg) -> StarDataset:
    dataset_cfg = StarDatasetConfig(
        star=cfg.dataset.star,
        root=cfg.dataset.root,
        apix=cfg_get(cfg.dataset, "apix", None),
        size=cfg_get(cfg.dataset, "size", None),
        down_size=cfg_get(cfg.dataset, "down_size", None),
        down_method=cfg_get(cfg.dataset, "down_method", "interp"),
        mask_radius=cfg_get(cfg.dataset, "mask_radius", 1.0),
        scale_images=float(cfg_get(cfg.dataset, "scale_images", 1.0)),
        power_images=float(cfg_get(cfg.dataset, "power_images", 1.0)),
        half=cfg_get(cfg.dataset, "half", None),
        half_split_mode=cfg_get(cfg.dataset, "half_split_mode", "relion"),
        parity_offset=int(cfg_get(cfg.dataset, "parity_offset", 0)),
        random_subset_col=cfg_get(cfg.dataset, "random_subset_col", "rlnRandomSubset"),
        require_random_subset=bool(cfg_get(cfg.dataset, "require_random_subset", True)),
        lazy=bool(cfg_get(cfg.dataset, "lazy", True)),
        cache_metadata=bool(cfg_get(cfg.dataset, "cache_metadata", False)),
        cache_normalized_images=bool(cfg_get(cfg.dataset, "cache_normalized_images", False)),
        cache_shifted_target=bool(cfg_get(cfg.dataset, "cache_shifted_target", False)),
        cache_chunk_size=int(cfg_get(cfg.dataset, "cache_chunk_size", 256)),
        return_fproj=bool(
            cfg_get(
                cfg.dataset,
                "return_fproj",
                str(cfg_get(cfg.loss, "type", "fourier_mse")).lower() != "real_mse",
            )
        ),
        minimal_batch=bool(
            cfg_get(
                cfg.dataset,
                "minimal_batch",
                (
                    str(cfg_get(cfg.loss, "type", "fourier_mse")).lower() == "real_mse"
                    and not bool(cfg_get(cfg.model, "hetero", True))
                    and bool(cfg_get(cfg.dataset, "cache_shifted_target", False))
                ),
            )
        ),
    )
    dataset = StarDataset(dataset_cfg)
    cfg.dataset_attr.apix = dataset.apix
    cfg.dataset_attr.side_shape = dataset.size
    cfg.data_process.down_side_shape = dataset.down_size
    cfg.data_process.down_apix = dataset.down_apix
    return dataset


def train():
    cfg = pl_init_exp(
        exp_prefix=TASK_NAME,
        backup_list=[__file__, _dataio.__file__],
        inplace=False,
    )
    cfg = ensure_ctf_compatibility(cfg)
    if not bool(cfg_get(cfg.model, "hetero", True)):
        raise ValueError("Use train_homo.py for homogeneous reconstruction settings.")

    seed = cfg_get(cfg, "seed", None)
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)
        log_to_current(f"[seed] {seed}")

    dataset = build_dataset(cfg)
    log_to_current(
        f"[dataset] particles={len(dataset)} / "
        f"total={getattr(dataset, 'n_particles_total', len(dataset))}; "
        f"half={getattr(dataset, 'half', None)}; "
        f"lazy={getattr(dataset, 'lazy', True)}"
    )
    log_to_current(
        "[dataset-cache] "
        f"metadata={getattr(dataset, 'cache_metadata', False)} | "
        f"normalized_images={getattr(dataset, 'cache_normalized_images', False)} | "
        f"shifted_target={getattr(dataset, 'cache_shifted_target', False)} | "
        f"return_fproj={getattr(dataset, 'return_fproj', True)} | "
        f"minimal_batch={getattr(dataset, 'minimal_batch', False)}"
    )
    if cfg_get(cfg.dataset, "f_mu", None) is not None:
        dataset.f_mu = float(cfg.dataset.f_mu)
        dataset.f_std = float(cfg.dataset.f_std)
    else:
        dataset.estimate_normalization()
    log_to_current(f"[normalization] Fourier mean/std={dataset.f_mu:.6f}/{dataset.f_std:.6f}")
    if hasattr(dataset, "finalize_caches"):
        dataset.finalize_caches(shift_method=cfg.model.shift_method)

    rank_zero_only(cfg.dump)(osp.join(cfg.work_dir, "config.py"))
    num_workers = int(cfg_get(cfg.loader, "num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg_get(cfg.loader, "batch_size", 8)),
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=bool(cfg_get(cfg.loader, "pin_memory", torch.cuda.is_available())),
        persistent_workers=bool(cfg_get(cfg.loader, "persistent_workers", num_workers > 0)) and num_workers > 0,
    )

    model = VAEGaussianRecon(cfg, dataset)
    devices = int(cfg_get(cfg.trainer, "devices", 1))
    strategy = DDPStrategy(find_unused_parameters=False) if devices > 1 else "auto"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        **cfg.trainer,
    )
    trainer.fit(model, train_dataloaders=loader)


if __name__ == "__main__":
    train()
