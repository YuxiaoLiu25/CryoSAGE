# Adapted from CryoSTAR under Apache-2.0. See NOTICE and provenance.json.
"""Gaussian reconstruction using known particle poses.

The public homogeneous entry point optimizes one shared volume and rejects
heterogeneous settings. Common Gaussian and encoder definitions are retained
for numerical compatibility with the existing training implementation.

    centered particle -> (optional CTF phase flip) -> Hartley image
                      -> encoder -> (z_mu, z_logvar) -> sampled z
                      -> Gaussian deformation field -> projection

The RELION rotation/translation stored in the STAR file remain the known imaging
pose.  The VAE latent represents particle conformation; it is not an SO(3) pose.
With ``model.hetero=False``, neither the VAE encoder nor the deformation field is
constructed; all particles share one canonical Gaussian volume.

Latent-conditioned runs use the separate ``train.py`` entry point.
Gaussian and training settings include:

    model = dict(
        ...,
        hetero=True,               # False: canonical homo reconstruction
        gaussian_parameterization="density",  # "density" (peak) or "amplitude" (integral)
        init_density=9e-4,          # also defines equivalent amplitude initialization
        # init_amplitude=...,       # optional explicit override in amplitude mode
        renderer_gain=1.0,          # >1 lowers R2's effective absolute alpha cutoff
        deform_attributes="xyz",    # "xyz" or "all"; per-particle hetero attributes
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
        kmeans_k=10,                # representative conformations marked in PCA
    )

At every configured export epoch, and unconditionally after the final epoch,
heterogeneous mode saves posterior means/log-variances and a two-dimensional PCA
plot of z_mu. The marked C000/C001/... points correspond exactly to the exported
k-means volumes. Homogeneous mode exports only its checkpoint and consensus volume.
"""
from __future__ import annotations

import csv
import math
import os.path as osp
import shutil
from typing import Any

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from mmengine import mkdir_or_exist
from torch.utils.data import DataLoader

from cryosage.utils.ctf_utils import CTFParticle, CTFRelion
from cryosage.utils.fft_utils import fourier_to_primal_2d, primal_to_fourier_2d
from cryosage.utils.latent import cluster_kmeans, get_nearest_point, sample_along_pca
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
        "all": "all",
        "all_attributes": "all",
        "full": "all",
    }
    if text not in aliases:
        raise ValueError(
            "model.deform_attributes must be 'xyz'/'position' or 'all', "
            f"got {value!r}."
        )
    return aliases[text]


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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = pca_scores[:, 0], pca_scores[:, 1]
    if len(x) > int(max_points):
        keep = np.random.default_rng(42).choice(len(x), int(max_points), replace=False)
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
            ax.legend(loc="best", fontsize=7, frameon=False)
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
    fig.savefig(output_path, dpi=240)
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.savefig(output_path, dpi=240)
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
        self.deform_output_dim = 3 if self.deform_attributes == "xyz" else 11
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
            self.deform: DeformationMLP | None = DeformationMLP(
                z_dim=int(model_cfg.z_dim),
                hidden=int(cfg_get(model_cfg, "deform_hidden", 64)),
                layers=int(cfg_get(model_cfg, "deform_layers", 3)),
                pe_freqs=int(cfg_get(model_cfg, "deform_pe_freqs", 0)),
                pe_include_xyz=bool(cfg_get(model_cfg, "deform_pe_include_xyz", True)),
                output_dim=self.deform_output_dim,
            )
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
        if not self.hetero:
            return dict(
                xyz=g.xyz,
                density=g.density,
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
            scaling=scaling,
            rotation=rotation,
        )

    def positions(self, z: torch.Tensor | None) -> torch.Tensor:
        return self.attributes(z)["xyz"]

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
    def voxelize(self, z: torch.Tensor | None) -> torch.Tensor:
        size = self.image_size
        settings = GaussianVoxelizationSettings(
            scale_modifier=1.0,
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
                f"scale={self.volume.deform_scale:g} | "
                f"tanh={self.volume.deform_tanh} | "
                f"pe_freqs={self.volume.deform.pe_freqs} | "
                f"input_dim={self.volume.deform.input_dim} | "
                f"output_dim={self.volume.deform.output_dim}"
            )
            if self.volume.deform_attributes == "all":
                log_to_current(
                    "[deform-attributes] "
                    f"scale_raw_scale={self.volume.deform_scale_raw_scale:g} | "
                    f"rotation_raw_scale={self.volume.deform_rotation_raw_scale:g} | "
                    f"strength_raw_scale={self.volume.deform_strength_raw_scale:g}"
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
    def _export_volume(self, z: torch.Tensor | None, path: str) -> None:
        volume = self.volume.voxelize(z).detach().cpu().numpy()
        pixel = float(self.cfg.data_process.down_apix)
        save_mrc(volume, path, pixel, -pixel * (volume.shape[0] // 2))

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
            {"state_dict": self.state_dict(), "epoch": epoch, "step": int(self.global_step)},
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

        eval_cfg = getattr(self.cfg, "eval", None)
        selected_latents = None
        selected_indices = None
        selected_labels = None
        kmeans_k = min(int(cfg_get(eval_cfg, "kmeans_k", 0)), len(z_mu))
        if kmeans_k > 1:
            try:
                _, centers = cluster_kmeans(z_mu, kmeans_k)
                selected_latents, selected_indices = get_nearest_point(z_mu, centers)
                selected_labels = [f"C{i:03d}" for i in range(len(selected_indices))]
                np.savetxt(osp.join(output_dir, "z_kmeans.txt"), selected_latents, fmt="%.7f")
                np.savetxt(
                    osp.join(output_dir, "z_kmeans_indices.txt"),
                    selected_indices,
                    fmt="%d",
                )
                original_indices = self.dataset.df.iloc[selected_indices]["__orig_index"].to_numpy(
                    dtype=np.int64
                )
                np.savetxt(
                    osp.join(output_dir, "z_kmeans_original_indices.txt"),
                    original_indices,
                    fmt="%d",
                )
            except Exception as exc:
                selected_latents = None
                selected_indices = None
                selected_labels = None
                log_to_current(f"[export] k-means selection skipped: {exc}")

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
            scatter_path = osp.join(output_dir, "latent_pca_scatter.png")
            if scores.shape[1] >= 2:
                plotted = plot_latent_pca_scatter(
                    pca_scores=scores,
                    explained_variance_ratio=ev_ratio,
                    output_path=scatter_path,
                    title=title,
                    max_points=int(cfg_get(eval_cfg, "latent_scatter_max_points", 100_000)),
                    bins=int(cfg_get(eval_cfg, "latent_scatter_bins", 60)),
                    alpha=float(cfg_get(eval_cfg, "latent_scatter_alpha", 0.08)),
                    selected_indices=selected_indices,
                    selected_labels=selected_labels,
                )
                if plotted:
                    shutil.copyfile(scatter_path, osp.join(output_dir, "pose_pca_2d.png"))
                    shutil.copyfile(
                        scatter_path,
                        osp.join(output_dir, "vae_pca_2d_selected.png"),
                    )
            else:
                log_to_current("[export] 2D PCA plot skipped because model.z_dim < 2.")

            if selected_indices is not None:
                selected_rows = self.dataset.df.iloc[selected_indices]
                mapping_path = osp.join(output_dir, "selected_conformations.csv")
                with open(mapping_path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "selection_id",
                            "volume_file",
                            "local_particle_index",
                            "original_star_index",
                            "image_name",
                            "pc1",
                            "pc2",
                        ]
                        + [f"z{i}" for i in range(self.z_dim)]
                    )
                    selections = zip(selected_labels, selected_indices, selected_latents)
                    for row_number, (selection_id, local_index, latent) in enumerate(selections):
                        row = selected_rows.iloc[row_number]
                        writer.writerow(
                            [
                                selection_id,
                                f"vol_kmeans_{int(selection_id[1:]):03d}.mrc",
                                int(local_index),
                                int(row["__orig_index"]),
                                str(row.get("rlnImageName", "")),
                                float(scores[local_index, 0]),
                                float(scores[local_index, 1]) if scores.shape[1] >= 2 else float("nan"),
                            ]
                            + [float(value) for value in latent]
                        )

            plot_latent_pc1_distribution(
                pc1_scores=scores[:, 0],
                explained_variance_ratio=ev_ratio,
                output_path=osp.join(output_dir, "latent_pc1_distribution.png"),
                title=title,
                z_norm_mean=float(np.linalg.norm(z_mu, axis=1).mean()),
                bins=int(cfg_get(eval_cfg, "latent_hist_bins", 80)),
                smooth_sigma_bins=float(cfg_get(eval_cfg, "latent_hist_smooth_sigma", 1.5)),
            )
        except Exception as exc:
            log_to_current(f"[export] posterior PCA plots skipped: {exc}")

        if selected_latents is not None:
            try:
                for index, center in enumerate(selected_latents):
                    z = torch.from_numpy(center).to(self.device, dtype=dtype)
                    self._export_volume(z, osp.join(output_dir, f"vol_kmeans_{index:03d}.mrc"))
            except Exception as exc:
                log_to_current(f"[export] k-means volume export skipped: {exc}")

        pca_dims = min(int(cfg_get(eval_cfg, "pca_dims", 0)), self.z_dim)
        pca_steps = int(cfg_get(eval_cfg, "pca_steps", 0))
        if pca_dims > 0 and pca_steps > 1:
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
            except Exception as exc:
                log_to_current(f"[export] PCA volume path skipped: {exc}")

        log_to_current(f"[export] VAE posterior / PCA / volumes -> {output_dir}")


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
    if bool(cfg_get(cfg.model, "hetero", True)):
        raise ValueError("Use train.py for latent-conditioned reconstruction settings.")

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
