# Adapted from CryoSTAR under Apache-2.0. See NOTICE and provenance.json.
from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path

import mrcfile
import numpy as np
import starfile
import torch
import torchvision.transforms.functional as tvf
from torch.utils.data import Dataset

from cryosage.utils.fft_utils import downsample_2d, fourier_to_primal_2d, primal_to_fourier_2d
from cryosage.utils.geom_utils import euler_angles2matrix
from cryosage.utils.transforms import FourierGridTranslate, SpatialGridTranslate


class CircularMask(torch.nn.Module):
    def __init__(self, size: int, radius: float):
        super().__init__()
        y = torch.linspace(-1, 1, int(size))[:, None]
        x = torch.linspace(-1, 1, int(size))[None, :]
        self.register_buffer("mask", ((x * x + y * y) < float(radius) ** 2).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask


@dataclass
class StarDatasetConfig:
    star: str
    root: str | None = None
    apix: float | None = None
    size: int | None = None
    down_size: int | None = None
    down_method: str = "interp"  # interp or fft
    mask_radius: float | None = None
    scale_images: float = 1.0
    power_images: float = 1.0
    ignore_shift: bool = False
    ignore_rot: bool = False
    # Optional half-set selection.
    # half can be None/"all" to use all particles, or 1/2/"half1"/"half2".
    half: str | int | None = None
    #
    # "relion"  : select with random_subset_col (gold-standard RELION split).
    # "parity"  : CryoSplat-style alternating split of the original STAR row order:
    #             half1 = 0-based rows 0, 2, 4, ...; half2 = rows 1, 3, 5, ...
    half_split_mode: str = "relion"
    #
    # Optional label flip for parity mode. Keep 0 for CryoSplat-style
    # ii % 2 == 0 -> half1. Set 1 only when deliberately swapping half labels.
    parity_offset: int = 0
    #
    # Used only by half_split_mode="relion".
    random_subset_col: str = "rlnRandomSubset"
    require_random_subset: bool = True
    # Default behavior matches the previous implementation: load each particle
    # on demand from the MRC stack. Set lazy=False for small performance
    # benchmarks to keep processed particle images in CPU RAM.
    lazy: bool = True
    # Optional speed caches used only when lazy=False. They are deliberately
    # opt-in so existing configs keep the old behavior unless they request it.
    cache_metadata: bool = False
    cache_normalized_images: bool = False
    cache_shifted_target: bool = False
    cache_chunk_size: int = 256
    # Training with real_mse does not consume fproj. Keep True by default for
    # backward compatibility and for Fourier-domain losses/normalization checks.
    return_fproj: bool = True
    # Return only tensors required by the current training step. This is useful
    # for non-lazy real_mse speed tests with cached shifted targets, because
    # Lightning moves every tensor in the batch to GPU before training_step.
    minimal_batch: bool = False


class StarDataset(Dataset):
    """Minimal RELION STAR particle dataset for one-stage Gaussian training.

    Returns both clean names (`image`, `rot`, `shift_x`, `idx`) and legacy aliases
    (`proj`, `rotmat`, `shiftX`) so old training utilities still work.

    If cfg.half is set to 1/2 (or "half1"/"half2"), the particles table can be
    filtered in two ways:

    - half_split_mode="relion": RELION's rlnRandomSubset column.
    - half_split_mode="parity": 0-based original STAR row parity, matching the
      CryoSplat convention ii % 2 == 0 -> half1.

    The returned `idx` is always contiguous inside the selected subset, so it can
    safely index z_table. `orig_idx` is the stable original STAR row number and is
    retained for bookkeeping / parity verification.
    """

    @staticmethod
    def _normalize_half_value(half):
        if half is None:
            return None
        if isinstance(half, str):
            h = half.strip().lower()
            if h in ("", "all", "none", "full", "both"):
                return None
            if h in ("1", "half1", "h1", "random1", "subset1"):
                return 1
            if h in ("2", "half2", "h2", "random2", "subset2"):
                return 2
        try:
            v = int(half)
        except Exception as e:
            raise ValueError(f"Invalid half setting: {half!r}. Use None/'all'/1/2/'half1'/'half2'.") from e
        if v not in (1, 2):
            raise ValueError(f"Invalid half setting: {half!r}. Use 1 or 2.")
        return v


    @staticmethod
    def _normalize_half_split_mode(mode) -> str:
        """Normalize split-mode aliases while keeping the public config compact."""
        if mode is None:
            return "relion"
        value = str(mode).strip().lower()
        if value in ("relion", "random_subset", "gold_standard", "gold-standard"):
            return "relion"
        if value in (
            "parity", "alternating", "alternate", "even_odd", "even-odd",
            "index_parity", "cryosplat",
        ):
            return "parity"
        raise ValueError(
            "Invalid half_split_mode={!r}. Use 'relion' or 'parity' "
            "('cryosplat' is accepted as an alias).".format(mode)
        )

    def __init__(self, cfg: StarDatasetConfig):
        super().__init__()
        self.cfg = cfg
        star_path = Path(cfg.star).resolve()
        self.root = Path(cfg.root).resolve() if cfg.root else star_path.parent

        table = starfile.read(star_path)
        if isinstance(table, dict) and "particles" in table:
            self.optics_df = table.get("optics", None)
            self.df = table["particles"]
        else:
            self.optics_df = None
            self.df = table

        self.half = self._normalize_half_value(cfg.half)
        self.half_split_mode = self._normalize_half_split_mode(cfg.half_split_mode)
        self.parity_offset = int(cfg.parity_offset)
        self.random_subset_col = str(cfg.random_subset_col)

        # Preserve the original *row order* before any filtering. This is purposely
        # not pandas' dataframe index: it remains stable even if STAR parsing changes
        # the dataframe index, and it gives parity mode an explicit 0-based ii.
        self.df = self.df.copy()
        self.n_particles_total = len(self.df)
        self.df["__orig_index"] = np.arange(self.n_particles_total, dtype=np.int64)

        if self.half is not None:
            before = len(self.df)

            if self.half_split_mode == "relion":
                if self.random_subset_col not in self.df.columns:
                    msg = (
                        f"Requested half={self.half} with half_split_mode='relion', "
                        f"but column {self.random_subset_col!r} was not found in STAR file. "
                        f"Available columns include: {list(self.df.columns)[:20]}"
                    )
                    if cfg.require_random_subset:
                        raise KeyError(msg)
                    print("WARNING:", msg, "Using all particles instead.")
                    self.half = None
                else:
                    vals = self.df[self.random_subset_col].astype(int)
                    self.df = self.df.loc[vals == self.half].copy()
                    if len(self.df) == 0:
                        raise RuntimeError(
                            f"No particles found for {self.random_subset_col} == {self.half}. "
                            f"Unique values are {sorted(vals.unique().tolist())}."
                        )
                    print(
                        f"Using RELION half{self.half}: {len(self.df)} / {before} particles "
                        f"from {self.random_subset_col}."
                    )

            elif self.half_split_mode == "parity":
                # CryoSplat-style convention:
                #   ii % 2 == 0 -> half1
                #   ii % 2 == 1 -> half2
                # `ii` here is the stable 0-based original STAR row order.
                target_parity = (self.half - 1) % 2
                row_parity = (self.df["__orig_index"].to_numpy(dtype=np.int64) + self.parity_offset) % 2
                self.df = self.df.loc[row_parity == target_parity].copy()
                if len(self.df) == 0:
                    raise RuntimeError(
                        f"No particles found for parity half{self.half}. "
                        f"Check parity_offset={self.parity_offset} and input STAR rows."
                    )
                parity_name = "even" if target_parity == 0 else "odd"
                print(
                    f"Using parity half{self.half} ({parity_name} 0-based STAR rows; "
                    f"parity_offset={self.parity_offset}): {len(self.df)} / {before} particles."
                )

        # Make local indexes contiguous after optional selection. `__orig_index`
        # remains untouched and is returned as orig_idx by __getitem__.
        self.df = self.df.reset_index(drop=True)

        self.apix = float(cfg.apix) if cfg.apix is not None else self._infer_apix()
        self.size = int(cfg.size) if cfg.size is not None else self._infer_size()
        self.down_size = int(cfg.down_size) if cfg.down_size is not None else self.size
        self.down_apix = self.size * self.apix / self.down_size
        self.mask = CircularMask(self.down_size, cfg.mask_radius) if cfg.mask_radius is not None else None
        self.f_mu = None
        self.f_std = None
        self.lazy = bool(cfg.lazy)
        self.cache_metadata = bool(cfg.cache_metadata) and not self.lazy
        self.cache_normalized_images = bool(cfg.cache_normalized_images) and not self.lazy
        self.cache_shifted_target = bool(cfg.cache_shifted_target) and not self.lazy
        self.cache_chunk_size = max(1, int(cfg.cache_chunk_size))
        self.return_fproj = bool(cfg.return_fproj)
        self.minimal_batch = bool(cfg.minimal_batch)
        self._image_cache: torch.Tensor | None = None
        self._image_name_cache: list[str] | None = None
        self._image_cache_is_normalized = False
        self._shifted_target_cache: torch.Tensor | None = None
        self._rot_cache: torch.Tensor | None = None
        self._shift_x_cache: torch.Tensor | None = None
        self._shift_y_cache: torch.Tensor | None = None
        self._defocus_u_cache: torch.Tensor | None = None
        self._defocus_v_cache: torch.Tensor | None = None
        self._astig_cache: torch.Tensor | None = None
        self._orig_idx_cache: torch.Tensor | None = None
        self._random_subset_cache: torch.Tensor | None = None
        self._selected_half_cache: torch.Tensor | None = None
        self._parity_label_cache: torch.Tensor | None = None
        self._class_id_cache: torch.Tensor | None = None
        if self.cache_metadata:
            self.precompute_metadata()
        if not self.lazy:
            self.preload_images()

    def _infer_apix(self) -> float:
        if self.optics_df is not None and "rlnImagePixelSize" in self.optics_df:
            return float(self.optics_df["rlnImagePixelSize"].iloc[0])
        row = self.df.iloc[0]
        if "rlnDetectorPixelSize" in row and "rlnMagnification" in row:
            return float(row["rlnDetectorPixelSize"] / row["rlnMagnification"] * 1e4)
        raise AttributeError("Cannot infer pixel size. Set dataset.apix in config.")

    def _infer_size(self) -> int:
        img_name = str(self.df.iloc[0]["rlnImageName"]).split("@")[-1]
        with mrcfile.mmap(osp.join(self.root, img_name), mode="r", permissive=True) as m:
            return int(m.data.shape[-1])

    def __len__(self) -> int:
        return len(self.df)

    def estimate_normalization(self, n_samples: int = 100) -> None:
        if self.f_mu is not None or self.f_std is not None:
            raise RuntimeError("Normalization has already been estimated.")
        # Normalization always needs Fourier samples, even when the training loss
        # does not. Temporarily request fproj without changing the steady-state
        # training path.
        old_return_fproj = self.return_fproj
        self.return_fproj = True
        try:
            step = max(1, len(self) // max(1, int(n_samples)))
            vals = []
            for i in range(0, len(self), step):
                vals.append(self[i]["fproj"])
                if len(vals) >= int(n_samples):
                    break
            f = torch.cat(vals, dim=0)
        finally:
            self.return_fproj = old_return_fproj
        self.f_mu = 0.0
        self.f_std = torch.std(f).item()

    def preload_images(self) -> None:
        """Load all selected, processed particle images into CPU memory."""
        if self._image_cache is not None:
            return
        images = torch.empty(
            (len(self), 1, self.down_size, self.down_size),
            dtype=torch.float32,
        )
        image_names: list[str] = []
        for idx in range(len(self)):
            row = self.df.iloc[idx]
            try:
                image, raw_name = self._load_image(row)
            except Exception as e:
                raw_name = str(row.get("rlnImageName", f"idx={idx}"))
                print(f"WARNING: invalid particle {raw_name}; using zeros. {e}")
                image = torch.zeros(1, self.down_size, self.down_size)
            images[idx].copy_(image)
            image_names.append(raw_name)
        self._image_cache = images
        self._image_name_cache = image_names
        gib = images.numel() * images.element_size() / 1024**3
        print(
            f"Preloaded {len(self)} particle images into CPU RAM "
            f"({gib:.2f} GiB, shape={tuple(images.shape)})."
        )

    def precompute_metadata(self) -> None:
        """Cache fixed STAR pose/shift/CTF tensors to avoid per-sample pandas work."""
        if self._rot_cache is not None:
            return
        n = len(self.df)
        rots = torch.empty((n, 3, 3), dtype=torch.float32)
        shift_x = torch.empty((n,), dtype=torch.float32)
        shift_y = torch.empty((n,), dtype=torch.float32)
        for idx in range(n):
            row = self.df.iloc[idx]
            rots[idx].copy_(self._rot(row))
            sx, sy = self._shift_angstrom(row)
            shift_x[idx] = sx
            shift_y[idx] = sy

        self._rot_cache = rots
        self._shift_x_cache = shift_x
        self._shift_y_cache = shift_y
        self._defocus_u_cache = torch.as_tensor(
            self.df["rlnDefocusU"].to_numpy(dtype=np.float32) / 1e4,
            dtype=torch.float32,
        ).reshape(n, 1, 1)
        self._defocus_v_cache = torch.as_tensor(
            self.df["rlnDefocusV"].to_numpy(dtype=np.float32) / 1e4,
            dtype=torch.float32,
        ).reshape(n, 1, 1)
        self._astig_cache = torch.as_tensor(
            np.radians(self.df["rlnDefocusAngle"].to_numpy(dtype=np.float32)),
            dtype=torch.float32,
        ).reshape(n, 1, 1)
        self._orig_idx_cache = torch.as_tensor(
            self.df["__orig_index"].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )
        if self.random_subset_col in self.df.columns:
            random_subset = self.df[self.random_subset_col].to_numpy(dtype=np.int64)
        else:
            random_subset = np.zeros(n, dtype=np.int64)
        self._random_subset_cache = torch.as_tensor(random_subset, dtype=torch.long)
        self._selected_half_cache = torch.full((n,), int(self.half or 0), dtype=torch.long)
        self._parity_label_cache = torch.as_tensor(
            self.df["__orig_index"].to_numpy(dtype=np.int64) % 2,
            dtype=torch.long,
        )
        if "rlnClassNumber" in self.df.columns:
            self._class_id_cache = torch.as_tensor(
                self.df["rlnClassNumber"].to_numpy(dtype=np.int64),
                dtype=torch.long,
            )
        print(f"Precomputed fixed pose/shift/CTF metadata for {n} particles.")

    def cache_normalized_images_inplace(self) -> None:
        """Apply the configured Fourier normalization once to the image cache."""
        if not self.cache_normalized_images:
            return
        if self._image_cache is None:
            raise RuntimeError("cache_normalized_images requires lazy=False image preload.")
        if self._image_cache_is_normalized:
            return
        if self.f_mu is None or self.f_std is None:
            raise RuntimeError("Set/estimate f_mu and f_std before caching normalized images.")
        if float(self.f_std) == 0.0:
            raise RuntimeError("Cannot normalize image cache with f_std=0.")

        # In all current configs f_mu is 0, so the exact Fourier normalization is
        # simply a real-space division. Keep the generic path for completeness.
        if abs(float(self.f_mu)) < 1e-12:
            self._image_cache.div_(float(self.f_std))
        else:
            chunk = self.cache_chunk_size
            for start in range(0, len(self), chunk):
                end = min(start + chunk, len(self))
                fproj = primal_to_fourier_2d(self._image_cache[start:end])
                fproj = (fproj - float(self.f_mu)) / float(self.f_std)
                self._image_cache[start:end].copy_(fourier_to_primal_2d(fproj).real)
        self._image_cache_is_normalized = True
        print("Cached normalized preloaded particle images.")

    def precompute_shifted_targets(self, shift_method: str = "fft") -> None:
        """Cache RELION-shift-corrected training targets for fixed-pose training."""
        if not self.cache_shifted_target:
            return
        if self._shifted_target_cache is not None:
            return
        if self._image_cache is None:
            raise RuntimeError("cache_shifted_target requires lazy=False image preload.")
        if self.f_mu is not None and not self._image_cache_is_normalized:
            self.cache_normalized_images_inplace()
        if self._shift_x_cache is None or self._shift_y_cache is None:
            self.precompute_metadata()

        assert self._shift_x_cache is not None and self._shift_y_cache is not None
        method = str(shift_method).strip().lower()
        if method == "interp":
            translate = SpatialGridTranslate(self.down_size)
        elif method == "fft":
            translate = FourierGridTranslate(self.down_size)
        else:
            raise ValueError(f"Unknown shift_method={shift_method!r}.")

        targets = torch.empty_like(self._image_cache)
        chunk = self.cache_chunk_size
        for start in range(0, len(self), chunk):
            end = min(start + chunk, len(self))
            shifts = torch.stack(
                [
                    self._shift_y_cache[start:end] / float(self.down_apix),
                    self._shift_x_cache[start:end] / float(self.down_apix),
                ],
                dim=-1,
            )[:, None, :]
            images = self._image_cache[start:end]
            if method == "interp":
                shifted = translate.transform(images.squeeze(1), shifts)
            else:
                shifted_fourier = translate.transform(
                    primal_to_fourier_2d(images).squeeze(1), shifts
                )
                shifted = fourier_to_primal_2d(shifted_fourier).real
            targets[start:end].copy_(shifted[:, 0:1])
        self._shifted_target_cache = targets
        print(f"Cached shifted training targets with shift_method={method!r}.")

    def finalize_caches(self, shift_method: str = "fft") -> None:
        """Finalize optional caches after normalization statistics are known."""
        if self.cache_normalized_images:
            self.cache_normalized_images_inplace()
        if self.cache_shifted_target:
            self.precompute_shifted_targets(shift_method=shift_method)

    def _load_image(self, row) -> tuple[torch.Tensor, str]:
        raw = str(row["rlnImageName"])
        mrc_index, mrc_name = raw.split("@")
        mrc_index = int(mrc_index) - 1
        with mrcfile.mmap(osp.join(self.root, mrc_name), mode="r", permissive=True) as m:
            arr = np.array(m.data[mrc_index] if m.data.ndim > 2 else m.data)
        image = torch.from_numpy(arr).float()[None] * float(self.cfg.scale_images)
        if self.down_size != self.size:
            if self.cfg.down_method == "interp":
                image = tvf.resize(image, [self.down_size, self.down_size], antialias=True)
            elif self.cfg.down_method == "fft":
                image = downsample_2d(image[0], self.down_size)[None]
            else:
                raise ValueError(f"Unknown down_method={self.cfg.down_method}")
        if self.mask is not None:
            image = self.mask(image)
        if self.cfg.power_images != 1.0:
            image = image * float(self.cfg.power_images)
        return image, raw

    def _rot(self, row) -> torch.Tensor:
        if self.cfg.ignore_rot:
            return torch.eye(3).float()
        return torch.from_numpy(euler_angles2matrix(
            np.radians(-row["rlnAngleRot"]),
            np.radians(-row["rlnAngleTilt"]),
            np.radians(-row["rlnAnglePsi"]),
        )).float()

    def _shift_angstrom(self, row) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.ignore_shift:
            return torch.tensor([0.0]), torch.tensor([0.0])
        if "rlnOriginXAngst" in row:
            sx = float(row["rlnOriginXAngst"])
            sy = float(row["rlnOriginYAngst"])
        elif "rlnOriginXAngstrom" in row:
            sx = float(row["rlnOriginXAngstrom"])
            sy = float(row["rlnOriginYAngstrom"])
        else:
            sx = float(row.get("rlnOriginX", 0.0)) * self.apix
            sy = float(row.get("rlnOriginY", 0.0)) * self.apix
        return torch.tensor(sx, dtype=torch.float32), torch.tensor(sy, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        row = self.df.iloc[idx]
        if self._image_cache is not None:
            image = self._image_cache[idx]
            assert self._image_name_cache is not None
            raw_name = self._image_name_cache[idx]
        else:
            try:
                image, raw_name = self._load_image(row)
            except Exception as e:
                raw_name = str(row.get("rlnImageName", f"idx={idx}"))
                print(f"WARNING: invalid particle {raw_name}; using zeros. {e}")
                image = torch.zeros(1, self.down_size, self.down_size)

        needs_normalization = self.f_mu is not None and not self._image_cache_is_normalized
        fproj: torch.Tensor
        if needs_normalization:
            if abs(float(self.f_mu)) < 1e-12 and not self.return_fproj:
                # The common cryoSAGE normalization has f_mu=0, so Fourier-domain
                # scaling is exactly real-space division. This avoids a CPU
                # FFT/IFFT per sample when real_mse does not consume fproj.
                image = image / float(self.f_std)
                fproj = torch.empty(0, dtype=torch.complex64)
            else:
                fproj = primal_to_fourier_2d(image)
                fproj = (fproj - self.f_mu) / self.f_std
                image = fourier_to_primal_2d(fproj).real
                if not self.return_fproj:
                    fproj = torch.empty(0, dtype=torch.complex64)
        elif self.return_fproj:
            fproj = primal_to_fourier_2d(image)
        else:
            fproj = torch.empty(0, dtype=torch.complex64)

        if self._rot_cache is not None:
            assert self._shift_x_cache is not None and self._shift_y_cache is not None
            assert self._defocus_u_cache is not None and self._defocus_v_cache is not None
            assert self._astig_cache is not None
            rot = self._rot_cache[idx]
            sx = self._shift_x_cache[idx]
            sy = self._shift_y_cache[idx]
            defocus_u = self._defocus_u_cache[idx]
            defocus_v = self._defocus_v_cache[idx]
            astig = self._astig_cache[idx]
        else:
            rot = self._rot(row)
            sx, sy = self._shift_angstrom(row)
            defocus_u = torch.tensor([[float(row["rlnDefocusU"]) / 1e4]], dtype=torch.float32)
            defocus_v = torch.tensor([[float(row["rlnDefocusV"]) / 1e4]], dtype=torch.float32)
            astig = torch.tensor([[np.radians(float(row["rlnDefocusAngle"]))]], dtype=torch.float32)

        if self._orig_idx_cache is not None:
            orig_idx = self._orig_idx_cache[idx]
            random_subset = self._random_subset_cache[idx] if self._random_subset_cache is not None else torch.tensor(0, dtype=torch.long)
            selected_half = self._selected_half_cache[idx] if self._selected_half_cache is not None else torch.tensor(int(self.half or 0), dtype=torch.long)
            parity_label = self._parity_label_cache[idx] if self._parity_label_cache is not None else torch.tensor(int(row["__orig_index"]) % 2, dtype=torch.long)
        else:
            orig_idx = torch.tensor(int(row.get("__orig_index", idx)), dtype=torch.long)
            random_subset = torch.tensor(int(row.get(self.random_subset_col, 0)) if self.random_subset_col in row else 0, dtype=torch.long)
            selected_half = torch.tensor(int(self.half or 0), dtype=torch.long)
            parity_label = torch.tensor(int(row["__orig_index"]) % 2, dtype=torch.long)

        out = {
            "image": image,
            "rot": rot,
            "shift_x": sx,
            "shift_y": sy,
            "defocus_u": defocus_u,
            "defocus_v": defocus_v,
            "astig_angle": astig,
            # Local contiguous index used for z_table and the CTF module in this run.
            "idx": torch.tensor(idx, dtype=torch.long),
            # Original row index in the unfiltered STAR table; useful for debugging and pairing.
            "orig_idx": orig_idx,
            "random_subset": random_subset,
            # Explicitly record the split assignment used by this run:
            # 0 for full data, 1/2 for selected half data.
            "selected_half": selected_half,
            "parity_label": parity_label,
            "fproj": fproj,
            "image_name": raw_name,
            # Legacy aliases used by CTF/old training code.
            "proj": image,
            "rotmat": rot,
            "shiftX": sx,
            "shiftY": sy,
            "defocusU": defocus_u,
            "defocusV": defocus_v,
            "angleAstigmatism": astig,
            "imgname_raw": raw_name,
        }
        if self._shifted_target_cache is not None:
            out["shifted_target"] = self._shifted_target_cache[idx]

        if self.minimal_batch and self._shifted_target_cache is not None:
            minimal = {
                "shifted_target": out["shifted_target"],
                "rotmat": rot,
                "idx": out["idx"],
                "defocusU": defocus_u,
                "defocusV": defocus_v,
                "angleAstigmatism": astig,
            }
            return minimal

        if self._class_id_cache is not None:
            out["class_id"] = self._class_id_cache[idx]
        elif "rlnClassNumber" in row:
            out["class_id"] = row["rlnClassNumber"]
        return out


# Backward-compatible names for older scripts.
StarfileDatasetConfig = StarDatasetConfig
StarfileDataSet = StarDataset
