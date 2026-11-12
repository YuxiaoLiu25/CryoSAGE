# CryoSAGE

CryoSAGE reconstructs cryo-EM density from pose-annotated particle images using
anisotropic Gaussian primitives. Homogeneous reconstruction optimizes a shared
Gaussian volume. Heterogeneous reconstruction uses particle latents to predict
position offsets and, when enabled, factorized occupancy weights.

The reference scaffold is randomly initialized and trainable. Particle rotations,
translations and CTF metadata are supplied in a RELION STAR file. Template-free
initialization does not mean that particle poses are estimated by this package.

## Installation

Use Python 3.9 or newer, a CUDA-capable NVIDIA GPU, and a CUDA compiler compatible
with your PyTorch installation. Install the appropriate PyTorch/torchvision builds
for your GPU environment before building the extension.

```bash
cd /path/to/cryosage
python -m pip install -e .
python -m pip install --no-build-isolation ./third_party/xray_gaussian_rasterization_voxelization
python -m unittest discover -s tests -v
```

The extension is built from the bundled CUDA sources. Precompiled libraries,
datasets, learned weights and local environment directories are not distributed.
`requirements-tested.txt` records reference Python dependency versions; it does
not select a CUDA toolkit or download a particular PyTorch CUDA build.

## Data And Configuration

Every dataset path in the example settings starts with `/path/to/`. Replace the
paths or override them on the command line. Keep the STAR image names consistent
with `dataset.root`; relative stack paths are resolved under that directory.
See [data format](docs/data.md) for required metadata.

All supplied settings train at the input image size without downsampling.
Homogeneous settings use 32,000 Gaussians for 20 epochs. Heterogeneous settings
use 64,000 Gaussians and an eight-dimensional latent space for 30 epochs.

Training hyperparameters are shared within each reconstruction mode. The 10180
setting uses position-only dynamics; 10176 and 10076 use position offsets and
factorized occupancy. Dataset-specific normalization and renderer gains are
retained. `scale_images=-1` is retained for 10076; the other supplied datasets use
positive contrast.

### Homogeneous Reconstruction

```bash
CUDA_VISIBLE_DEVICES=0 python train_homo.py configs/homo/10049.py
CUDA_VISIBLE_DEVICES=0 python train_homo.py configs/homo/10076.py
```

Or using multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_homo.py configs/homo/10049.py --cfg-options trainer.devices=2
```

Independent-half settings are supplied as `10049_half1.py`, `10049_half2.py`,
`10076_half1.py` and `10076_half2.py` under `configs/homo/`:

```bash
CUDA_VISIBLE_DEVICES=0 python train_homo.py configs/homo/10049_half1.py
CUDA_VISIBLE_DEVICES=1 python train_homo.py configs/homo/10049_half2.py
```

Replace the placeholder data paths first. Half selection uses `rlnRandomSubset`
and distinct random seeds (43 for half1 and 44 for half2). Full-data settings use
seed 42. Never use two states decoded by one model as an independent half-map pair.

### Heterogeneous Reconstruction

```bash
CUDA_VISIBLE_DEVICES=0 python train_hetero.py configs/hetero/10176.py
CUDA_VISIBLE_DEVICES=0 python train_hetero.py configs/hetero/10076.py
CUDA_VISIBLE_DEVICES=0 python train_hetero.py configs/hetero/10180.py
```

Or using multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_hetero.py configs/hetero/10176.py --cfg-options trainer.devices=2
```

All settings default to a single GPU with four particles per GPU. For multiple
GPUs, set `trainer.devices` to the number of GPUs listed in `CUDA_VISIBLE_DEVICES`;
the same override works with any supplied setting. `loader.batch_size` is per
GPU, not the effective global batch size. Within each mode, the examples share
optimizer settings, network widths, positional encoding, loss settings and shape
bounds; occupancy-specific
settings apply only when occupancy is enabled. Evaluation defaults export four
maps on each of two PCA axes and at least ten observed representatives selected
from UMAP. UMAP and PCA exports can be disabled separately.

Published labels are not needed for training. The optional
`configs/hetero/10076_labeled.py` enables post-hoc class/fine-class validation;
it requires additional label files whose paths must also be configured.

## Outputs

Each run writes a new directory under `work_dirs/`, containing its resolved
configuration and source snapshots. Export epochs contain `model_state.pt` and
density maps. Heterogeneous runs also export `z_mu.npy`, `z_logvar.npy`, UMAP/PCA
coordinates, representative maps and the corresponding particle-index tables.
Training resumes or state decoding must use the same architecture and settings.

The defaults preserve the original numerical renderer. Increasing map grid size
does not automatically increase physical resolution. Half-map FSC additionally
requires independent data/model halves, matched states, coordinates and pixel
sizes; unmasked/native map-export artifacts should be checked separately.

## Project Layout

```text
cryosage/          Training, STAR dataset, optics and numerical utilities
configs/homo/     Full-data and independent-half homogeneous settings
configs/hetero/   Dataset-specific heterogeneous settings
third_party/      Separately licensed CUDA rasterization/voxelization sources
tests/            Small synthetic-data and numerical checks
```

## Attribution And License

CryoSAGE is a separately organized research codebase with selected utilities
adapted from CryoSTAR. See [NOTICE](NOTICE) and `provenance.json` for the source
history and modifications. The main Python package retains Apache-2.0 terms.
The bundled Gaussian CUDA backend has separate research/non-commercial terms;
its license and original notices are included and are not replaced by the root
license. No GitHub remote or publication-specific citation metadata is assumed.
