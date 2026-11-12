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

| Setting | Input / Training Box | Gaussian Count | Latent | Dynamics | Epochs |
|---|---|---:|---:|---|---:|
| `configs/hetero/10176.py` | 384 / 192 | 32,768 | 8 | position + factorized occupancy | 30 |
| `configs/hetero/10076.py` | 320 / 320 | 65,536 | 10 | position + factorized occupancy | 30 |
| `configs/hetero/10180.py` | 320 / 320 | 65,536 | 8 | position only | 15 |
| `configs/homo/10049.py` | 192 / 192 | 65,536 | none | shared volume | 20 |
| `configs/homo/10076.py` | 320 / 320 | 65,536 | none | shared volume | 20 |

The 10180 position-only setting preserves its existing experiment rather than
claiming an untested occupancy run. The renderer and hetero trainer support
factorized occupancy as used by 10176 and 10076. `scale_images=-1` is retained
for 10076; the other supplied datasets use positive contrast.

### Homogeneous Reconstruction

```bash
CUDA_VISIBLE_DEVICES=0 python train_homo.py configs/homo/10049.py \
  --cfg-options trainer.devices=1 dataset.root=/path/to/10049/data \
  dataset.star=/path/to/10049/data/Refine3D_10049/job001/run_data.star
```

Independent-half settings are supplied as `10049_half1.py`, `10049_half2.py`,
`10076_half1.py` and `10076_half2.py` under `configs/homo/`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_homo.py configs/homo/10049_half1.py --cfg-options trainer.devices=2
CUDA_VISIBLE_DEVICES=2,3 python train_homo.py configs/homo/10049_half2.py --cfg-options trainer.devices=2
```

Replace the placeholder data paths first. Half selection uses `rlnRandomSubset`
and distinct random seeds. Never use two states decoded by one model as an
independent half-map pair.

### Heterogeneous Reconstruction

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_hetero.py configs/hetero/10176.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_hetero.py configs/hetero/10076.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_hetero.py configs/hetero/10180.py
```

`--cfg-options trainer.devices=1` selects a one-GPU run. `loader.batch_size` is per
GPU, not the effective global batch size. The examples retain their existing
optimizer, loss, shape bounds and dataset-specific training schedules. Evaluation
defaults export four maps on each of two PCA axes and at least ten observed
representatives selected from UMAP. UMAP and PCA exports can be disabled separately.

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
