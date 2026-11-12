# Release Validation

The release was checked on Linux with Python 3.9, PyTorch 2.8 and NVIDIA RTX 3090
GPUs. These are execution checks on tiny synthetic inputs, not accuracy or
performance benchmarks on the five experimental datasets.

Completed checks:

- Eight CPU unit checks: centered FFT round-trip, CTF at zero frequency, RELION
  rotation identity, zero translation, PCA observed representatives, disjoint
  STAR half subsets, all ten example settings, and controlled training settings
  with an explicit position-only exception for 10180.
- Bundled CUDA sources compiled successfully into a separate temporary build.
  Training smoke tests used that new build, not a replacement installed into the
  working research environment.
- One-GPU homogeneous training: one epoch with 24 generated 32-pixel particles
  and 32 Gaussian primitives, checkpoint and consensus map verified.
- One-GPU factorized occupancy and position-only heterogeneous training: same
  small inputs, checkpoint, latent coordinates, UMAP, ten UMAP representative
  maps and eight PCA maps verified for each mode.
- Two-GPU DDP factorized occupancy training: same data and output checks passed.
- Both training CLIs provide their configuration and override help.
- A Python wheel was built and checked for the two trainer modules and correct
  `cryosage-recon` project metadata.
- Source audit checks for machine-specific absolute paths, accidental model/data
  binaries and imports of the original Python package.

To repeat the CPU checks:

```bash
python -m unittest discover -s tests -v
python tools/audit_release.py
```

To repeat the GPU smoke checks after building the extension:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/smoke_train.py
CUDA_VISIBLE_DEVICES=0,1 python tests/smoke_train.py --devices 2 --mode hetero
```

Temporary inputs and outputs are generated automatically and removed when the
test exits. No public data download, pretrained weight or private STAR file is
required. CPU-only machines can run the numerical tests, but cannot run Gaussian
rasterization or reconstruction training. Full experimental retraining, a clean
OS/container installation and execution of the hosted GitHub workflow have not
been performed as part of this local release check.
