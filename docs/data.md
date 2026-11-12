# Particle Input

All examples use a RELION STAR file and one or more MRC particle stacks.
`dataset.size` is the original particle box width in pixels and `dataset.apix`
its pixel spacing in angstroms. Training at `down_size` preserves physical box
width, so the effective pixel spacing becomes `apix * size / down_size`.

Required particle fields:

- `rlnImageName`: one-based image index followed by `@` and the stack path.
- `rlnAngleRot`, `rlnAngleTilt`, `rlnAnglePsi`: known RELION Euler angles in degrees.
- `rlnDefocusU`, `rlnDefocusV`: defocus in angstroms.
- `rlnDefocusAngle`: astigmatism angle in degrees.
- Microscope voltage, spherical aberration and amplitude contrast, supplied as
  `rlnVoltage`, `rlnSphericalAberration`, `rlnAmplitudeContrast` in particle or
  optics metadata. The current CTF wrapper assumes these are common across optics.

Translations may be supplied as `rlnOriginXAngst` / `rlnOriginYAngst`, the
`Angstrom` variants, or pixel-valued `rlnOriginX` / `rlnOriginY`. Missing shifts
default to zero. Optics and particle tables are supported. Supply `dataset.apix`
and `dataset.size` explicitly to avoid ambiguous metadata inference.

Independent-half settings require `rlnRandomSubset` containing 1 and 2. The loader
retains original STAR row indices while assigning contiguous local particle
indices within each half. Do not create the two halves by copying one model's
decoded maps. Choose contrast polarity using `dataset.scale_images` and check
it before comparing reconstructions.

For 10176, the example assumes a cleaned 384-pixel stack/STAR with valid externally
estimated poses. Placeholder zero poses from an unrefined release are not a
replacement. No private STAR conversion, CryoSPARC database or dataset path is
bundled. Optional post-hoc labels must follow original STAR row order.

Example path override:

```bash
python train_hetero.py configs/hetero/10076.py --cfg-options \
  dataset.root=/path/to/10076/data \
  dataset.star=/path/to/10076/data/10076_particles_320_cryodrgn_pose.star
```

Image-stack paths inside the STAR must also be made portable by the dataset
provider. Replacing paths in Python settings does not rewrite STAR contents.
