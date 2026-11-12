"""Particle CTF evaluation in physical units and shared STAR microscope metadata."""
from __future__ import annotations

import numpy as np
import starfile
import torch


def compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift=None, bfactor=None):
    """Return the conventional sine/amplitude-contrast CTF.

    Frequencies are inverse angstroms, defocus is in angstroms, voltage in kV,
    spherical aberration in mm, and both angle inputs in degrees. The caller
    controls the image-contrast sign convention.
    """
    x, y = freqs.unbind(-1)
    radius_squared = x.square()+y.square()
    voltage = volt*1000.
    wavelength = 12.2639/torch.sqrt(voltage+0.97845e-6*voltage.square())
    angle = torch.atan2(y, x)-torch.deg2rad(dfang)
    defocus = (dfu+dfv+(dfu-dfv)*torch.cos(2*angle))*.5
    phase = 2*torch.pi*(-.5*defocus*wavelength*radius_squared+
                        .25*(cs*1e7)*wavelength.pow(3)*radius_squared.square())
    if phase_shift is not None:
        phase = phase-torch.deg2rad(phase_shift)
    transfer = torch.sqrt(1-w.square())*torch.sin(phase)-w*torch.cos(phase)
    if bfactor is not None:
        transfer = transfer*torch.exp(-.25*bfactor*radius_squared)
    return transfer


def infer_ctf_params_from_config(cfg):
    """Check that all optics groups share the microscope parameters of one CTF."""
    metadata = starfile.read(cfg.dataset_attr.starfile_path)
    if isinstance(metadata, dict):
        if "optics" in metadata:
            table = metadata["optics"]
        elif "particles" in metadata:
            table = metadata["particles"]
        else:
            tables = [item for item in metadata.values() if hasattr(item, "columns")]
            if len(tables) != 1:
                raise ValueError("Cannot identify microscope metadata in STAR file")
            table = tables[0]
    else:
        table = metadata
    result = {"size": cfg.data_process.down_side_shape, "resolution": cfg.data_process.down_apix}
    for column, key in (("rlnVoltage", "kV"), ("rlnSphericalAberration", "cs"),
                        ("rlnAmplitudeContrast", "amplitudeContrast")):
        if column not in table or len(table) == 0:
            raise ValueError(f"Missing STAR microscope metadata: {column}")
        values = table[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values, values[0], atol=1e-6, rtol=0):
            raise ValueError(f"The configured single CTF requires common {column} across particles/optics groups")
        result[key] = float(values[0])
    return result
