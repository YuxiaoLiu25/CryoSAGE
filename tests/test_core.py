"""Small CPU checks; no experimental datasets, weights or original package needed."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import mrcfile
from mmengine import Config
import numpy as np
import pandas as pd
import starfile
import torch

from cryosage.dataio import StarDataset, StarDatasetConfig
from cryosage.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
from cryosage.utils.geom_utils import euler_angles2matrix
from cryosage.utils.latent import sample_along_pca
from cryosage.utils.optics import compute_ctf, infer_ctf_params_from_config
from cryosage.utils.transforms import FourierGridTranslate


def write_particles(folder, n=24, size=32):
    folder = Path(folder)
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    images = np.asarray([np.exp(-((xx-.1*(i % 2))**2+yy**2)/.04)+
                         .1*rng.standard_normal((size, size)) for i in range(n)], dtype=np.float32)
    with mrcfile.new(folder/"particles.mrcs", overwrite=True) as handle:
        handle.set_data(images)
        handle.voxel_size = 2.
    table = pd.DataFrame(dict(rlnImageName=[f"{i+1}@particles.mrcs" for i in range(n)],
                             rlnAngleRot=np.zeros(n), rlnAngleTilt=np.zeros(n), rlnAnglePsi=np.zeros(n),
                             rlnOriginXAngst=np.zeros(n), rlnOriginYAngst=np.zeros(n),
                             rlnDefocusU=np.full(n, 12000.), rlnDefocusV=np.full(n, 13000.),
                             rlnDefocusAngle=np.full(n, 15.), rlnVoltage=np.full(n, 300.),
                             rlnSphericalAberration=np.full(n, 2.7), rlnAmplitudeContrast=np.full(n, .1),
                             rlnRandomSubset=1+np.arange(n) % 2))
    # Use a minimal loop fixture; this also works with older pandas/STAR writers.
    header = "data_particles\n\nloop_\n"+"".join(f"_{name}\n" for name in table.columns)
    (folder/"particles.star").write_text(header+table.to_csv(sep=" ", index=False, header=False))
    return folder/"particles.star"


class NumericalTests(unittest.TestCase):
    def test_fourier_roundtrip(self):
        value = torch.randn(2, 1, 32, 32)
        actual = fourier_to_primal_2d(primal_to_fourier_2d(value)).real
        torch.testing.assert_close(actual, value, atol=1e-6, rtol=1e-5)

    def test_ctf_zero_frequency(self):
        params = dict(dfu=torch.tensor([[12000.]]), dfv=torch.tensor([[13000.]]),
                      dfang=torch.tensor([[15.]]), volt=torch.tensor([[300.]]),
                      cs=torch.tensor([[2.7]]), w=torch.tensor([[.1]]))
        value = compute_ctf(torch.zeros(1, 1, 2), **params)
        torch.testing.assert_close(value, torch.tensor([[-.1]]))

    def test_euler_identity(self):
        np.testing.assert_allclose(euler_angles2matrix(0., 0., 0.), np.eye(3))

    def test_zero_fourier_shift(self):
        value = torch.randn(2, 32, 32, dtype=torch.complex64)
        actual = FourierGridTranslate(32).transform(value, torch.zeros(2, 1, 2))
        torch.testing.assert_close(actual[:, 0], value)

    def test_pca_representatives_are_observed(self):
        z = np.random.default_rng(2).normal(size=(50, 4))
        points, indices = sample_along_pca(z, pca_dim=2, num=4)
        self.assertEqual(points.shape, (4, 4))
        np.testing.assert_array_equal(points, z[indices])

    def test_halfsets_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_particles(folder)
            halves = []
            for half in ("half1", "half2"):
                data = StarDataset(StarDatasetConfig(star=str(path), root=folder, apix=2., size=32,
                                                     down_size=32, half=half, half_split_mode="relion"))
                self.assertEqual(len(data), 12)
                halves.append(data.df["__orig_index"].to_numpy())
                self.assertEqual(tuple(data[0]["image"].shape), (1, 32, 32))
            self.assertEqual(len(np.intersect1d(*halves)), 0)
            np.testing.assert_array_equal(np.sort(np.concatenate(halves)), np.arange(24))
            config = Config(dict(dataset_attr=dict(starfile_path=str(path)),
                                 data_process=dict(down_side_shape=32, down_apix=2.)))
            self.assertEqual(infer_ctf_params_from_config(config)["kV"], 300.)

    def test_release_settings(self):
        root = Path(__file__).resolve().parents[1]
        configs = list((root/"configs").rglob("*.py"))
        self.assertEqual(len(configs), 10)
        for file in configs:
            cfg = Config.fromfile(str(file))
            self.assertTrue(cfg.dataset.root.startswith("/path/to/"))
            self.assertTrue(cfg.dataset.star.startswith("/path/to/"))
            self.assertTrue(cfg.dataset["lazy"])
            if cfg.model.hetero:
                self.assertEqual(cfg.eval.pca_steps, 4)
                self.assertGreaterEqual(cfg.eval.umap_num_representatives, 10)
                if cfg.model.deform_attributes == "xyz":
                    self.assertEqual(cfg.model.deform_decoder, "joint")


if __name__ == "__main__":
    unittest.main()
