"""Generate tiny temporary inputs and exercise actual GPU training/export."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from mmengine import Config
import numpy as np

from test_core import write_particles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--mode", choices=("homo", "hetero", "xyz", "all"), default="all")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="cryosage-smoke-") as directory:
        folder = Path(directory)
        star = write_particles(folder)
        modes = ("homo", "hetero", "xyz") if args.mode == "all" else (args.mode,)
        for mode in modes:
            template = repo/("configs/homo/10049.py" if mode == "homo" else
                             "configs/hetero/10180.py" if mode == "xyz" else "configs/hetero/10076.py")
            cfg = Config.fromfile(str(template))
            cfg.dataset.update(root=str(folder), star=str(star), size=32, down_size=32, apix=2.,
                               f_mu=0., f_std=1., half="all", lazy=True, scale_images=1.)
            cfg.model.num_points = 32
            cfg.model.init_scale = .04
            cfg.exp_name = "smoke_"+mode
            cfg.loader.update(batch_size=4, num_workers=0, persistent_workers=False)
            cfg.trainer.update(devices=args.devices, max_epochs=1, precision="32-true")
            cfg.eval.every_n_epochs = 1
            if mode != "homo":
                cfg.model.update(z_dim=3, deform_hidden=16, deform_layers=2, deform_pe_freqs=1)
                cfg.model.update(occupancy_factor_dim=4, occupancy_factor_hidden=8)
                cfg.vae.update(encoder_hidden=16, encoder_layers=2, export_batch_size=8)
                cfg.eval.update(umap_n_neighbors=5, umap_num_representatives=10, pca_dims=2, pca_steps=4)
            path = folder/f"{mode}.py"
            cfg.dump(str(path))
            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo)+os.pathsep+env.get("PYTHONPATH", "")
            subprocess.run([sys.executable, str(repo/("train_homo.py" if mode == "homo" else "train.py")),
                            str(path)], cwd=folder, env=env, check=True)
            epochs = sorted((folder/"work_dirs").glob(f"smoke_{mode}*/epoch*"))
            if len(epochs) != 1:
                raise AssertionError(f"No unique exported checkpoint for {mode}")
            output = epochs[0]
            assert (output/"model_state.pt").is_file()
            assert (output/"vol_consensus.mrc").is_file()
            if mode != "homo":
                assert np.load(output/"z_mu.npy").shape == (24, 3)
                assert np.load(output/"z_umap.npy").shape == (24, 2)
                if len(list(output.glob("vol_umap_*.mrc"))) < 10:
                    raise AssertionError("UMAP representative volume export failed")
                if len(list(output.glob("vol_pca_*.mrc"))) != 8:
                    raise AssertionError("PCA volume export failed")
            print(f"PASS: {mode}, {args.devices} GPU(s), training and output checks", flush=True)


if __name__ == "__main__":
    main()
