"""Check a source tree for private machine paths and accidental large artifacts."""
from __future__ import annotations

import ast
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parents[1]
    ignored = {".git", "__pycache__", "work_dirs", "lightning_logs", "build", "dist", ".pytest_cache"}
    private_path = re.compile(r"(?<![A-Za-z0-9_./-])/(?:data[0-9]*|home|mnt|scratch|media)/")
    failures, files, total = [], 0, 0
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in ignored or part.endswith(".egg-info") for part in relative.parts) or not path.is_file():
            continue
        files += 1
        total += path.stat().st_size
        if path.suffix in {".so", ".o", ".pt", ".ckpt", ".mrc", ".mrcs", ".pkl", ".npy", ".npz"}:
            failures.append(f"Unexpected binary/data artifact: {relative}")
        try:
            text = path.read_text()
        except UnicodeError:
            failures.append(f"Unexpected binary file: {relative}")
            continue
        if private_path.search(text):
            failures.append(f"Private absolute machine path: {relative}")
        if path.suffix == ".py":
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "cryostar":
                    failures.append(f"Runtime dependency on old package: {relative}")
                if isinstance(node, ast.Import) and any(item.name.split(".")[0] == "cryostar" for item in node.names):
                    failures.append(f"Runtime dependency on old package: {relative}")
            if "configs" in relative.parts:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        if node.value.startswith("/") and not node.value.startswith("/path/to/"):
                            failures.append(f"Non-placeholder absolute config path: {relative}")
    for required in ("LICENSE", "NOTICE", "provenance.json",
                     "third_party/xray_gaussian_rasterization_voxelization/LICENSE.md",
                     "third_party/xray_gaussian_rasterization_voxelization/third_party/glm/copying.txt"):
        if not (root/required).is_file():
            failures.append(f"Missing license/provenance file: {required}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {files} source files, {total/2**20:.2f} MiB; no private paths, data/weight binaries or old-package imports")


if __name__ == "__main__":
    main()
