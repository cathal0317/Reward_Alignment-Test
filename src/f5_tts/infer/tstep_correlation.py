from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


MAIN_CONFIG = {
    # Directory used in `midstep_ssim_experiment.py`
    "output_dir": "inference_output4",
}


def run_correlation_analysis() -> None:
    cfg = MAIN_CONFIG

    repo_root = Path(__file__).resolve().parents[3]  # .../Reward_Alignment-Test
    output_root = repo_root / cfg["output_dir"]

    if not output_root.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_root}")

    # Collect all per-run S-SIM vs t files
    npy_paths: List[Path] = sorted(output_root.glob("run*_ssim_vs_t.npy"))
    if not npy_paths:
        raise FileNotFoundError(
            f"No 'run*_ssim_vs_t.npy' files found in {output_root}. "
            "Run midstep_ssim_experiment.py with num_runs > 1 first."
        )

    print(f"[INFO] Found {len(npy_paths)} runs in {output_root}")

    # Load first file to get common t indices
    first = np.load(npy_paths[0])
    # first: [K, 2] with columns [t_idx, ssim_t]
    t_arr = first[:, 0].astype(np.int32)
    K = t_arr.shape[0]

    sims_t = np.zeros((len(npy_paths), K), dtype=np.float32)
    sim_final = np.zeros(len(npy_paths), dtype=np.float32)

    for run_idx, path in enumerate(npy_paths):
        data = np.load(path)  # [K, 2]
        t_arr_run = data[:, 0].astype(np.int32)
        ssim_arr = data[:, 1].astype(np.float32)

        if t_arr_run.shape[0] != K or not np.all(t_arr_run == t_arr):
            raise ValueError(
                f"Timestep indices in {path.name} do not match those in {npy_paths[0].name}"
            )

        sims_t[run_idx] = ssim_arr
        sim_final[run_idx] = ssim_arr[-1]  # treat last timestep as "final" S-SIM

        print(
            f"[RUN {run_idx}] loaded {path.name}, "
            f"min S-SIM={ssim_arr.min():.4f}, max S-SIM={ssim_arr.max():.4f}, "
            f"final S-SIM={sim_final[run_idx]:.4f}"
        )

    # Compute correlation per timestep
    corrs = np.zeros(K, dtype=np.float32)
    for k in range(K):
        x = sims_t[:, k]
        y = sim_final

        if np.allclose(x, x[0]):
            # all identical → variance zero → correlation undefined; set to 0
            corr = 0.0
        else:
            C = np.corrcoef(x, y)
            corr = float(C[0, 1])
        corrs[k] = corr

    # Save results
    out_npy = output_root / "corr_vs_t.npy"
    np.save(out_npy, np.stack([t_arr.astype(np.float32), corrs], axis=1))
    print(f"[INFO] Saved correlation vs t → {out_npy}")

    # Plot correlation vs timestep index
    plt.figure(figsize=(6, 4))
    plt.plot(t_arr, corrs, marker="o")
    plt.xlabel("ODE step index (t_idx)")
    plt.ylabel("Corr(S-SIM_t, S-SIM_final)")
    plt.title("Timestep-wise correlation between intermediate and final S-SIM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = output_root / "corr_vs_t.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot → {out_png}")


def main() -> None:
    run_correlation_analysis()


if __name__ == "__main__":
    main()


