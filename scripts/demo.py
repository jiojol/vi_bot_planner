"""Replay one recorded simulation observation through the released navigator."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from navigation.runtime import Navigator

ROOT = Path(__file__).resolve().parents[1]
MODES = ("s1_astar_v4", "s1_guided_flow_25d_full")
FIXTURE_SEED = 2026125380


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default=MODES[0])
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.set_num_threads(3)
    config = json.loads((ROOT / "configs/navigator_config.json").read_text())
    checkpoint = ROOT / config["checkpoint"]
    if not checkpoint.is_file():
        parser.error("Missing model.pt. Run python -m scripts.fetch_weights first.")

    with np.load(ROOT / "examples/oncoming_step0.npz", allow_pickle=False) as frame:
        observation = {key: frame[key] for key in frame.files if key != "goal"}
        goal = frame["goal"]

    nav = Navigator(config, checkpoint, ROOT / config["ground_head"],
                    device=args.device, seed=FIXTURE_SEED, mode=args.mode)
    result = nav.command(observation, goal)
    trajectory = np.asarray(result["neural_trajectory"])
    command = [float(result[key]) for key in ("vx", "vy", "yaw_rate")]
    if trajectory.shape != (10, 4) or not np.isfinite(trajectory).all():
        raise RuntimeError("The navigator did not return a finite 10×4 trajectory")
    if not np.isfinite(command).all():
        raise RuntimeError("The navigator returned a non-finite command")

    print(json.dumps({
        "mode": args.mode,
        "device": args.device,
        "vx": command[0],
        "vy": command[1],
        "yaw_rate": command[2],
        "trajectory_xy_sin_yaw_cos_yaw": trajectory.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
