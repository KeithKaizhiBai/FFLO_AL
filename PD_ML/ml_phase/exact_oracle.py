from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from eta_phase_diagram_cuda import (
    EtaPhaseConfig,
    build_q_vec,
    compute_current_from_omega,
    compute_omega_min_q_batch,
    find_eta_from_jq,
    maybe_set_linalg_backend,
)


@dataclass
class OracleResult:
    kT: np.ndarray
    JA: np.ndarray
    eta: np.ndarray
    q_opt: np.ndarray
    delta_opt: np.ndarray
    ic_plus: np.ndarray
    ic_minus: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        return asdict(self)


def _device_from_arg(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def evaluate_points(
    points: np.ndarray,
    cfg: EtaPhaseConfig | None = None,
    device: str | torch.device | None = None,
    save_every: int = 1,
    output_file: Path | None = None,
) -> OracleResult:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be shape (n, 2) with columns [kT, JA]")

    raw_cfg = cfg if cfg is not None else EtaPhaseConfig()
    cfg_scaled = raw_cfg.scaled()
    device_obj = torch.device(device) if isinstance(device, str) else (device or _device_from_arg(None))
    maybe_set_linalg_backend(cfg_scaled)

    q_vec = build_q_vec(cfg_scaled)
    q_vec_t = torch.as_tensor(q_vec, device=device_obj, dtype=cfg_scaled.dtype)
    k_vec = torch.linspace(-math.pi, math.pi, cfg_scaled.n_k, dtype=cfg_scaled.dtype, device=device_obj)

    out_kt: list[float] = []
    out_ja: list[float] = []
    out_eta: list[float] = []
    out_q_opt: list[float] = []
    out_delta_opt: list[float] = []
    out_ip: list[float] = []
    out_im: list[float] = []

    def flush_partial() -> None:
        if output_file is None:
            return
        payload = {
            "kT": np.asarray(out_kt, dtype=np.float64),
            "JA": np.asarray(out_ja, dtype=np.float64),
            "eta": np.asarray(out_eta, dtype=np.float64),
            "q_opt": np.asarray(out_q_opt, dtype=np.float64),
            "delta_opt": np.asarray(out_delta_opt, dtype=np.float64),
            "ic_plus": np.asarray(out_ip, dtype=np.float64),
            "ic_minus": np.asarray(out_im, dtype=np.float64),
            "completed_points": np.asarray([len(out_kt)], dtype=np.int64),
        }
        np.savez(output_file, **payload)

    for i, (kT, JA) in enumerate(points):
        kt_batch = torch.as_tensor([float(kT)], dtype=cfg_scaled.dtype, device=device_obj)
        ja_batch = torch.as_tensor([float(JA)], dtype=cfg_scaled.dtype, device=device_obj)
        omega_min_q_t, _delta_opt_q_t, q_opt_t, delta_opt_t = compute_omega_min_q_batch(
            kt_batch,
            ja_batch,
            cfg_scaled,
            k_vec,
            q_vec_t,
        )

        omega_min_q = omega_min_q_t.detach().cpu().numpy()
        q_opt = float(q_opt_t.detach().cpu().numpy()[0, 0])
        delta_opt = float(delta_opt_t.detach().cpu().numpy()[0, 0])
        j_q = compute_current_from_omega(omega_min_q, q_vec)[0, 0]
        iq_opt = int(np.argmin(np.abs(q_vec - q_opt)))
        eta, ic_plus, ic_minus = find_eta_from_jq(j_q, q_vec, iq_opt)

        out_kt.append(float(kT))
        out_ja.append(float(JA))
        out_eta.append(float(eta))
        out_q_opt.append(float(q_opt))
        out_delta_opt.append(float(delta_opt))
        out_ip.append(float(ic_plus))
        out_im.append(float(ic_minus))

        if save_every > 0 and output_file is not None and ((i + 1) % save_every == 0):
            flush_partial()

    flush_partial()
    return OracleResult(
        kT=np.asarray(out_kt, dtype=np.float64),
        JA=np.asarray(out_ja, dtype=np.float64),
        eta=np.asarray(out_eta, dtype=np.float64),
        q_opt=np.asarray(out_q_opt, dtype=np.float64),
        delta_opt=np.asarray(out_delta_opt, dtype=np.float64),
        ic_plus=np.asarray(out_ip, dtype=np.float64),
        ic_minus=np.asarray(out_im, dtype=np.float64),
    )


def _infer_rank_and_world_size(rank: int | None, world_size: int | None) -> tuple[int, int]:
    if rank is not None and world_size is not None:
        return rank, world_size
    if "SLURM_ARRAY_TASK_ID" in os.environ and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        return int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    return (rank or 0), (world_size or 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact pointwise BdG oracle for active-learning refinement.")
    p.add_argument("--points-file", type=Path, default=None, help="CSV file with columns kT, JA.")
    p.add_argument("--output-file", type=Path, default=None, help="Output npz path for shard result.")
    p.add_argument("--device", type=str, default=None, help="Torch device, e.g., cuda:0 or cpu.")
    p.add_argument("--save-every", type=int, default=1, help="Flush partial output every N points.")

    p.add_argument("--run-id", type=str, default=None, help="Run id under ML_Phase/active_runs.")
    p.add_argument("--iteration", type=int, default=None, help="Iteration index for default shard paths.")
    p.add_argument("--rank", type=int, default=None, help="Rank index.")
    p.add_argument("--world-size", type=int, default=None, help="Total ranks.")
    p.add_argument("--active-root", type=Path, default=Path("ML_Phase/active_runs"), help="Active runs root.")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace, rank: int, world_size: int) -> tuple[Path, Path]:
    if args.points_file is not None and args.output_file is not None:
        return args.points_file, args.output_file
    if args.run_id is None or args.iteration is None:
        raise ValueError("Either --points-file/--output-file or --run-id/--iteration must be provided.")
    iter_dir = args.active_root / args.run_id / f"iter{args.iteration:03d}"
    points_file = iter_dir / f"selected_points_rank{rank:03d}_of{world_size:03d}.csv"
    output_file = iter_dir / f"exact_shard_rank{rank:03d}_of{world_size:03d}.npz"
    return points_file, output_file


def main() -> None:
    args = _parse_args()
    rank, world_size = _infer_rank_and_world_size(args.rank, args.world_size)
    points_file, output_file = _resolve_paths(args, rank, world_size)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    points_df = pd.read_csv(points_file)
    points = points_df[["kT", "JA"]].to_numpy(dtype=np.float64)

    t0 = time.perf_counter()
    result = evaluate_points(
        points=points,
        cfg=EtaPhaseConfig(),
        device=args.device,
        save_every=max(1, int(args.save_every)),
        output_file=output_file,
    )
    elapsed = time.perf_counter() - t0

    # rewrite once with metadata fields appended
    payload: Dict[str, Any] = result.to_dict()
    payload["rank"] = np.asarray([rank], dtype=np.int64)
    payload["world_size"] = np.asarray([world_size], dtype=np.int64)
    payload["elapsed_sec"] = np.asarray([elapsed], dtype=np.float64)
    payload["hostname"] = np.asarray([socket.gethostname()])
    payload["device"] = np.asarray([str(args.device or _device_from_arg(None))])
    payload["cuda_visible_devices"] = np.asarray([os.environ.get("CUDA_VISIBLE_DEVICES", "")])
    np.savez(output_file, **payload)

    meta_path = output_file.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "points_file": str(points_file),
                "output_file": str(output_file),
                "n_points": int(points.shape[0]),
                "elapsed_sec": elapsed,
                "hostname": socket.gethostname(),
                "device": str(args.device or _device_from_arg(None)),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Oracle finished rank {rank}/{world_size} with {points.shape[0]} points in {elapsed:.2f}s")
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()

