from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.metrics import denorm, gap_indices, mape, mse, r2


torch = None
DataLoader = None
TCNResidualImputer = None
fit_predict_fourier = None
fit_predict_fourier_dynamic = None
load_star = None
choose_big_gap_points = None
FixedSpecsDataset = None
FixedTrainDataset = None
build_star_split = None
collate_fn = None


def load_runtime_dependencies() -> None:
    global torch
    global DataLoader
    global TCNResidualImputer
    global fit_predict_fourier
    global fit_predict_fourier_dynamic
    global load_star
    global choose_big_gap_points
    global FixedSpecsDataset
    global FixedTrainDataset
    global build_star_split
    global collate_fn

    import torch as torch_module
    from torch.utils.data import DataLoader as DataLoader_class

    from src.fourier_baseline import fit_predict_fourier as fit_predict_fourier_fn
    from src.fourier_dynamic import fit_predict_fourier_dynamic as fit_predict_fourier_dynamic_fn
    from src.io_gaps import load_star as load_star_fn
    from src.models.tcn_residual import TCNResidualImputer as TCNResidualImputer_class
    from src.protocol import choose_big_gap_points as choose_big_gap_points_fn
    from src.segment_split import (
        FixedSpecsDataset as FixedSpecsDataset_class,
        FixedTrainDataset as FixedTrainDataset_class,
        build_star_split as build_star_split_fn,
        collate_fn as collate_fn_fn,
    )

    torch = torch_module
    DataLoader = DataLoader_class
    TCNResidualImputer = TCNResidualImputer_class
    fit_predict_fourier = fit_predict_fourier_fn
    fit_predict_fourier_dynamic = fit_predict_fourier_dynamic_fn
    load_star = load_star_fn
    choose_big_gap_points = choose_big_gap_points_fn
    FixedSpecsDataset = FixedSpecsDataset_class
    FixedTrainDataset = FixedTrainDataset_class
    build_star_split = build_star_split_fn
    collate_fn = collate_fn_fn


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mse(y_true: torch.Tensor, y_pred: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    miss = (obs_mask < 0.5).float()
    denom = miss.sum().clamp(min=1.0)
    return ((y_true - y_pred) ** 2 * miss).sum() / denom


def build_fourier_aware_x(batch: dict, device: str) -> torch.Tensor:
    return torch.stack(
        [
            batch["y_obs_zeroed"].to(device),
            batch["y_fourier_full"].to(device),
            batch["obs_mask"].to(device),
            batch["residual_visible"].to(device),
        ],
        dim=1,
    )


def forward_delta(model: TCNResidualImputer, batch: dict, device: str, input_mode: str) -> torch.Tensor:
    if input_mode == "fourier_aware":
        return model(build_fourier_aware_x(batch, device))
    return model(batch["y_in"].to(device), batch["obs_mask"].to(device))


def prediction_and_loss(
    model: TCNResidualImputer,
    batch: dict,
    device: str,
    input_mode: str,
    lambda_visible_residual: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_true = batch["y_true"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    delta = forward_delta(model, batch, device, input_mode)

    if input_mode == "fourier_aware":
        y_hat = batch["y_fourier_full"].to(device) + delta * (1.0 - obs_mask)
        loss = masked_mse(y_true, y_hat, obs_mask)
        if lambda_visible_residual > 0.0:
            visible = obs_mask
            target_residual = batch["residual_visible"].to(device)
            denom = visible.sum().clamp(min=1.0)
            loss_visible = ((delta - target_residual) ** 2 * visible).sum() / denom
            loss = loss + lambda_visible_residual * loss_visible
        return y_hat, loss

    y_in = batch["y_in"].to(device)
    y_hat = y_in + delta * (1.0 - obs_mask)
    return y_hat, masked_mse(y_true, y_hat, obs_mask)


def compute_fourier_batch(batch: dict, fourier, fourier_mode: str) -> np.ndarray:
    y_true_n = batch["y_true"].cpu().numpy()
    obs_mask = batch["obs_mask"].cpu().numpy()
    y_fourier_n = np.empty_like(y_true_n)
    for i in range(y_true_n.shape[0]):
        gs = int(batch["gap_start"][i])
        gl = int(batch["gap_len"][i])
        if fourier_mode == "dynamic":
            y_fourier_n[i] = fit_predict_fourier_dynamic(
                fourier.A_full,
                y_true_n[i],
                obs_mask[i],
                gap_start=gs,
                gap_len=gl,
            )
        else:
            y_fourier_n[i] = fit_predict_fourier(fourier.A_full, y_true_n[i], obs_mask[i])
    return y_fourier_n


def predict_loader_standard(
    model: TCNResidualImputer,
    loader: DataLoader,
    device: str,
    fourier,
    fourier_mode: str,
) -> dict:
    model.eval()
    chunks = {"y_true": [], "y_pred": [], "y_fourier": [], "gap_start": [], "gap_len": []}

    with torch.no_grad():
        for batch in loader:
            y_true_n = batch["y_true"].cpu().numpy()
            obs_mask = batch["obs_mask"].cpu().numpy()
            mu = batch["mu"].cpu().numpy()
            sd = batch["sd"].cpu().numpy()

            y_fourier_n = compute_fourier_batch(batch, fourier, fourier_mode)
            delta = forward_delta(model, batch, device, input_mode="standard").cpu().numpy()
            y_hat_n = batch["y_in"].cpu().numpy() + delta * (1.0 - obs_mask)

            chunks["y_true"].append(denorm(y_true_n, mu[:, None], sd[:, None]))
            chunks["y_pred"].append(denorm(y_hat_n, mu[:, None], sd[:, None]))
            chunks["y_fourier"].append(denorm(y_fourier_n, mu[:, None], sd[:, None]))
            chunks["gap_start"].append(batch["gap_start"].cpu().numpy())
            chunks["gap_len"].append(batch["gap_len"].cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def predict_loader_fourier_aware(
    model: TCNResidualImputer,
    loader: DataLoader,
    device: str,
) -> dict:
    model.eval()
    chunks = {"y_true": [], "y_pred": [], "y_fourier": [], "gap_start": [], "gap_len": []}

    with torch.no_grad():
        for batch in loader:
            y_true_n = batch["y_true"].cpu().numpy()
            obs_mask = batch["obs_mask"].cpu().numpy()
            mu = batch["mu"].cpu().numpy()
            sd = batch["sd"].cpu().numpy()

            y_fourier_n = batch["y_fourier_full"].cpu().numpy()
            delta = forward_delta(model, batch, device, input_mode="fourier_aware").cpu().numpy()
            y_hat_n = y_fourier_n + delta * (1.0 - obs_mask)

            chunks["y_true"].append(denorm(y_true_n, mu[:, None], sd[:, None]))
            chunks["y_pred"].append(denorm(y_hat_n, mu[:, None], sd[:, None]))
            chunks["y_fourier"].append(denorm(y_fourier_n, mu[:, None], sd[:, None]))
            chunks["gap_start"].append(batch["gap_start"].cpu().numpy())
            chunks["gap_len"].append(batch["gap_len"].cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def metric_triplet_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, gap_start: np.ndarray, gap_len: np.ndarray) -> dict:
    mse_list, mape_list, r2_list = [], [], []
    L = int(y_true.shape[1])
    for i in range(y_true.shape[0]):
        gi = gap_indices(L, int(gap_start[i]), int(gap_len[i]))
        yt = y_true[i][gi]
        yp = y_pred[i][gi]
        mse_list.append(mse(yt, yp))
        mape_list.append(mape(yt, yp))
        r2_list.append(r2(yt, yp))
    return {
        "mse": float(np.mean(mse_list)),
        "mape": float(np.mean(mape_list)),
        "r2": float(np.mean(r2_list)),
    }


def select_alpha_on_validation(none_pred: dict, aware_pred: dict, alpha_steps: int) -> tuple[float, dict]:
    if alpha_steps < 2:
        raise ValueError("--alpha_steps must be at least 2")

    assert_prediction_alignment(none_pred, aware_pred, split_name="validation")

    alphas = np.linspace(0.0, 1.0, alpha_steps)
    best_alpha = 0.0
    best_mse = float("inf")
    scores = {}
    for alpha in alphas:
        y_fusion = none_pred["y_pred"] + alpha * (aware_pred["y_pred"] - none_pred["y_pred"])
        val_mse = metric_triplet_from_arrays(
            none_pred["y_true"], y_fusion, none_pred["gap_start"], none_pred["gap_len"]
        )["mse"]
        scores[float(alpha)] = float(val_mse)
        if val_mse < best_mse:
            best_mse = float(val_mse)
            best_alpha = float(alpha)

    return best_alpha, scores


def assert_prediction_alignment(left: dict, right: dict, split_name: str) -> None:
    for key in ("gap_start", "gap_len"):
        if not np.array_equal(left[key], right[key]):
            raise RuntimeError(f"{split_name} predictions are not aligned for {key}")
    if left["y_true"].shape != right["y_true"].shape:
        raise RuntimeError(f"{split_name} prediction shapes differ")
    if not np.allclose(left["y_true"], right["y_true"], rtol=1e-5, atol=1e-5):
        raise RuntimeError(f"{split_name} y_true values differ between branches")


def train_tcn_branch(
    train_dataset,
    val_loader: DataLoader,
    device: str,
    L: int,
    seed: int,
    batch_train: int,
    max_epochs: int,
    patience: int,
    lr: float,
    input_mode: str,
    lambda_visible_residual: float,
) -> TCNResidualImputer:
    set_all_seeds(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_train,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = TCNResidualImputer(
        seq_len=L,
        channels=128,
        num_blocks=10,
        kernel_size=9,
        dropout=0.1,
        input_channels=4 if input_mode == "fourier_aware" else 2,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, max_epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            _, loss = prediction_and_loss(
                model,
                batch,
                device,
                input_mode=input_mode,
                lambda_visible_residual=lambda_visible_residual,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                _, loss = prediction_and_loss(
                    model,
                    batch,
                    device,
                    input_mode=input_mode,
                    lambda_visible_residual=lambda_visible_residual,
                )
                va_loss += float(loss.item())
        va_loss /= max(1, len(val_loader))

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def build_none_datasets(star, star_build: dict, L: int, gap_len: int, min_context: int, n_train: int, seed: int) -> tuple:
    split = star_build["split"]
    train_dataset = FixedTrainDataset(
        star=star,
        train_segs=split.train_segs,
        L=L,
        gap_len=gap_len,
        min_context=min_context,
        fourier=None,
        fourier_mode="static",
        has_gate=False,
        samples_per_epoch=n_train,
        seed=seed,
    )
    val_dataset = FixedSpecsDataset(
        star=star,
        specs=star_build["val_specs"],
        L=L,
        min_context=min_context,
        fourier=None,
        fourier_mode="static",
        has_gate=False,
    )
    test_dataset = FixedSpecsDataset(
        star=star,
        specs=star_build["test_specs"],
        L=L,
        min_context=min_context,
        fourier=None,
        fourier_mode="static",
        has_gate=False,
    )
    return train_dataset, val_dataset, test_dataset


def train_one_star_calibrated_fusion(
    star_path: Path,
    device: str,
    dt_factor: float,
    which_gap: str,
    fourier_mode: str,
    L: int,
    min_context: int,
    n_train: int,
    n_val: int,
    n_test: int,
    k_freqs: int,
    seed: int,
    batch_train: int,
    batch_eval: int,
    max_epochs: int,
    patience: int,
    lr: float,
    stride_eval: int,
    max_eval_per_segment: Optional[int],
    lambda_visible_residual: float,
    alpha_steps: int,
) -> dict:
    t0 = time.perf_counter()
    set_all_seeds(seed)

    star = load_star(star_path, dt_factor=dt_factor)
    cap_points = int(0.5 * L)
    gap_choice = choose_big_gap_points(
        star,
        which=which_gap,
        min_points=200,
        default_points_if_no_gaps=800,
        cap_points=cap_points,
    )
    gap_len = int(gap_choice.gap_points)

    star_build = build_star_split(
        star=star,
        L=L,
        gap_len=gap_len,
        min_context=min_context,
        k_freqs=k_freqs,
        samples_per_epoch=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        fourier_mode=fourier_mode,
        has_gate=False,
        stride_eval=stride_eval,
        max_eval_per_segment=max_eval_per_segment,
        train_sampling="fixed",
    )
    if star_build is None:
        raise RuntimeError("too_few_eligible_segments")

    fourier = star_build["fourier"]
    aware_train_dataset = star_build["train_dataset"]
    aware_val_dataset = star_build["val_dataset"]
    aware_test_dataset = star_build["test_dataset"]
    none_train_dataset, none_val_dataset, none_test_dataset = build_none_datasets(
        star=star,
        star_build=star_build,
        L=L,
        gap_len=gap_len,
        min_context=min_context,
        n_train=n_train,
        seed=seed,
    )

    none_val_loader = DataLoader(none_val_dataset, batch_size=batch_eval, shuffle=False, collate_fn=collate_fn)
    none_test_loader = DataLoader(none_test_dataset, batch_size=batch_eval, shuffle=False, collate_fn=collate_fn)
    aware_val_loader = DataLoader(aware_val_dataset, batch_size=batch_eval, shuffle=False, collate_fn=collate_fn)
    aware_test_loader = DataLoader(aware_test_dataset, batch_size=batch_eval, shuffle=False, collate_fn=collate_fn)

    print("  Training TCN none branch...")
    none_model = train_tcn_branch(
        train_dataset=none_train_dataset,
        val_loader=none_val_loader,
        device=device,
        L=L,
        seed=seed,
        batch_train=batch_train,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        input_mode="standard",
        lambda_visible_residual=0.0,
    )

    print("  Training TCN Fourier-aware branch...")
    aware_model = train_tcn_branch(
        train_dataset=aware_train_dataset,
        val_loader=aware_val_loader,
        device=device,
        L=L,
        seed=seed,
        batch_train=batch_train,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        input_mode="fourier_aware",
        lambda_visible_residual=lambda_visible_residual,
    )

    print("  Collecting validation predictions and calibrating alpha...")
    none_val = predict_loader_standard(none_model, none_val_loader, device, fourier, fourier_mode)
    aware_val = predict_loader_fourier_aware(aware_model, aware_val_loader, device)
    alpha_best, alpha_scores = select_alpha_on_validation(none_val, aware_val, alpha_steps)

    val_none = metric_triplet_from_arrays(none_val["y_true"], none_val["y_pred"], none_val["gap_start"], none_val["gap_len"])
    val_aware = metric_triplet_from_arrays(aware_val["y_true"], aware_val["y_pred"], aware_val["gap_start"], aware_val["gap_len"])
    y_fusion_val = none_val["y_pred"] + alpha_best * (aware_val["y_pred"] - none_val["y_pred"])
    val_fusion = metric_triplet_from_arrays(none_val["y_true"], y_fusion_val, none_val["gap_start"], none_val["gap_len"])

    print("  Collecting test predictions...")
    none_test = predict_loader_standard(none_model, none_test_loader, device, fourier, fourier_mode)
    aware_test = predict_loader_fourier_aware(aware_model, aware_test_loader, device)
    assert_prediction_alignment(none_test, aware_test, split_name="test")

    y_fusion_test = none_test["y_pred"] + alpha_best * (aware_test["y_pred"] - none_test["y_pred"])
    fourier_test = metric_triplet_from_arrays(
        none_test["y_true"], none_test["y_fourier"], none_test["gap_start"], none_test["gap_len"]
    )
    none_test_metrics = metric_triplet_from_arrays(
        none_test["y_true"], none_test["y_pred"], none_test["gap_start"], none_test["gap_len"]
    )
    aware_test_metrics = metric_triplet_from_arrays(
        aware_test["y_true"], aware_test["y_pred"], aware_test["gap_start"], aware_test["gap_len"]
    )
    fusion_test_metrics = metric_triplet_from_arrays(
        none_test["y_true"], y_fusion_test, none_test["gap_start"], none_test["gap_len"]
    )

    row = {
        "star_file": star_path.name,
        "n_points": int(len(star.t)),
        "n_segments": int(len(star.segments)),
        "n_real_gaps": int(len(star.real_gap_idx)),
        "which_gap": which_gap,
        "gap_percentile": float(gap_choice.percentile),
        "gap_points_used": int(gap_len),
        "seed": int(seed),
        "split_kind": "segment_level_clean",
        "train_sampling": "fixed",
        "fourier_mode": fourier_mode,
        "input_mode_none": "standard",
        "input_mode_aware": "fourier_aware",
        "warm_start_none": "none",
        "warm_start_aware": "fourier",
        "lambda_visible_residual": float(lambda_visible_residual),
        "alpha_steps": int(alpha_steps),
        "alpha_grid": ";".join(f"{a:.6g}" for a in np.linspace(0.0, 1.0, alpha_steps)),
        "alpha_best": float(alpha_best),
        "val_mse_none": val_none["mse"],
        "val_mse_aware": val_aware["mse"],
        "val_mse_fusion": val_fusion["mse"],
        "val_r2_none": val_none["r2"],
        "val_r2_aware": val_aware["r2"],
        "val_r2_fusion": val_fusion["r2"],
        "fourier_mse": fourier_test["mse"],
        "fourier_mape": fourier_test["mape"],
        "fourier_r2": fourier_test["r2"],
        "none_mse": none_test_metrics["mse"],
        "none_mape": none_test_metrics["mape"],
        "none_r2": none_test_metrics["r2"],
        "aware_mse": aware_test_metrics["mse"],
        "aware_mape": aware_test_metrics["mape"],
        "aware_r2": aware_test_metrics["r2"],
        "fusion_mse": fusion_test_metrics["mse"],
        "fusion_mape": fusion_test_metrics["mape"],
        "fusion_r2": fusion_test_metrics["r2"],
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    row.update(star_build["meta"])
    row["split_kind"] = "segment_level_clean"
    row["train_sampling"] = "fixed"
    row["fourier_mode"] = fourier_mode
    row["alpha_scores"] = ";".join(f"{a:.6g}:{s:.12g}" for a, s in alpha_scores.items())
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="TCN-only calibrated fusion between none and Fourier-aware branches."
    )
    ap.add_argument("--data_dir", type=str, default="data/LADS")
    ap.add_argument("--out_csv", type=str, default="outputs/segment_level_calibrated_fusion_tcn/tcn_calibrated_fusion_dynamic_clean.csv")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dt_factor", type=float, default=5.0)
    ap.add_argument("--which_gap", type=str, default="max", choices=["p90", "p95", "max"])
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--min_context", type=int, default=256)
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--stride_eval", type=int, default=1)
    ap.add_argument("--max_eval_per_segment", type=int, default=0)
    ap.add_argument("--k_freqs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--seed_per_star", action="store_true")
    ap.add_argument("--batch_train", type=int, default=128)
    ap.add_argument("--batch_eval", type=int, default=128)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_visible_residual", type=float, default=0.1)
    ap.add_argument("--alpha_steps", type=int, default=21)
    ap.add_argument("--fourier_mode", type=str, default="dynamic", choices=["static", "dynamic"])
    ap.add_argument("--train_sampling", type=str, default="fixed", choices=["fixed"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.alpha_steps < 2:
        raise ValueError("--alpha_steps must be at least 2")

    load_runtime_dependencies()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.dat"))
    if not files:
        raise RuntimeError(f"No .dat files found in {data_dir}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    skipped_csv = out_csv.with_name(out_csv.stem + "_skipped.csv")
    for csv_path in (out_csv, skipped_csv):
        if csv_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing CSV: {csv_path}")

    rows = []
    skipped_rows = []
    for si, star_path in enumerate(files):
        star_seed = int(args.seed + 1000 * si) if args.seed_per_star else int(args.seed)
        print(
            f"\n[{si + 1}/{len(files)}] Star: {star_path.name} | seed={star_seed} | "
            f"device={device} | calibrated_fusion=tcn_none+tcn_fourier_aware"
        )
        try:
            row = train_one_star_calibrated_fusion(
                star_path=star_path,
                device=device,
                dt_factor=args.dt_factor,
                which_gap=args.which_gap,
                fourier_mode=args.fourier_mode,
                L=args.L,
                min_context=args.min_context,
                n_train=args.n_train,
                n_val=args.n_val,
                n_test=args.n_test,
                k_freqs=args.k_freqs,
                seed=star_seed,
                batch_train=args.batch_train,
                batch_eval=args.batch_eval,
                max_epochs=args.max_epochs,
                patience=args.patience,
                lr=args.lr,
                stride_eval=args.stride_eval,
                max_eval_per_segment=args.max_eval_per_segment or None,
                lambda_visible_residual=args.lambda_visible_residual,
                alpha_steps=args.alpha_steps,
            )
        except RuntimeError as e:
            reason = str(e)
            print("  SKIP:", reason)
            skipped_rows.append(
                {
                    "star_file": star_path.name,
                    "reason": reason,
                    "n_eligible_segments": "",
                    "L": args.L,
                    "split_mode": "segment_level_clean",
                }
            )
            continue

        if row.get("split_balance_warning"):
            print(f"  Split balance warning: {row.get('split_balance_note', '')}")
        rows.append(row)
        print(
            f"  alpha={row['alpha_best']:.3g} | "
            f"none MSE={row['none_mse']:.4g} | aware MSE={row['aware_mse']:.4g} | "
            f"fusion MSE={row['fusion_mse']:.4g} | Fourier MSE={row['fourier_mse']:.4g} | "
            f"time={row['elapsed_sec']:.1f}s"
        )

    if not rows:
        raise RuntimeError("No stars were processed successfully (rows is empty).")

    fieldnames = [
        "star_file",
        "n_points",
        "n_segments",
        "n_real_gaps",
        "which_gap",
        "gap_percentile",
        "gap_points_used",
        "seed",
        "split_kind",
        "train_sampling",
        "n_eligible_segments",
        "n_train_segments",
        "n_val_segments",
        "n_test_segments",
        "train_weight",
        "val_weight",
        "test_weight",
        "split_balance_warning",
        "split_balance_note",
        "samples_per_epoch",
        "stride_eval",
        "max_eval_per_segment",
        "fourier_train_only",
        "fourier_mode",
        "input_mode_none",
        "input_mode_aware",
        "warm_start_none",
        "warm_start_aware",
        "alpha_best",
        "val_mse_none",
        "val_mse_aware",
        "val_mse_fusion",
        "val_r2_none",
        "val_r2_aware",
        "val_r2_fusion",
        "fourier_mse",
        "fourier_mape",
        "fourier_r2",
        "none_mse",
        "none_mape",
        "none_r2",
        "aware_mse",
        "aware_mape",
        "aware_r2",
        "fusion_mse",
        "fusion_mape",
        "fusion_r2",
        "elapsed_sec",
        "alpha_grid",
        "alpha_steps",
        "alpha_scores",
        "lambda_visible_residual",
        "train_n_valid_starts",
        "val_n_valid_starts",
        "test_n_valid_starts",
        "split_seed",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with open(skipped_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["star_file", "reason", "n_eligible_segments", "L", "split_mode"])
        writer.writeheader()
        for row in skipped_rows:
            writer.writerow(row)

    print("\nSaved results to:", out_csv)
    print(f"Skipped {len(skipped_rows)} stars; logged to:", skipped_csv)


if __name__ == "__main__":
    main()
