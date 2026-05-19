from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from scripts import run_all_lads_hybrid_tcn_calibrated_fusion as calibrated


torch = None
DataLoader = None
TransformerResidualImputer = None


CALIBRATED_FUSION_FIELDNAMES = [
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


def load_runtime_dependencies() -> None:
    global torch
    global DataLoader
    global TransformerResidualImputer

    calibrated.load_runtime_dependencies()

    import torch as torch_module
    from torch.utils.data import DataLoader as DataLoader_class

    from src.models.transformer_residual import (
        TransformerResidualImputer as TransformerResidualImputer_class,
    )

    torch = torch_module
    DataLoader = DataLoader_class
    TransformerResidualImputer = TransformerResidualImputer_class


def train_attention_branch(
    train_dataset,
    val_loader,
    device: str,
    seed: int,
    batch_train: int,
    max_epochs: int,
    patience: int,
    lr: float,
    input_mode: str,
    lambda_visible_residual: float,
    model_factory: Callable[[int], object],
):
    calibrated.set_all_seeds(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_train,
        shuffle=False,
        collate_fn=calibrated.collate_fn,
    )

    input_channels = 4 if input_mode == "fourier_aware" else 2
    model = model_factory(input_channels).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, max_epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            _, loss = calibrated.prediction_and_loss(
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
                _, loss = calibrated.prediction_and_loss(
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


def train_one_star_attention_calibrated_fusion(
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
    train_sampling: str,
    lambda_visible_residual: float,
    alpha_steps: int,
    model_label: str,
    model_factory: Callable[[int], object],
) -> dict:
    t0 = time.perf_counter()
    calibrated.set_all_seeds(seed)

    star = calibrated.load_star(star_path, dt_factor=dt_factor)
    cap_points = int(0.5 * L)
    gap_choice = calibrated.choose_big_gap_points(
        star,
        which=which_gap,
        min_points=200,
        default_points_if_no_gaps=800,
        cap_points=cap_points,
    )
    gap_len = int(gap_choice.gap_points)

    star_build = calibrated.build_star_split(
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
        train_sampling=train_sampling,
    )
    if star_build is None:
        raise RuntimeError("too_few_eligible_segments")

    fourier = star_build["fourier"]
    aware_train_dataset = star_build["train_dataset"]
    aware_val_dataset = star_build["val_dataset"]
    aware_test_dataset = star_build["test_dataset"]
    none_train_dataset, none_val_dataset, none_test_dataset = calibrated.build_none_datasets(
        star=star,
        star_build=star_build,
        L=L,
        gap_len=gap_len,
        min_context=min_context,
        n_train=n_train,
        seed=seed,
    )

    none_val_loader = DataLoader(none_val_dataset, batch_size=batch_eval, shuffle=False, collate_fn=calibrated.collate_fn)
    none_test_loader = DataLoader(none_test_dataset, batch_size=batch_eval, shuffle=False, collate_fn=calibrated.collate_fn)
    aware_val_loader = DataLoader(aware_val_dataset, batch_size=batch_eval, shuffle=False, collate_fn=calibrated.collate_fn)
    aware_test_loader = DataLoader(aware_test_dataset, batch_size=batch_eval, shuffle=False, collate_fn=calibrated.collate_fn)

    print(f"  Training {model_label} none branch...")
    none_model = train_attention_branch(
        train_dataset=none_train_dataset,
        val_loader=none_val_loader,
        device=device,
        seed=seed,
        batch_train=batch_train,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        input_mode="standard",
        lambda_visible_residual=0.0,
        model_factory=model_factory,
    )

    print(f"  Training {model_label} Fourier-aware branch...")
    aware_model = train_attention_branch(
        train_dataset=aware_train_dataset,
        val_loader=aware_val_loader,
        device=device,
        seed=seed,
        batch_train=batch_train,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        input_mode="fourier_aware",
        lambda_visible_residual=lambda_visible_residual,
        model_factory=model_factory,
    )

    print("  Collecting validation predictions and calibrating alpha...")
    none_val = calibrated.predict_loader_standard(none_model, none_val_loader, device, fourier, fourier_mode)
    aware_val = calibrated.predict_loader_fourier_aware(aware_model, aware_val_loader, device)
    alpha_best, alpha_scores = calibrated.select_alpha_on_validation(none_val, aware_val, alpha_steps)

    val_none = calibrated.metric_triplet_from_arrays(
        none_val["y_true"], none_val["y_pred"], none_val["gap_start"], none_val["gap_len"]
    )
    val_aware = calibrated.metric_triplet_from_arrays(
        aware_val["y_true"], aware_val["y_pred"], aware_val["gap_start"], aware_val["gap_len"]
    )
    y_fusion_val = none_val["y_pred"] + alpha_best * (aware_val["y_pred"] - none_val["y_pred"])
    val_fusion = calibrated.metric_triplet_from_arrays(
        none_val["y_true"], y_fusion_val, none_val["gap_start"], none_val["gap_len"]
    )

    print("  Collecting test predictions...")
    none_test = calibrated.predict_loader_standard(none_model, none_test_loader, device, fourier, fourier_mode)
    aware_test = calibrated.predict_loader_fourier_aware(aware_model, aware_test_loader, device)
    calibrated.assert_prediction_alignment(none_test, aware_test, split_name="test")

    y_fusion_test = none_test["y_pred"] + alpha_best * (aware_test["y_pred"] - none_test["y_pred"])
    fourier_test = calibrated.metric_triplet_from_arrays(
        none_test["y_true"], none_test["y_fourier"], none_test["gap_start"], none_test["gap_len"]
    )
    none_test_metrics = calibrated.metric_triplet_from_arrays(
        none_test["y_true"], none_test["y_pred"], none_test["gap_start"], none_test["gap_len"]
    )
    aware_test_metrics = calibrated.metric_triplet_from_arrays(
        aware_test["y_true"], aware_test["y_pred"], aware_test["gap_start"], aware_test["gap_len"]
    )
    fusion_test_metrics = calibrated.metric_triplet_from_arrays(
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
        "train_sampling": train_sampling,
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
    row["train_sampling"] = train_sampling
    row["fourier_mode"] = fourier_mode
    row["alpha_scores"] = ";".join(f"{a:.6g}:{s:.12g}" for a, s in alpha_scores.items())
    return row


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
    train_sampling: str,
    lambda_visible_residual: float,
    alpha_steps: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_ff: int,
    dropout: float,
) -> dict:
    def model_factory(input_channels: int):
        return TransformerResidualImputer(
            seq_len=L,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            input_channels=input_channels,
        )

    return train_one_star_attention_calibrated_fusion(
        star_path=star_path,
        device=device,
        dt_factor=dt_factor,
        which_gap=which_gap,
        fourier_mode=fourier_mode,
        L=L,
        min_context=min_context,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        k_freqs=k_freqs,
        seed=seed,
        batch_train=batch_train,
        batch_eval=batch_eval,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        stride_eval=stride_eval,
        max_eval_per_segment=max_eval_per_segment,
        train_sampling=train_sampling,
        lambda_visible_residual=lambda_visible_residual,
        alpha_steps=alpha_steps,
        model_label="Transformer",
        model_factory=model_factory,
    )


def add_common_args(ap: argparse.ArgumentParser, default_out_csv: str) -> None:
    ap.add_argument("--data_dir", type=str, default="data/LADS")
    ap.add_argument("--out_csv", type=str, default=default_out_csv)
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
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_visible_residual", type=float, default=0.1)
    ap.add_argument("--alpha_steps", type=int, default=21)
    ap.add_argument("--fourier_mode", type=str, default="dynamic", choices=["static", "dynamic"])
    ap.add_argument("--train_sampling", type=str, default="fixed", choices=["fixed"])
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dim_ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Transformer calibrated fusion between none and Fourier-aware branches."
    )
    add_common_args(
        ap,
        default_out_csv="outputs/segment_level_calibrated_fusion_transformer/transformer_calibrated_fusion_dynamic_clean.csv",
    )
    return ap.parse_args()


def run_all(
    args: argparse.Namespace,
    model_label: str,
    train_one_star_fn: Callable[..., dict],
    extra_train_kwargs: Optional[Callable[[argparse.Namespace], dict]] = None,
) -> None:
    if args.alpha_steps < 2:
        raise ValueError("--alpha_steps must be at least 2")

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
            f"device={device} | calibrated_fusion={model_label.lower()}_none+{model_label.lower()}_fourier_aware"
        )
        try:
            train_kwargs = {
                "star_path": star_path,
                "device": device,
                "dt_factor": args.dt_factor,
                "which_gap": args.which_gap,
                "fourier_mode": args.fourier_mode,
                "L": args.L,
                "min_context": args.min_context,
                "n_train": args.n_train,
                "n_val": args.n_val,
                "n_test": args.n_test,
                "k_freqs": args.k_freqs,
                "seed": star_seed,
                "batch_train": args.batch_train,
                "batch_eval": args.batch_eval,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "lr": args.lr,
                "stride_eval": args.stride_eval,
                "max_eval_per_segment": args.max_eval_per_segment or None,
                "train_sampling": args.train_sampling,
                "lambda_visible_residual": args.lambda_visible_residual,
                "alpha_steps": args.alpha_steps,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_ff": args.dim_ff,
                "dropout": args.dropout,
            }
            if extra_train_kwargs is not None:
                train_kwargs.update(extra_train_kwargs(args))
            row = train_one_star_fn(**train_kwargs)
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

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CALIBRATED_FUSION_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CALIBRATED_FUSION_FIELDNAMES})

    with open(skipped_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["star_file", "reason", "n_eligible_segments", "L", "split_mode"])
        writer.writeheader()
        for row in skipped_rows:
            writer.writerow(row)

    print("\nSaved results to:", out_csv)
    print(f"Skipped {len(skipped_rows)} stars; logged to:", skipped_csv)


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    run_all(args, model_label="Transformer", train_one_star_fn=train_one_star_calibrated_fusion)


if __name__ == "__main__":
    main()
