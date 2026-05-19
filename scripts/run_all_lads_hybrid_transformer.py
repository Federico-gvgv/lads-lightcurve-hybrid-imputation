# scripts/run_all_lads_hybrid_transformer.py
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.io_gaps import load_star
from src.protocol import choose_big_gap_points
from src.fourier_baseline import fit_predict_fourier
from src.fourier_dynamic import fit_predict_fourier_dynamic
from src.segment_split import build_star_split, collate_fn
from src.metrics import denorm, gap_indices, mse, mape, r2
from src.models.transformer_residual import TransformerResidualImputer


def set_all_seeds(seed: int):
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
    return torch.stack([
        batch["y_obs_zeroed"].to(device),
        batch["y_fourier_full"].to(device),
        batch["obs_mask"].to(device),
        batch["residual_visible"].to(device),
    ], dim=1)


def forward_delta(model, batch: dict, device: str, input_mode: str) -> torch.Tensor:
    if input_mode == "fourier_aware":
        return model(build_fourier_aware_x(batch, device))
    return model(batch["y_in"].to(device), batch["obs_mask"].to(device))


def prediction_and_loss(
    model,
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



@torch.no_grad()
def eval_loader(model, loader, device: str, L: int, fourier, fourier_mode: str, input_mode: str) -> dict:
    mse_list, mape_list, r2_list = [], [], []
    mse_f_list, mape_f_list, r2_f_list = [], [], []

    model.eval()
    for batch in loader:
        y_true_n = batch["y_true"].cpu().numpy()
        obs_mask = batch["obs_mask"].cpu().numpy()
        mu = batch["mu"].cpu().numpy()
        sd = batch["sd"].cpu().numpy()

        B = y_true_n.shape[0]
        if input_mode == "fourier_aware":
            y_fourier_n = batch["y_fourier_full"].cpu().numpy()
        else:
            y_fourier_n = np.empty_like(y_true_n)
            for i in range(B):
                gs = int(batch["gap_start"][i])
                gl = int(batch["gap_len"][i])
                if fourier_mode == "dynamic":
                    y_fourier_n[i] = fit_predict_fourier_dynamic(
                        fourier.A_full, y_true_n[i], obs_mask[i], gap_start=gs, gap_len=gl,
                    )
                else:
                    y_fourier_n[i] = fit_predict_fourier(fourier.A_full, y_true_n[i], obs_mask[i])

        y_true   = denorm(y_true_n,   mu[:, None], sd[:, None])
        y_fourier = denorm(y_fourier_n, mu[:, None], sd[:, None])

        delta = forward_delta(model, batch, device, input_mode).cpu().numpy()

        if input_mode == "fourier_aware":
            y_hat_n = y_fourier_n + delta * (1.0 - obs_mask)
        else:
            y_hat_n = batch["y_in"].cpu().numpy() + delta * (1.0 - obs_mask)
        y_hat   = denorm(y_hat_n, mu[:, None], sd[:, None])

        for i in range(B):
            gs = int(batch["gap_start"][i])
            gl = int(batch["gap_len"][i])
            gi = gap_indices(L, gs, gl)

            yt = y_true[i][gi]
            yh = y_hat[i][gi]
            yf = y_fourier[i][gi]

            mse_list.append(mse(yt, yh))
            mape_list.append(mape(yt, yh))
            r2_list.append(r2(yt, yh))

            mse_f_list.append(mse(yt, yf))
            mape_f_list.append(mape(yt, yf))
            r2_f_list.append(r2(yt, yf))

    return {
        "fourier_mse": float(np.mean(mse_f_list)),
        "fourier_mape": float(np.mean(mape_f_list)),
        "fourier_r2": float(np.mean(r2_f_list)),
        "hybrid_mse": float(np.mean(mse_list)),
        "hybrid_mape": float(np.mean(mape_list)),
        "hybrid_r2": float(np.mean(r2_list)),
    }


def train_one_star(
    star_path: Path,
    device: str,
    dt_factor: float,
    which_gap: str,
    warm_start: str,
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
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_ff: int,
    dropout: float,
    input_mode: str,
    lambda_visible_residual: float,
):
    t0 = time.perf_counter()
    set_all_seeds(seed)

    star = load_star(star_path, dt_factor=dt_factor)

    cap_points = int(0.5 * L)
    gap_choice = choose_big_gap_points(
        star, which=which_gap,
        min_points=200,
        default_points_if_no_gaps=800,
        cap_points=cap_points,
    )
    G = int(gap_choice.gap_points)

    # --- Segment-level clean split (skip if < 3 eligible segments) ---
    star_build = build_star_split(
        star=star, L=L, gap_len=G, min_context=min_context,
        k_freqs=k_freqs, samples_per_epoch=n_train,
        n_val=n_val, n_test=n_test, seed=seed,
        fourier_mode=fourier_mode if warm_start == "fourier" else "static",
        has_gate=False,
        stride_eval=stride_eval,
        max_eval_per_segment=max_eval_per_segment,
        train_sampling=train_sampling,
    )
    if star_build is None:
        raise RuntimeError("too_few_eligible_segments")

    fourier       = star_build["fourier"]
    train_dataset = star_build["train_dataset"]
    val_dataset   = star_build["val_dataset"]
    test_dataset  = star_build["test_dataset"]
    split_meta    = star_build["meta"]

    # If warm_start=="none", rebuild datasets without Fourier warm-start
    if warm_start == "none":
        star_build_nf = build_star_split(
            star=star, L=L, gap_len=G, min_context=min_context,
            k_freqs=k_freqs, samples_per_epoch=n_train,
            n_val=n_val, n_test=n_test, seed=seed,
            fourier_mode="static", has_gate=False,
        )
        # Rebuild with fourier=None by temporarily patching A_full reference
        # Simplest approach: build_star_split with warm_start None is handled
        # by passing fourier=None → we do it inline below
        from src.segment_split import (
            TrainEpochDataset, FixedSpecsDataset,
            make_fixed_eval_specs, build_fourier_from_longest_train_seg,
        )
        split_obj = star_build["split"]
        train_dataset = TrainEpochDataset(
            star=star, train_segs=split_obj.train_segs,
            L=L, gap_len=G, min_context=min_context,
            fourier=None, fourier_mode="static",
            has_gate=False, samples_per_epoch=n_train, seed=seed,
        )
        val_specs = star_build["val_specs"]
        test_specs = star_build["test_specs"]
        val_dataset = FixedSpecsDataset(
            star=star, specs=val_specs, L=L, min_context=min_context,
            fourier=None, fourier_mode="static", has_gate=False,
        )
        test_dataset = FixedSpecsDataset(
            star=star, specs=test_specs, L=L, min_context=min_context,
            fourier=None, fourier_mode="static", has_gate=False,
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_train,
                              shuffle=False, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_eval,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_eval,
                              shuffle=False, collate_fn=collate_fn)

    model = TransformerResidualImputer(
        seq_len=L,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
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
                model, batch, device, input_mode, lambda_visible_residual
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
                    model, batch, device, input_mode, lambda_visible_residual
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

    metrics = eval_loader(model, test_loader, device=device, L=L,
                          fourier=fourier, fourier_mode=fourier_mode,
                          input_mode=input_mode)

    meta = {
        "star_file": star_path.name,
        "n_points": int(len(star.t)),
        "n_segments": int(len(star.segments)),
        "n_real_gaps": int(len(star.real_gap_idx)),
        "which_gap": which_gap,
        "gap_points_used": int(G),
        "gap_percentile": float(gap_choice.percentile),
        "seed": int(seed),
        "warm_start": warm_start,
        "fourier_mode": fourier_mode,
        "input_mode": input_mode,
        "lambda_visible_residual": float(lambda_visible_residual),
    }
    meta.update(split_meta)
    meta.update(metrics)
    meta["elapsed_sec"] = float(time.perf_counter() - t0)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/LADS")
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--dt_factor", type=float, default=5.0)
    ap.add_argument("--which_gap", type=str, default="max", choices=["p90","p95","max"])

    ap.add_argument("--warm_start", type=str, default="fourier", choices=["fourier", "none"])

    ap.add_argument("--fourier_mode", type=str, default="static", choices=["static", "dynamic"],
                    help="Fourier baseline/warm-start type")
    ap.add_argument(
        "--input_mode",
        type=str,
        default="standard",
        choices=["standard", "fourier_aware"],
        help="Input representation. standard keeps the existing y_in+mask behavior; fourier_aware uses explicit Fourier/full residual channels.",
    )
    ap.add_argument("--lambda_visible_residual", type=float, default=0.0)

    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--min_context", type=int, default=256)

    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)

    ap.add_argument("--stride_eval", type=int, default=1,
                    help="Stride between candidate eval window starts (1=dense)")
    ap.add_argument("--max_eval_per_segment", type=int, default=0,
                    help="Max eval windows per segment before global cap (0=no cap)")
    ap.add_argument("--train_sampling", type=str, default="fixed", choices=["fixed", "online"],
                    help="Use fixed materialized train dataset or online resampling")

    ap.add_argument("--k_freqs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--seed_per_star", action="store_true")

    ap.add_argument("--batch_train", type=int, default=128)
    ap.add_argument("--batch_eval", type=int, default=128)

    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)

    # model hparams
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dim_ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    args = ap.parse_args()

    if args.input_mode == "fourier_aware" and args.warm_start != "fourier":
        raise ValueError("--input_mode fourier_aware requires --warm_start fourier")

    if args.train_sampling == "online" and args.fourier_mode == "dynamic" and args.warm_start == "fourier":
        print("\nWARNING: online train sampling with dynamic Fourier is extremely slow because train windows are rematerialized every epoch.\n")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.dat"))
    if len(files) == 0:
        raise RuntimeError(f"No .dat files found in {data_dir}")

    if args.out_csv is None:
        if args.input_mode == "fourier_aware":
            out_csv = Path(f"outputs/segment_level_fourier_aware/transformer_fourier_aware_{args.fourier_mode}_clean.csv")
        else:
            out_csv = Path(f"outputs/transformer_{args.warm_start}_{args.fourier_mode}.csv")
    else:
        out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Añadimos la lista manual de exclusión de las estrellas anómalas
    manual_skip_list = {
        # "TIC 106886169lc.dat",
        # "TIC 12524129 lc.dat",
        # "TIC 196921106lc.dat",
        # "TIC 7808834lc.dat",
        # "TIC 255548143lc.dat"
    }

    rows = []
    skipped_rows = []
    for si, star_path in enumerate(files):
        # 2. Comprobamos la lista antes de ejecutar
        if star_path.name in manual_skip_list:
            print(f"\n[{si+1}/{len(files)}] Star: {star_path.name} -> SKIP (En lista manual)")
            continue

        star_seed = int(args.seed + 1000 * si) if args.seed_per_star else int(args.seed)

        print(f"\n[{si+1}/{len(files)}] Star: {star_path.name} | seed={star_seed} | device={device} | warm_start={args.warm_start} | fourier_mode={args.fourier_mode} | input_mode={args.input_mode}")
        try:
            row = train_one_star(
                star_path=star_path,
                device=device,
                dt_factor=args.dt_factor,
                which_gap=args.which_gap,
                warm_start=args.warm_start,
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
                train_sampling=args.train_sampling,
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_ff=args.dim_ff,
                dropout=args.dropout,
                input_mode=args.input_mode,
                lambda_visible_residual=args.lambda_visible_residual,
            )
        except RuntimeError as e:
            reason = str(e)
            print("  SKIP:", reason)
            skipped_rows.append(dict(
                star_file=star_path.name,
                reason=reason,
                n_eligible_segments="",
                L=args.L,
                split_mode="segment_level_clean",
            ))
            continue

        if row.get("split_balance_warning"):
            print(f"  ⚠ Split balance warning: {row.get('split_balance_note', '')}")
        rows.append(row)
        print(
            f"  gap_points={row['gap_points_used']} | "
            f"Fourier MSE={row['fourier_mse']:.4g} | Hybrid MSE={row['hybrid_mse']:.4g} | "
            f"time={row['elapsed_sec']:.1f}s | split={row['split_kind']}"
        )

    if not rows:
        raise RuntimeError("No stars were processed successfully.")

    fieldnames = [
        "star_file", "n_points", "n_segments", "n_real_gaps",
        "which_gap", "gap_percentile", "gap_points_used",
        "seed", "split_kind", "warm_start", "fourier_mode", "input_mode", "lambda_visible_residual",
        "n_eligible_segments", "n_train_segments", "n_val_segments", "n_test_segments",
        "train_weight", "val_weight", "test_weight",
        "split_balance_warning", "split_balance_note",
        "split_kind", "train_sampling", "samples_per_epoch", "stride_eval", "max_eval_per_segment", "fourier_train_only",
        "fourier_mse", "fourier_mape", "fourier_r2",
        "hybrid_mse", "hybrid_mape", "hybrid_r2",
        "elapsed_sec",
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Write skipped-stars log
    skipped_csv = out_csv.with_name(out_csv.stem + "_skipped.csv")
    skip_fieldnames = ["star_file", "reason", "n_eligible_segments", "L", "split_mode"]
    with open(skipped_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=skip_fieldnames)
        w.writeheader()
        for r in skipped_rows:
            w.writerow(r)

    print("\nSaved results to:", out_csv)
    print(f"Skipped {len(skipped_rows)} stars — logged to:", skipped_csv)


if __name__ == "__main__":
    main()
