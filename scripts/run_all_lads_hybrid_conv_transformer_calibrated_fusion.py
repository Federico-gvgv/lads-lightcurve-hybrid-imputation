from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from scripts import run_all_lads_hybrid_transformer_calibrated_fusion as attention_calibrated


ConvTransformerResidualImputer = None


def load_runtime_dependencies() -> None:
    global ConvTransformerResidualImputer

    attention_calibrated.load_runtime_dependencies()

    from src.models.conv_transformer_residual import (
        ConvTransformerResidualImputer as ConvTransformerResidualImputer_class,
    )

    ConvTransformerResidualImputer = ConvTransformerResidualImputer_class


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
    conv_kernel: int,
    conv_layers: int,
) -> dict:
    def model_factory(input_channels: int):
        return ConvTransformerResidualImputer(
            seq_len=L,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            conv_kernel=conv_kernel,
            conv_layers=conv_layers,
            input_channels=input_channels,
        )

    return attention_calibrated.train_one_star_attention_calibrated_fusion(
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
        model_label="Conv-Transformer",
        model_factory=model_factory,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Conv-Transformer calibrated fusion between none and Fourier-aware branches."
    )
    attention_calibrated.add_common_args(
        ap,
        default_out_csv=(
            "outputs/segment_level_calibrated_fusion_conv_transformer/"
            "conv_transformer_calibrated_fusion_dynamic_clean.csv"
        ),
    )
    ap.add_argument("--conv_kernel", type=int, default=9)
    ap.add_argument("--conv_layers", type=int, default=2)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    attention_calibrated.run_all(
        args,
        model_label="Conv-Transformer",
        train_one_star_fn=train_one_star_calibrated_fusion,
        extra_train_kwargs=lambda parsed: {
            "conv_kernel": parsed.conv_kernel,
            "conv_layers": parsed.conv_layers,
        },
    )


if __name__ == "__main__":
    main()
