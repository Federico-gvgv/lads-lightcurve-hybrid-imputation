# Hybrid Reconstruction of LADS Light Curves

This repository contains the code used in the Bachelor’s Thesis:

**Hybrid reconstruction of low-amplitude Delta Scuti star light curves using signal processing and machine learning**

The project studies the reconstruction of large gaps in light curves from low-amplitude Delta Scuti (LADS) stars observed by the TESS mission. The final method combines neural time-series models with a dynamic Fourier-based physical initialization and a validation-calibrated fusion mechanism.

The main contribution is a calibrated hybrid reconstruction scheme that combines:

- a neural branch without physical warm-start;
- a Fourier-Aware Residual (FAR) branch;
- a validation-selected fusion weight alpha, used to decide how much the final prediction should rely on the FAR branch.

The calibrated fusion is evaluated with TCN, Transformer, Conv-Transformer and U-Net 1D architectures under a clean segment-level protocol.

---

## Repository structure

```text
.
├── configs/        # Optional configuration files
├── data/           # Placeholder for local LADS data, not included
├── notebooks/      # Final analysis / figure-generation notebooks
├── outputs/        # Generated results, ignored by Git
├── scripts/        # Training, evaluation and utility scripts
├── src/            # Reusable source code
├── README.md
└── .gitignore
```

The core source code is located in `src/`, while the main executable scripts are located in `scripts/`.

---

## Data

Raw LADS light-curve files are not included in this repository.

Place the `.dat` files under:

```text
data/LADS/
```

Each file is expected to contain a light curve for a single star, with time and flux information. The experiments assume that each star is processed independently.

---

## Experimental protocol

The final protocol follows a clean segment-level split:

1. Load one LADS light curve.
2. Detect continuous temporal segments.
3. Split complete segments into train, validation and test partitions.
4. Extract fixed-length windows with L = 2048.
5. Insert a contiguous synthetic gap based on the largest real gap of the star.
6. Train the model independently for each star.
7. Select the calibrated fusion weight alpha on the validation split.
8. Evaluate only inside the imputed gap on the test split.

The Fourier model is estimated only from training segments, avoiding leakage from validation or test data.

---

## Main method

The calibrated fusion combines two predictions:

```text
y_fusion = (1 - alpha) * y_none + alpha * y_FAR
```

where:

- `y_none` is the prediction of the neural model without physical warm-start;
- `y_FAR` is the prediction of the Fourier-Aware Residual branch;
- `alpha` is selected on validation by minimizing MSE.

The final CSV files produced by the calibrated-fusion runners contain per-star metrics for all three variants:

- no warm-start branch;
- FAR branch;
- calibrated fusion.

They also include the selected `alpha_best`.

---

## Models

The following architectures are implemented:

- TCN;
- Transformer;
- Conv-Transformer;
- U-Net 1D.

Model definitions are located in:

```text
src/models/
```

The final thesis results focus mainly on the TCN because it provides the most robust neural branch, but calibrated fusion is evaluated for all implemented architectures.

---

## Main execution script

The main entry point for the final experiments is:

```bash
bash scripts/run_calibrated_fusion.sh run
```

For long runs, use `nohup`:

```bash
nohup bash scripts/run_calibrated_fusion.sh run > outputs/nohup_calibrated_fusion_all.log 2>&1 &
```

By default, the script runs the calibrated-fusion experiments for all architectures and seeds defined inside the script.

You can restrict the execution with environment variables. For example, to run only TCN:

```bash
ARCHS="tcn" SEEDS="123 456 789" nohup bash scripts/run_calibrated_fusion.sh run > outputs/nohup_tcn_calibrated_fusion.log 2>&1 &
```

To run only Transformer and Conv-Transformer:

```bash
ARCHS="transformer conv_transformer" SEEDS="123 456 789" nohup bash scripts/run_calibrated_fusion.sh run > outputs/nohup_transformers_calibrated_fusion.log 2>&1 &
```

The GPU can be selected with:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_calibrated_fusion.sh run
```

---

## Optional scripts

Some scripts are kept for utility or reproducibility checks:

```text
scripts/sanity_check_segment_split.py
scripts/check_min_segment_len.py
scripts/summarize_lightcurves.py
scripts/make_lads_split_specs.py
```

If present, `scripts/run_segment_level_fourier_aware.sh` can be used to run FAR-related experiments separately. However, the calibrated-fusion pipeline already evaluates both the no-warm-start and FAR branches, so this script is not required to reproduce the final calibrated-fusion results.

---

## Notebooks

The final notebook kept in the repository is:

```text
notebooks/final_qualitative_figures.ipynb
```

It is used to regenerate the qualitative figures included in the thesis results chapter.

Before committing notebooks, it is recommended to clear their outputs:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

---

## Outputs

Generated outputs are not versioned by Git.

Typical output directories include:

```text
outputs/segment_level_calibrated_fusion_tcn/
outputs/segment_level_calibrated_fusion_transformer/
outputs/segment_level_calibrated_fusion_conv_transformer/
outputs/segment_level_calibrated_fusion_unet1d/
```

Each directory contains CSV files and logs for the corresponding architecture and seed.

---

## Requirements

The project is written in Python and uses PyTorch.

Main dependencies:

```text
numpy
pandas
matplotlib
scikit-learn
torch
jupyter
```

A minimal installation can be done with:

```bash
pip install numpy pandas matplotlib scikit-learn torch jupyter
```

Depending on the server or local machine, PyTorch may need to be installed following the official instructions for the available CUDA version.

---

## Reproducibility notes

The final experiments are designed to be reproducible at the level of:

- fixed segment-level train/validation/test partitions;
- fixed random seeds;
- per-star training and evaluation;
- explicit CSV outputs for each architecture and seed.

The raw data and generated outputs are intentionally excluded from the repository to keep it lightweight.

---

## Author

Federico García Valenzuela

Bachelor’s Thesis — Computer Engineering
University of Granada
