import sys
from pathlib import Path
import torch

def run_smoke_test():
    # 1. Test imports
    try:
        from scripts.run_all_lads_hybrid_tcn import train_one_star as tcn_train
        from scripts.run_all_lads_hybrid_transformer import train_one_star as transformer_train
        from scripts.run_all_lads_hybrid_conv_transformer import train_one_star as conv_train
        from scripts.run_all_lads_hybrid_tcn_basegate_r2 import train_one_star as tcn_gate_train
        from scripts.run_all_lads_hybrid_transformer_basegate_r2 import train_one_star as transformer_gate_train
        from scripts.run_all_lads_hybrid_conv_transformer_basegate_r2 import train_one_star as conv_gate_train
        print("✅ All 6 scripts imported successfully.")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)

    # 2. Test one-star TCN dynamic fixed train (1 epoch)
    data_dir = Path("data/LADS")
    files = sorted(data_dir.glob("*.dat"))
    if not files:
        print("No dat files found.")
        sys.exit(1)
    
    star_path = files[0]
    print(f"\nRunning Smoke Test on {star_path.name} (TCN Dynamic Fixed Train, 1 Epoch)")
    try:
        result = tcn_train(
            star_path=star_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            dt_factor=5.0,
            which_gap="max",
            warm_start="fourier",
            fourier_mode="dynamic",
            L=2048,
            min_context=256,
            n_train=1200,
            n_val=200,
            n_test=200,
            k_freqs=8,
            seed=123,
            batch_train=200,
            batch_eval=200,
            max_epochs=1,
            patience=5,
            lr=1e-3,
            stride_eval=1,
            max_eval_per_segment=0,
            train_sampling="fixed"
        )
        print("\n✅ Smoke test completed successfully!")
        print(f"Result metrics: Fourier MSE={result['fourier_mse']:.4f}, Hybrid MSE={result['hybrid_mse']:.4f}, Elapsed={result['elapsed_sec']:.1f}s")
    except Exception as e:
        print(f"❌ TCN test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()
