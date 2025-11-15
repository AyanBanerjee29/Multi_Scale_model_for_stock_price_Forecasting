# -*- coding: utf-8 -*-
"""
Main script for SAMBA stock price forecasting using Approach 2:
1.  Split 80% (Dev) / 20% (Final Test)
2.  Stage 1: Hyperparameter tuning on 80% Dev Set
3.  Stage 2: Train final model on 80% Dev Set
4.  Stage 3: Evaluate final model on 20% Final Test Set
"""
import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

# --- Import project modules ---
from paper_config import get_paper_config, get_dataset_info
from models import SAMBA
from utils import (
    init_seed, pearson_correlation, rank_information_coefficient, All_Metrics
)
# Import the new/updated data utils
from utils.data_utils import load_raw_data, create_sequences, MinMaxNorm01, data_loader
from trainer import Trainer

# --- GPU/CPU Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------


def main(cli_args):
    """Main training and testing function"""
    
    # --- 1. Initial Setup ---
    model_args, config = get_paper_config()
    dataset_info = get_dataset_info()
    init_seed(config.seed)
    
    # Create a directory for outputs
    output_dir = "final_model_outputs"
    os.makedirs(output_dir, exist_ok=True)
    best_params_file = os.path.join(output_dir, "best_params.json")
    final_model_file = os.path.join(output_dir, "final_samba_model.pth")
    
    print("🚀 SAMBA: A Graph-Mamba Approach for Stock Price Prediction (Approach 2)")
    print(f"Using device: {device}")
    
    # --- 2. STAGE 0: Load and Split Data CORRECTLY ---
    print("STAGE 0: Loading, Splitting, and Scaling Data...")
    dataset_file = "Dataset/NIFTY50_features_wide.csv"
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset {dataset_file} not found! Run create_feature_dataset.py first.")
        return

    df_full, price_index = load_raw_data(dataset_file, target_col_name='close')
    num_features = len(df_full.columns)
    model_args.vocab_size = num_features # Update model config
    config.num_nodes = num_features
    
    # --- Create 80% (Dev) / 20% (Final Test) Split ---
    test_split_ratio = 0.20
    total_samples_df = len(df_full)
    test_size = int(total_samples_df * test_split_ratio)
    dev_size = total_samples_df - test_size
    
    df_dev = df_full.iloc[:dev_size]
    df_test = df_full.iloc[dev_size:]
    
    print(f"Total data: {total_samples_df} rows")
    print(f"Development Set (80%): {len(df_dev)} rows")
    print(f"Final Test Set (20%): {len(df_test)} rows")

    # --- 3. Scale Data CORRECTLY ---
    print("Fitting scaler on Development Set...")
    mmn = MinMaxNorm01()
    mmn.fit(df_dev.to_numpy(dtype=np.float32))
    
    data_dev_scaled = mmn.transform(df_dev.to_numpy(dtype=np.float32))
    data_test_scaled = mmn.transform(df_test.to_numpy(dtype=np.float32))

    # --- 4. Create Sequences ---
    print("Creating sequences...")
    window = config.lag
    predict = config.horizon
    
    XX_dev, YY_dev = create_sequences(data_dev_scaled, window, predict, price_index)
    XX_test, YY_test = create_sequences(data_test_scaled, window, predict, price_index)
    
    print(f"Development sequences: {len(XX_dev)}")
    print(f"Final Test sequences: {len(XX_test)}")

    args = config.to_dict() # Convert config to dict for trainer

    # --- 5. STAGE 1: Hyperparameter Tuning (if 'train' mode) ---
    if cli_args.mode == 'train':
        print("\n===== STAGE 1: HYPERPARAMETER TUNING (on 80% Dev Set) =====")

        param_grid = {
            'lr_init': [0.001, 0.0005],
            'hid': [32, 64],
            'embed_dim': [10], # Kept simple
            'cheb_k': [2, 3]
        }
        
        param_list = list(ParameterGrid(param_grid))
        results = []
        
        for i, params in enumerate(param_list):
            print(f"\n--- Tuning Run {i+1}/{len(param_list)}: {params} ---")
            fold_validation_scores = []
            
            n_splits = 5
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            for fold, (train_index, val_index) in enumerate(tscv.split(XX_dev)):
                print(f"Fold {fold+1}/{n_splits}...")
                
                X_train_fold, y_train_fold = XX_dev[train_index], YY_dev[train_index]
                X_val_fold, y_val_fold = XX_dev[val_index], YY_dev[val_index]

                # Split X_train_fold *again* for the trainer's early stopping
                val_ratio_inner = 0.15
                inner_val_size = int(len(X_train_fold) * val_ratio_inner)
                inner_train_size = len(X_train_fold) - inner_val_size

                if inner_train_size < 10 or inner_val_size < 1:
                    print("Fold too small, skipping.")
                    continue
                
                X_train_inner, y_train_inner = X_train_fold[:inner_train_size], y_train_fold[:inner_train_size]
                X_val_inner, y_val_inner = X_train_fold[inner_train_size:], y_train_fold[inner_train_size:]

                train_loader = data_loader(X_train_inner, y_train_inner, 64, shuffle=True, drop_last=True)
                val_loader_inner = data_loader(X_val_inner, y_val_inner, 64, shuffle=False, drop_last=False)
                fold_val_loader = data_loader(X_val_fold, y_val_fold, 64, shuffle=False, drop_last=False)

                model = SAMBA(
                    model_args, params['hid'], window, predict,
                    params['embed_dim'], params["cheb_k"]
                ).to(device)

                for p in model.parameters():
                    if p.dim() > 1: nn.init.xavier_uniform_(p)
                    else: nn.init.uniform_(p)

                loss_fn = torch.nn.MSELoss().to(device)
                optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr_init'])
                
                trainer = Trainer(
                    model, loss_fn, optimizer, train_loader, val_loader_inner,
                    args=args, lr_scheduler=None
                )
                
                best_model_state, _ = trainer.train()
                
                model.load_state_dict(best_model_state)
                
                y_pred_fold, y_true_fold = Trainer.test(model, args, fold_val_loader, trainer.logger)
                
                # Rescale to get true error
                y_p = mmn.inverse_transform(y_pred_fold.cpu().numpy())
                y_t = mmn.inverse_transform(y_true_fold.cpu().numpy())
                
                mae, _, _ = All_Metrics(torch.tensor(y_p), torch.tensor(y_t), None, None)
                fold_validation_scores.append(mae.item())

            avg_score = np.mean(fold_validation_scores)
            print(f"--- Avg. MAE for {params}: {avg_score:.4f} ---")
            results.append({'params': params, 'score': avg_score})

        best_result = min(results, key=lambda x: x['score'])
        best_params = best_result['params']
        print("\n===== STAGE 1 Complete =====")
        print(f"Best Validation MAE: {best_result['score']:.4f}")
        print(f"Best Hyperparameters: {best_params}")
        
        with open(best_params_file, 'w') as f:
            json.dump(best_params, f)

        # --- 6. STAGE 2: Train Final Model ---
        print("\n===== STAGE 2: Training Final Model (on all 80% Dev Set) =====")
        
        final_val_size = int(len(XX_dev) * 0.15)
        final_train_size = len(XX_dev) - final_val_size
        
        X_train_final, y_train_final = XX_dev[:final_train_size], YY_dev[:final_train_size]
        X_val_final, y_val_final = XX_dev[final_train_size:], YY_dev[final_train_size:]
        
        train_loader_final = data_loader(X_train_final, y_train_final, 64, shuffle=True, drop_last=True)
        val_loader_final = data_loader(X_val_final, y_val_final, 64, shuffle=False, drop_last=False)
        
        final_model = SAMBA(
            model_args, best_params['hid'], window, predict,
            best_params['embed_dim'], best_params["cheb_k"]
        ).to(device)
        
        for p in final_model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.uniform_(p)

        loss_fn = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(params=final_model.parameters(), lr=best_params['lr_init'])

        final_trainer = Trainer(
            final_model, loss_fn, optimizer, train_loader_final, val_loader_final,
            args=args, lr_scheduler=None
        )

        print("Starting final training...")
        best_model_state, _ = final_trainer.train()

        torch.save(best_model_state, final_model_file)
        print(f"Final model saved to {final_model_file}")

    # --- 7. STAGE 3: Final Evaluation ---
    print(f"\n===== STAGE 3: Final Evaluation (on 20% Held-Out Test Set) =====")
    
    if not os.path.exists(final_model_file) or not os.path.exists(best_params_file):
        print("Error: Model files not found. Please run in 'train' mode first.")
        return
        
    with open(best_params_file, 'r') as f:
        best_params = json.load(f)
    print(f"Loading model with best params: {best_params}")

    final_model = SAMBA(
        model_args, best_params['hid'], window, predict,
        best_params['embed_dim'], best_params["cheb_k"]
    ).to(device)
    
    final_model.load_state_dict(torch.load(final_model_file, map_location=device))
    
    test_loader_final = data_loader(XX_test, YY_test, 64, shuffle=False, drop_last=False)
    
    # We need a logger for the static test method
    temp_logger = get_logger(output_dir, name='test_run', debug=True)
    y_pred, y_true = Trainer.test(final_model, args, test_loader_final, temp_logger)
    
    # Rescale
    y_p = mmn.inverse_transform(y_pred.cpu().numpy())
    y_t = mmn.inverse_transform(y_true.cpu().numpy())

    # Calculate final metrics
    mae, rmse, _ = All_Metrics(torch.tensor(y_p), torch.tensor(y_t), None, None)
    IC = pearson_correlation(torch.tensor(y_t), torch.tensor(y_p))
    RIC = rank_information_coefficient(torch.tensor(y_t).squeeze(), torch.tensor(y_p).squeeze())

    print("\n===== FINAL MODEL PERFORMANCE ON UNSEEN TEST SET =====")
    print(f"MAE:  {mae.item():.4f}")
    print(f"RMSE: {rmse.item():.4f}")
    print(f"IC:   {IC.item():.4f}")
    print(f"RIC:  {RIC.item() if torch.is_tensor(RIC) else RIC:.4f}")

    # --- Plotting Logic ---
    print("\nPlotting final results...")
    y_t_plot = y_t.squeeze()
    y_p_plot = y_p.squeeze()
    if y_t_plot.ndim == 1: y_t_plot = y_t_plot.reshape(-1, 1)
    if y_p_plot.ndim == 1: y_p_plot = y_p_plot.reshape(-1, 1)

    for i in range(config.horizon):
        plt.figure(figsize=(12, 6))
        plt.plot(y_t_plot[:, i], label=f"Actual (Day {i+1})", linewidth=2)
        plt.plot(y_p_plot[:, i], label=f"Predicted (Day {i+1})", linewidth=2, linestyle="--")
        plt.xlabel("Time (Final Test Samples)")
        plt.ylabel("Stock Price")
        plt.title(f"SAMBA Final Model Prediction - (Day-{i+1} Forecast)")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, f"final_test_plot_day_{i+1}.png")
        plt.savefig(plot_path)
        print(f"Final test plot for day {i+1} saved to {plot_path}")
        plt.close()

    print("\nScript finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAMBA Model Training and Testing (Approach 2)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: "train" (runs all 3 stages) or "test" (runs only stage 3)')
    cli_args = parser.parse_args()
    
    main(cli_args)