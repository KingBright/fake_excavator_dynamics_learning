# -*- coding: utf-8 -*-
# ... (Import statements and helper functions remain the same) ...
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
import random
import copy
import time
from tqdm import tqdm

# ==============================================================================
# LSTM Model Definition (保持不变)
# ==============================================================================
class ExcavatorLSTM(nn.Module):
    # ... (代码同上一版本) ...
    def __init__(self, input_size=13, hidden_size=256, output_size=9, num_layers=3, dropout=0.1):
        super(ExcavatorLSTM, self).__init__()
        self.hidden_size = hidden_size; self.num_layers = num_layers; lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_size, output_size); self._init_weights()
    def forward(self, x, h_c=None):
        device = next(self.parameters()).device
        if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)
        if h_c is None: h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device); c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device); h_c = (h_0, c_0)
        elif isinstance(h_c, tuple): h_c = (h_c[0].to(device), h_c[1].to(device))
        out, (hn, cn) = self.lstm(x, h_c); out = self.fc(out[:, -1, :]); return out, (hn, cn)
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name: nn.init.constant_(param, 0.0)
            elif 'weight' in name: nn.init.xavier_uniform_(param)

# ==============================================================================
# Helper Functions (保持不变)
# ==============================================================================
def create_sequences(input_features, output_labels, seq_length):
    # ... (代码同上一版本) ...
    xs, ys = [], [];
    if len(input_features) < seq_length: return np.empty((0, seq_length, input_features.shape[1])), np.empty((0, output_labels.shape[1]))
    num_sequences = len(input_features) - seq_length
    if num_sequences < 0: return np.empty((0, seq_length, input_features.shape[1])), np.empty((0, output_labels.shape[1]))
    for i in range(num_sequences + 1):
        start_idx = i; end_idx = i + seq_length; label_idx = end_idx -1
        if label_idx < len(output_labels): xs.append(input_features[start_idx:end_idx]); ys.append(output_labels[label_idx])
        # else: break # Stop if labels run out
    if not xs: return np.empty((0, seq_length, input_features.shape[1])), np.empty((0, output_labels.shape[1]))
    return np.array(xs), np.array(ys)

def angle_difference(angle1, angle2):
    # ... (代码同上一版本) ...
    diff = angle1 - angle2; return (diff + np.pi) % (2 * np.pi) - np.pi

# ==============================================================================
# Main Training and Evaluation
# ==============================================================================
if __name__ == "__main__":
    # --- 参数解析 (保持不变) ---
    # ... (代码同上一版本, 包含 clip_rollout_input) ...
    parser = argparse.ArgumentParser(description='Train Excavator Dynamics LSTM (Cos/Sin Angles, Scaled Vel)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .npz data file')
    parser.add_argument('--output_prefix', type=str, default='excavator_lstm_cossin_qvelscaled', help='Prefix for saving files')
    parser.add_argument('--seq_length', type=int, default=50, help='Input sequence length')
    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty)')
    parser.add_argument('--val_split', type=float, default=0.15, help='Fraction of data for validation')
    parser.add_argument('--test_split', type=float, default=0.15, help='Fraction of data for final testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--qpos_indices', type=int, nargs=4, default=[0, 1, 2, 3], help='Indices of controlled joint positions in original data')
    parser.add_argument('--qvel_indices', type=int, nargs=4, default=[0, 1, 2, 3], help='Indices of controlled joint velocities in original data')
    parser.add_argument('--cab_angle_idx', type=int, default=0, help='Index of the Cab angle within the 4 controlled qpos (0-based)')
    parser.add_argument('--clip_rollout_input', type=float, default=None, help='Optional: Clip scaled rollout input features to +/- this value.')
    parser.add_argument('--grad_clip_norm', type=float, default=None, help='Optional: Max norm for gradient clipping.') # 新增梯度裁剪参数

    args = parser.parse_args()

    # --- Setup: Seed, Device, Directories, Paths ---
    # ... (同上一版本) ...
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    SAVE_DIR = f"h{args.hidden_size}_l{args.num_layers}_s{args.seq_length}_lr{args.learning_rate}_wd{args.weight_decay:.0e}_bs{args.batch_size}_cossin"
    if args.grad_clip_norm is not None: SAVE_DIR += f"_gc{args.grad_clip_norm}" # Add grad clip info to dir name
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR); print(f"Created results directory: {SAVE_DIR}")
    else: print(f"Results directory already exists: {SAVE_DIR}")
    scaler_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_scaler_input.joblib")
    scaler_qvel_out_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_scaler_qvel_out.joblib")
    model_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_model_best.pth")
    loss_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_losses.png")
    rollout_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_rollout_vs_single_step.png")
    rollout_error_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_rollout_error.png")
    single_step_error_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_single_step_error.png")


    # --- 1. Load Data ---
    # ... (同上一版本) ...
    print(f"Loading data from {args.data_path}...")
    try: data = np.load(args.data_path); time_data = data['time']; qpos_all = data['qpos']; qvel_all = data['qvel']; control_signal = data['control_signal']; print(f"Data loaded. Total steps: {len(time_data)}")
    except Exception as e: print(f"Error loading data: {e}"); exit()


    # --- 2. Feature Selection & Cos/Sin Engineering ---
    # ... (同上一版本) ...
    qpos_indices = args.qpos_indices; qvel_indices = args.qvel_indices; cab_idx = args.cab_angle_idx
    other_qpos_indices_rel = [i for i in range(4) if i != cab_idx]
    print(f"Using qpos indices: {qpos_indices}, qvel indices: {qvel_indices}"); print(f"Cab angle index relative to controlled qpos: {cab_idx}")
    try: qpos_controlled = qpos_all[:, qpos_indices].copy(); qvel_controlled = qvel_all[:, qvel_indices]
    except IndexError as e: print(f"Error selecting indices: {e}. Check indices."); exit()
    T = min(len(qpos_controlled), len(qvel_controlled), len(control_signal))
    if T <= 1: print("Error: Not enough data points (<= 1) after checking lengths."); exit()
    control_feat = control_signal[:T-1]; qvel_feat = qvel_controlled[:T-1]; qpos_feat_all = qpos_controlled[:T-1]
    cab_angle_raw = qpos_feat_all[:, cab_idx]; cab_cos_feat = np.cos(cab_angle_raw); cab_sin_feat = np.sin(cab_angle_raw)
    other_qpos_feat = qpos_feat_all[:, other_qpos_indices_rel]
    input_features = np.concatenate((control_feat, cab_cos_feat[:, None], cab_sin_feat[:, None], other_qpos_feat, qvel_feat), axis=1)
    input_size = input_features.shape[1]; print(f"Input feature size: {input_size}")
    qpos_labels_all = qpos_controlled[1:T]; qvel_labels_all = qvel_controlled[1:T]
    cab_angle_labels_raw = qpos_labels_all[:, cab_idx]; cab_cos_labels = np.cos(cab_angle_labels_raw); cab_sin_labels = np.sin(cab_angle_labels_raw)
    other_qpos_labels = qpos_labels_all[:, other_qpos_indices_rel]
    output_labels_unscaled_vel = np.concatenate((cab_cos_labels[:, None], cab_sin_labels[:, None], other_qpos_labels, qvel_labels_all), axis=1)
    output_size = output_labels_unscaled_vel.shape[1]; print(f"Output label size: {output_size}")
    time_labels = time_data[1:T]
    print(f"Input features shape: {input_features.shape}"); print(f"Output labels shape (before vel scaling): {output_labels_unscaled_vel.shape}"); print(f"Time labels shape: {time_labels.shape}")
    original_output_qpos = qpos_labels_all.copy(); original_output_qvel = qvel_labels_all.copy()


    # --- 3. Data Splitting (Train, Validation, Test) ---
    # ... (同上一版本) ...
    print("Splitting data into Train, Validation, Test sets...")
    if args.val_split + args.test_split >= 1.0: print("Error: Sum of validation and test splits must be less than 1."); exit()
    X_train_val, X_test, y_train_val_unscaled_vel, y_test_unscaled_vel, time_train_val, time_test = train_test_split(input_features, output_labels_unscaled_vel, time_labels, test_size=args.test_split, random_state=args.seed, shuffle=False)
    train_val_ratio = 1.0 - args.test_split; val_ratio_in_train_val = args.val_split / train_val_ratio
    if val_ratio_in_train_val >= 1.0: raise ValueError("val_split is too large relative to test_split")
    val_split_index = int(len(X_train_val) * (1.0 - val_ratio_in_train_val))
    X_train = X_train_val[:val_split_index]; X_val = X_train_val[val_split_index:]; y_train_unscaled_vel = y_train_val_unscaled_vel[:val_split_index]; y_val_unscaled_vel = y_train_val_unscaled_vel[val_split_index:]; time_train = time_train_val[:val_split_index]; time_val = time_train_val[val_split_index:]
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")


    # --- 4. Preprocessing (Scale Inputs, Scale Output Velocities) ---
    # ... (同上一版本) ...
    print("Scaling input features (fit on train)...")
    scaler_input = StandardScaler();
    if len(X_train) == 0: print("Error: Training set has size 0 after splitting."); exit()
    X_train_scaled = scaler_input.fit_transform(X_train)
    if len(X_val) > 0: X_val_scaled = scaler_input.transform(X_val); else: X_val_scaled = np.empty((0, X_train.shape[1]))
    if len(X_test) > 0: X_test_scaled = scaler_input.transform(X_test); else: X_test_scaled = np.empty((0, X_train.shape[1]))
    joblib.dump(scaler_input, scaler_filename); print(f"Input feature scaler saved to {scaler_filename}")
    print("Scaling ONLY output velocities (qvel part of labels, fit on train)...")
    qvel_output_indices = list(range(output_size - 4, output_size))
    y_train_non_vel = y_train_unscaled_vel[:, :qvel_output_indices[0]]; y_train_vel = y_train_unscaled_vel[:, qvel_output_indices]
    y_val_non_vel = y_val_unscaled_vel[:, :qvel_output_indices[0]]; y_val_vel = y_val_unscaled_vel[:, qvel_output_indices]
    y_test_non_vel = y_test_unscaled_vel[:, :qvel_output_indices[0]]; y_test_vel = y_test_unscaled_vel[:, qvel_output_indices]
    scaler_qvel_out = StandardScaler()
    if len(y_train_vel) == 0: print("Error: Training velocity labels have size 0."); exit()
    y_train_vel_scaled = scaler_qvel_out.fit_transform(y_train_vel)
    if len(y_val_vel) > 0: y_val_vel_scaled = scaler_qvel_out.transform(y_val_vel); else: y_val_vel_scaled = np.empty((0, 4))
    if len(y_test_vel) > 0: y_test_vel_scaled = scaler_qvel_out.transform(y_test_vel); else: y_test_vel_scaled = np.empty((0, 4))
    joblib.dump(scaler_qvel_out, scaler_qvel_out_filename); print(f"Output qvel scaler saved to {scaler_qvel_out_filename}")
    y_train_processed = np.hstack((y_train_non_vel, y_train_vel_scaled)); y_val_processed = np.hstack((y_val_non_vel, y_val_vel_scaled)); y_test_processed = np.hstack((y_test_non_vel, y_test_vel_scaled))
    y_test_original = y_test_unscaled_vel.copy() # Contains cos, sin, other_qp, qv_orig


    # --- 5. Sequence Creation ---
    # ... (同上一版本) ...
    print(f"Creating sequences with length {args.seq_length}...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_processed, args.seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_processed, args.seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_processed, args.seq_length)
    time_train_seq_end = time_train[args.seq_length-1:] if len(time_train) >= args.seq_length else np.array([])
    time_val_seq_end = time_val[args.seq_length-1:] if len(time_val) >= args.seq_length else np.array([])
    time_test_seq_end = time_test[args.seq_length-1:] if len(time_test) >= args.seq_length else np.array([])
    train_seq_len = min(len(X_train_seq), len(y_train_seq), len(time_train_seq_end)); val_seq_len = min(len(X_val_seq), len(y_val_seq), len(time_val_seq_end)); test_seq_len = min(len(X_test_seq), len(y_test_seq), len(time_test_seq_end))
    X_train_seq, y_train_seq = X_train_seq[:train_seq_len], y_train_seq[:train_seq_len]; X_val_seq, y_val_seq = X_val_seq[:val_seq_len], y_val_seq[:val_seq_len]; X_test_seq, y_test_seq = X_test_seq[:test_seq_len], y_test_seq[:test_seq_len]
    time_train_seq_end = time_train_seq_end[:train_seq_len]; time_val_seq_end = time_val_seq_end[:val_seq_len]; time_test_seq_end = time_test_seq_end[:test_seq_len]
    print(f"Train sequences final shape: X={X_train_seq.shape}, y={y_train_seq.shape}, t={time_train_seq_end.shape}")
    print(f"Validation sequences final shape: X={X_val_seq.shape}, y={y_val_seq.shape}, t={time_val_seq_end.shape}")
    print(f"Test sequences final shape: X={X_test_seq.shape}, y={y_test_seq.shape}, t={time_test_seq_end.shape}")

    # --- 6. Prepare PyTorch DataLoaders ---
    # ... (同上一版本) ...
    if X_train_seq.shape[0] == 0: print("Error: No training sequences created."); exit()
    if X_val_seq.shape[0] == 0: print("Warning: No validation sequences created.")
    if X_test_seq.shape[0] == 0: print("Warning: No test sequences created.")
    train_dataset = TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(y_train_seq).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val_seq).float(), torch.from_numpy(y_val_seq).float()) if X_val_seq.shape[0]>0 else None
    test_dataset = TensorDataset(torch.from_numpy(X_test_seq).float(), torch.from_numpy(y_test_seq).float()) if X_test_seq.shape[0]>0 else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if test_dataset else None

    # --- 7. Model, Loss, Optimizer ---
    # ... (同上一版本) ...
    model = ExcavatorLSTM(input_size, args.hidden_size, output_size, args.num_layers, args.dropout).to(device)
    criterion = nn.MSELoss(); print("Using standard nn.MSELoss.")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Using Adam optimizer with lr={args.learning_rate} and weight_decay={args.weight_decay}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"Model Summary: {num_params} trainable parameters")


    # --- 8. Training Loop with Validation (修改: 加入梯度裁剪) ---
    print("Starting training...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 15
    total_start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start_time = time.time(); model.train(); epoch_train_loss = 0
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Train", leave=False)
        for batch_x, batch_y_processed in progress_bar_train:
            batch_x, batch_y_processed = batch_x.to(device), batch_y_processed.to(device)
            optimizer.zero_grad(); outputs_processed, _ = model(batch_x); loss = criterion(outputs_processed, batch_y_processed); loss.backward()
            # --- 加入梯度裁剪 ---
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            # --------------------
            optimizer.step(); epoch_train_loss += loss.item(); progress_bar_train.set_postfix(loss=f"{loss.item():.6f}")
        avg_train_loss = epoch_train_loss / max(1, len(train_loader)); train_losses.append(avg_train_loss)

        model.eval(); epoch_val_loss = 0
        if val_loader:
            with torch.no_grad():
                progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Val", leave=False)
                for batch_x_val, batch_y_val_processed in progress_bar_val:
                    batch_x_val, batch_y_val_processed = batch_x_val.to(device), batch_y_val_processed.to(device)
                    outputs_val_processed, _ = model(batch_x_val); loss_val = criterion(outputs_val_processed, batch_y_val_processed); epoch_val_loss += loss_val.item(); progress_bar_val.set_postfix(loss=f"{loss_val.item():.6f}")
            avg_val_loss = epoch_val_loss / len(val_loader); val_losses.append(avg_val_loss)
        else: avg_val_loss = float('inf'); val_losses.append(avg_val_loss)

        epoch_end_time = time.time(); epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {epoch_duration:.2f}s')

        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.6f} --> {avg_val_loss:.6f}). Saving model...")
            best_val_loss = avg_val_loss; torch.save(model.state_dict(), model_filename); epochs_no_improve = 0
        else:
            epochs_no_improve += 1; print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience: print(f"Early stopping triggered after {patience} epochs without improvement."); break
    total_end_time = time.time(); total_training_time = total_end_time - total_start_time
    print(f"Training finished. Total time: {total_training_time:.2f}s"); print(f"Best validation loss: {best_val_loss:.6f}")


    # --- 9. Load Best Model ---
    # ... (同上一版本) ...
    if os.path.exists(model_filename):
        print(f"Loading best model from {model_filename} for final evaluation...")
        model = ExcavatorLSTM(input_size, args.hidden_size, output_size, args.num_layers, args.dropout).to(device)
        try: model.load_state_dict(torch.load(model_filename, map_location=device)); print("Best model loaded successfully.")
        except Exception as e: print(f"Error loading model state_dict: {e}. Evaluating model from last epoch.")
    else: print("Warning: No best model file found. Evaluating model from last epoch.")


    # --- 10. Final Evaluation on Test Set (Single Step) ---
    # ... (同上一版本) ...
    if test_loader is None: print("Skipping test set evaluation as test set is empty."); all_test_preds_processed = np.empty((0, output_size)); test_seq_len = 0
    else:
        print("Evaluating model on Test Set (Single Step)..."); model.eval(); test_loss_final = 0; all_test_preds_processed_list = []
        with torch.no_grad():
            for batch_x_test, batch_y_test_processed in tqdm(test_loader, desc="Testing (Single Step)", leave=False):
                batch_x_test, batch_y_test_processed = batch_x_test.to(device), batch_y_test_processed.to(device)
                outputs_test_processed, _ = model(batch_x_test); loss_test = criterion(outputs_test_processed, batch_y_test_processed); test_loss_final += loss_test.item()
                all_test_preds_processed_list.append(outputs_test_processed.cpu().numpy())
        avg_test_loss_final = test_loss_final / max(1, len(test_loader)); print(f"Average Test Set Loss (Single Step, MSE): {avg_test_loss_final:.6f}")
        all_test_preds_processed = np.concatenate(all_test_preds_processed_list, axis=0)
        # test_seq_len is already defined


    # --- 11. Inverse Transform Single Step Predictions & Recover Angles ---
    # ... (同上一版本) ...
    if test_seq_len > 0 and len(all_test_preds_processed) == test_seq_len:
        print("Recovering angles and inverse transforming velocities for single-step predictions...")
        pred_cab_cos = all_test_preds_processed[:, 0]; pred_cab_sin = all_test_preds_processed[:, 1]; pred_other_qpos_orig = all_test_preds_processed[:, 2:5]; pred_qvel_scaled = all_test_preds_processed[:, 5:]
        single_step_pred_cab_angle_orig = np.arctan2(pred_cab_sin, pred_cab_cos)
        try: single_step_pred_qvel_orig = scaler_qvel_out.inverse_transform(pred_qvel_scaled)
        except Exception as e: print(f"Error during qvel inverse transform: {e}"); single_step_pred_qvel_orig = np.full_like(pred_qvel_scaled, np.nan)
        single_step_pred_qpos_orig = np.zeros((len(all_test_preds_processed), 4)); single_step_pred_qpos_orig[:, cab_idx] = single_step_pred_cab_angle_orig
        current_col = 0
        for idx in range(4):
             if idx != cab_idx: single_step_pred_qpos_orig[:, idx] = pred_other_qpos_orig[:, current_col]; current_col += 1
        # Get corresponding Ground Truth in original qpos/qvel scale
        gt_align_start_idx = len(X_train) + len(X_val) + args.seq_length - 1
        gt_align_end_idx = gt_align_start_idx + test_seq_len
        if gt_align_end_idx > len(original_output_qpos): gt_align_end_idx = len(original_output_qpos); test_seq_len_adjusted = gt_align_end_idx - gt_align_start_idx; print(f"Warning: Ground truth alignment index out of bounds. Adjusted test seq len: {test_seq_len_adjusted}")
        else: test_seq_len_adjusted = test_seq_len
        if test_seq_len_adjusted < len(all_test_preds_processed): print(f"Trimming predictions to match available ground truth: {test_seq_len_adjusted} steps."); single_step_pred_qpos_orig = single_step_pred_qpos_orig[:test_seq_len_adjusted]; single_step_pred_qvel_orig = single_step_pred_qvel_orig[:test_seq_len_adjusted]; all_test_preds_processed = all_test_preds_processed[:test_seq_len_adjusted]; test_seq_len = test_seq_len_adjusted # Update test_seq_len to actual used length
        ground_truth_qpos_orig_for_test_seq = original_output_qpos[gt_align_start_idx:gt_align_end_idx, :]; ground_truth_qvel_orig_for_test_seq = original_output_qvel[gt_align_start_idx:gt_align_end_idx, :]
    else: print("No single-step predictions to process."); single_step_pred_qpos_orig = np.array([]); single_step_pred_qvel_orig = np.array([]); ground_truth_qpos_orig_for_test_seq = np.array([]); ground_truth_qvel_orig_for_test_seq = np.array([])


    # --- 11.5 Calculate Single Step Errors & Metrics ---
    # ... (同上一版本, 增加空集检查) ...
    joint_names = ['Cab', 'Boom', 'Arm', 'Bucket']
    if test_seq_len > 0 and len(single_step_pred_qpos_orig) == test_seq_len:
        print("\n--- Single-Step Prediction Metrics (Full Test Set) ---")
        single_step_qpos_error_full = np.zeros_like(single_step_pred_qpos_orig); single_step_qpos_error_full[:, cab_idx] = angle_difference(single_step_pred_qpos_orig[:, cab_idx], ground_truth_qpos_orig_for_test_seq[:, cab_idx])
        current_col = 0
        for idx in range(4):
            if idx != cab_idx: single_step_qpos_error_full[:, idx] = single_step_pred_qpos_orig[:, idx] - ground_truth_qpos_orig_for_test_seq[:, idx]
        single_step_qvel_error_full = single_step_pred_qvel_orig - ground_truth_qvel_orig_for_test_seq
        ss_qpos_mae = np.mean(np.abs(single_step_qpos_error_full), axis=0); ss_qvel_mae = np.mean(np.abs(single_step_qvel_error_full), axis=0); ss_qpos_rmse = np.sqrt(np.mean(single_step_qpos_error_full**2, axis=0)); ss_qvel_rmse = np.sqrt(np.mean(single_step_qvel_error_full**2, axis=0))
        print("MAE:"); [print(f"  {name}: Pos={np.rad2deg(ss_qpos_mae[i]):.4f} deg, Vel={ss_qvel_mae[i]:.6f} rad/s") for i, name in enumerate(joint_names)]
        print("RMSE:"); [print(f"  {name}: Pos={np.rad2deg(ss_qpos_rmse[i]):.4f} deg, Vel={ss_qvel_rmse[i]:.6f} rad/s") for i, name in enumerate(joint_names)]
        peak_qpos_error = np.max(np.abs(single_step_qpos_error_full), axis=0); peak_qvel_error = np.max(np.abs(single_step_qvel_error_full), axis=0)
        print("Peak Absolute Error:"); [print(f"  {name}: Pos={np.rad2deg(peak_qpos_error[i]):.4f} deg, Vel={peak_qvel_error[i]:.6f} rad/s") for i, name in enumerate(joint_names)]
        threshold = 1e-3; correct_sign = np.zeros(4); non_zero_steps = np.zeros(4)
        for j in range(4):
            valid_steps_mask = np.abs(ground_truth_qvel_orig_for_test_seq[:, j]) > threshold; non_zero_steps[j] = np.sum(valid_steps_mask)
            if non_zero_steps[j] > 0: sign_true = np.sign(ground_truth_qvel_orig_for_test_seq[valid_steps_mask, j]); sign_pred = np.sign(single_step_pred_qvel_orig[valid_steps_mask, j]); correct_sign[j] = np.sum(sign_true == sign_pred)
        direction_accuracy = np.zeros(4); mask_nz = non_zero_steps > 0; direction_accuracy[mask_nz] = (correct_sign[mask_nz] / non_zero_steps[mask_nz]) * 100
        print("Velocity Direction Accuracy (% of non-zero steps):"); [print(f"  {name}: {direction_accuracy[i]:.2f}% ({int(correct_sign[i])}/{int(non_zero_steps[i])} steps)") for i, name in enumerate(joint_names)]
        correlations_qpos = []; r2_scores_qpos = []; correlations_qvel = []; r2_scores_qvel = []
        print("Correlation (R) and R2 Score:")
        for j in range(4):
            mask_qpos = ~np.isnan(single_step_pred_qpos_orig[:, j]) & ~np.isnan(ground_truth_qpos_orig_for_test_seq[:, j]); mask_qvel = ~np.isnan(single_step_pred_qvel_orig[:, j]) & ~np.isnan(ground_truth_qvel_orig_for_test_seq[:, j])
            if np.sum(mask_qpos) > 1: corr_qpos, _ = pearsonr(ground_truth_qpos_orig_for_test_seq[mask_qpos, j], single_step_pred_qpos_orig[mask_qpos, j]); try: r2_qpos = r2_score(ground_truth_qpos_orig_for_test_seq[mask_qpos, j], single_step_pred_qpos_orig[mask_qpos, j]); except ValueError: r2_qpos = np.nan
            else: corr_qpos, r2_qpos = np.nan, np.nan
            correlations_qpos.append(corr_qpos); r2_scores_qpos.append(r2_qpos)
            if np.sum(mask_qvel) > 1: corr_qvel, _ = pearsonr(ground_truth_qvel_orig_for_test_seq[mask_qvel, j], single_step_pred_qvel_orig[mask_qvel, j]); try: r2_qvel = r2_score(ground_truth_qvel_orig_for_test_seq[mask_qvel, j], single_step_pred_qvel_orig[mask_qvel, j]); except ValueError: r2_qvel = np.nan
            else: corr_qvel, r2_qvel = np.nan, np.nan
            correlations_qvel.append(corr_qvel); r2_scores_qvel.append(r2_qvel)
            print(f"  {joint_names[j]}: Pos R={corr_qpos:.4f}, R2={r2_qpos:.4f} | Vel R={corr_qvel:.4f}, R2={r2_qvel:.4f}")
    else: print("Skipping single-step metric calculation due to missing predictions or ground truth.")


    # --- 12. Multi-step Rollout Prediction ---
    # ... (同上一版本) ...
    print("\nPerforming multi-step rollout on a test segment...")
    start_index_in_test_seq = 0
    if X_test_seq is None or len(X_test_seq) <= start_index_in_test_seq: print("Not enough test sequences for rollout evaluation."); predictions_rollout_pos = np.array([]); predictions_rollout_vel = np.array([]); ground_truth_rollout_pos = np.array([]); ground_truth_rollout_vel = np.array([]); rollout_steps = 0
    else:
        rollout_steps = len(X_test_seq) - start_index_in_test_seq
        current_sequence_rollout = torch.from_numpy(X_test_seq[start_index_in_test_seq:start_index_in_test_seq+1]).float().to(device)
        predictions_rollout_pos_list = []; predictions_rollout_vel_list = []
        if len(ground_truth_qpos_orig_for_test_seq) > 0: initial_rollout_qpos_true = ground_truth_qpos_orig_for_test_seq[0]; initial_rollout_qvel_true = ground_truth_qvel_orig_for_test_seq[0]
        else: print("Error: Cannot get initial state for rollout."); rollout_steps = 0; initial_rollout_qpos_true = np.zeros(4); initial_rollout_qvel_true = np.zeros(4)
        predictions_rollout_pos_list.append(initial_rollout_qpos_true); predictions_rollout_vel_list.append(initial_rollout_qvel_true)
        current_h_c_rollout = None; last_predicted_qpos = initial_rollout_qpos_true.copy(); last_predicted_qvel = initial_rollout_qvel_true.copy()
        model.eval()
        with torch.no_grad():
            for i in range(rollout_steps - 1):
                predicted_output_processed_tensor, current_h_c_rollout = model(current_sequence_rollout, current_h_c_rollout)
                predicted_output_processed = predicted_output_processed_tensor.cpu().numpy().flatten()
                pred_cab_cos_rollout = predicted_output_processed[0]; pred_cab_sin_rollout = predicted_output_processed[1]; pred_other_qpos_orig_rollout = predicted_output_processed[2:5]; pred_qvel_scaled_rollout = predicted_output_processed[5:]
                next_qpos_pred_orig = np.zeros(4); next_qpos_pred_orig[cab_idx] = np.arctan2(pred_cab_sin_rollout, pred_cab_cos_rollout)
                current_col = 0
                for idx in range(4):
                     if idx != cab_idx: next_qpos_pred_orig[idx] = pred_other_qpos_orig_rollout[current_col]; current_col += 1
                next_qvel_pred_orig = scaler_qvel_out.inverse_transform(pred_qvel_scaled_rollout.reshape(1, -1)).flatten()
                predictions_rollout_pos_list.append(next_qpos_pred_orig); predictions_rollout_vel_list.append(next_qvel_pred_orig)
                last_predicted_qpos = next_qpos_pred_orig; last_predicted_qvel = next_qvel_pred_orig
                control_signal_idx_in_X_test = start_index_in_test_seq + args.seq_length + i # Index relative to start of X_test features
                if control_signal_idx_in_X_test >= len(X_test): print(f"Rollout stopped early at step {i+1}: Reached end of available control signals in X_test."); rollout_steps = i + 2; break
                next_control_signal = X_test[control_signal_idx_in_X_test, :4]
                next_cab_angle = last_predicted_qpos[cab_idx]; next_cab_cos = np.cos(next_cab_angle); next_cab_sin = np.sin(next_cab_angle)
                next_other_qpos = last_predicted_qpos[other_qpos_indices_rel]; next_qvel = last_predicted_qvel
                next_input_frame_unscaled = np.concatenate((next_control_signal, [next_cab_cos], [next_cab_sin], next_other_qpos, next_qvel))
                next_input_frame_scaled = scaler_input.transform(next_input_frame_unscaled.reshape(1, -1)).flatten()
                if args.clip_rollout_input is not None: clip_val = abs(args.clip_rollout_input); next_input_frame_scaled = np.clip(next_input_frame_scaled, -clip_val, clip_val)
                current_sequence_np = current_sequence_rollout.cpu().numpy()[0]; next_sequence_np = np.vstack((current_sequence_np[1:, :], next_input_frame_scaled))
                current_sequence_rollout = torch.from_numpy(next_sequence_np).unsqueeze(0).float().to(device)
        predictions_rollout_pos = np.array(predictions_rollout_pos_list); predictions_rollout_vel = np.array(predictions_rollout_vel_list)
        # Adjust ground truth length based on actual rollout steps performed
        ground_truth_rollout_pos = ground_truth_qpos_orig_for_test_seq[:rollout_steps]; ground_truth_rollout_vel = ground_truth_qvel_orig_for_test_seq[:rollout_steps]


    # --- 13. Get Single Step Predictions for the Rollout Segment ---
    # ... (同上一版本) ...
    if rollout_steps > 0 and len(single_step_pred_qpos_orig) >= rollout_steps:
       single_step_pred_qpos_segment = single_step_pred_qpos_orig[start_index_in_test_seq : start_index_in_test_seq + rollout_steps]
       single_step_pred_vel_segment = single_step_pred_qvel_orig[start_index_in_test_seq : start_index_in_test_seq + rollout_steps]
    else:
       single_step_pred_qpos_segment = np.array([]); single_step_pred_vel_segment = np.array([])
       if rollout_steps > 0 and len(single_step_pred_qpos_segment)==0: print("Warning: Could not extract single step segment, clearing rollout results for plotting."); predictions_rollout_pos = np.array([]); predictions_rollout_vel = np.array([]); ground_truth_rollout_pos = np.array([]); ground_truth_rollout_vel = np.array([])


    # --- 14. Plotting Comparison (修改: 绘制 Cos/Sin 对比图) ---
    print("\nPlotting comparison results...")
    # Plot Train/Validation Loss (High DPI)
    plt.figure(figsize=(10, 5)); plt.plot(train_losses, label='Training Loss'); plt.plot(val_losses, label='Validation Loss')
    if val_losses: best_epoch_idx = np.argmin(val_losses); plt.axvline(x=best_epoch_idx, color='r', linestyle='--', label=f'Best Epoch ({best_epoch_idx+1})')
    plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.ylim(bottom=0); plt.legend(); plt.grid(True); plt.savefig(loss_plot_save_path, dpi=300); plt.close(); print(f"Loss plot saved to {loss_plot_save_path}")

    # Plot Rollout vs Single Step vs Ground Truth
    # Ensure data exists and lengths are consistent for plotting
    plot_len = 0
    if (len(predictions_rollout_pos) > 0 and
        len(single_step_pred_qpos_segment) > 0 and
        len(ground_truth_rollout_pos) > 0):
        plot_len = min(len(predictions_rollout_pos), len(single_step_pred_qpos_segment), len(ground_truth_rollout_pos))
        if plot_len <= 1: # Need more than one point to plot lines
             print("Warning: Not enough consistent data points across predictions and ground truth for comparison plot.")
             plot_len = 0 # Prevent plotting
        elif len(time_test_seq_end) < plot_len:
             print(f"Warning: Time axis length ({len(time_test_seq_end)}) is less than plot length ({plot_len}). Adjusting plot length.")
             plot_len = len(time_test_seq_end)
             # Adjust arrays again if needed, though should be consistent now
             predictions_rollout_pos = predictions_rollout_pos[:plot_len]; predictions_rollout_vel = predictions_rollout_vel[:plot_len]
             single_step_pred_qpos_segment = single_step_pred_qpos_segment[:plot_len]; single_step_pred_vel_segment = single_step_pred_vel_segment[:plot_len]
             ground_truth_rollout_pos = ground_truth_rollout_pos[:plot_len]; ground_truth_rollout_vel = ground_truth_rollout_vel[:plot_len]

    if plot_len > 0:
        num_joints_to_plot = 4; fig, axes = plt.subplots(num_joints_to_plot, 2, figsize=(18, 3.5 * num_joints_to_plot), sharex=True)
        if num_joints_to_plot == 1: axes = np.array([[axes[0], axes[1]]])
        time_axis_start_idx = start_index_in_test_seq; time_axis_end_idx = time_axis_start_idx + plot_len
        time_axis = time_test_seq_end[time_axis_start_idx:time_axis_end_idx]; time_axis = time_axis - time_axis[0]
        # joint_names defined earlier

        # --- 修改: 第一行绘制 Cos/Sin ---
        # Get Processed Predictions (Single Step) for the segment
        single_step_preds_processed_segment = all_test_preds_processed[start_index_in_test_seq : start_index_in_test_seq + plot_len]
        # Get Processed Ground Truth for the segment (from y_test_seq)
        ground_truth_processed_segment = y_test_seq[start_index_in_test_seq : start_index_in_test_seq + plot_len]
        # Get Processed Predictions (Rollout) - Need to reconstruct from rollout pos/vel
        # This requires running the prediction->reconstruction inside the plot loop, or storing processed rollout preds
        # Let's skip plotting rollout cos/sin for simplicity now, focus on single-step vs GT cos/sin

        # Plot Cos(Cab)
        axes[0, 0].plot(time_axis, ground_truth_processed_segment[:, 0], label='GT Cos(Cab)', color='blue', linewidth=1.5, alpha=0.8)
        axes[0, 0].plot(time_axis, single_step_preds_processed_segment[:, 0], label='SS Pred Cos(Cab)', color='green', linestyle=':', linewidth=1, alpha=0.8)
        # axes[0, 0].plot(time_axis, np.cos(predictions_rollout_pos[:plot_len, cab_idx]), label='Rollout Cos(Cab)', color='red', linestyle='--', linewidth=1, alpha=0.8) # Rollout Cos
        axes[0, 0].set_ylabel('Cos(Cab)'); axes[0, 0].legend(); axes[0, 0].grid(True); axes[0, 0].set_ylim([-1.1, 1.1])
        # Plot Sin(Cab)
        axes[0, 1].plot(time_axis, ground_truth_processed_segment[:, 1], label='GT Sin(Cab)', color='blue', linewidth=1.5, alpha=0.8)
        axes[0, 1].plot(time_axis, single_step_preds_processed_segment[:, 1], label='SS Pred Sin(Cab)', color='green', linestyle=':', linewidth=1, alpha=0.8)
        # axes[0, 1].plot(time_axis, np.sin(predictions_rollout_pos[:plot_len, cab_idx]), label='Rollout Sin(Cab)', color='red', linestyle='--', linewidth=1, alpha=0.8) # Rollout Sin
        axes[0, 1].set_ylabel('Sin(Cab)'); axes[0, 1].legend(); axes[0, 1].grid(True); axes[0, 1].set_ylim([-1.1, 1.1])
        # --- 结束修改 ---

        # Plot other joints (Position and Velocity) as before
        for i in range(1, num_joints_to_plot): # Start from joint index 1
            joint_idx = i # Assuming Boom=1, Arm=2, Bucket=3 if Cab=0
            axes[i, 0].plot(time_axis, np.rad2deg(ground_truth_rollout_pos[:plot_len, joint_idx]), label='Ground Truth', color='blue', linewidth=1.5, alpha=0.8)
            axes[i, 0].plot(time_axis, np.rad2deg(predictions_rollout_pos[:plot_len, joint_idx]), label='Rollout Pred', color='red', linestyle='--', linewidth=1, alpha=0.8)
            axes[i, 0].plot(time_axis, np.rad2deg(single_step_pred_qpos_segment[:plot_len, joint_idx]), label='Single-Step Pred', color='green', linestyle=':', linewidth=1, alpha=0.8)
            axes[i, 0].set_ylabel(f'{joint_names[joint_idx]} Pos (deg)'); axes[i, 0].legend(); axes[i, 0].grid(True)
            axes[i, 1].plot(time_axis, ground_truth_rollout_vel[:plot_len, joint_idx], label='Ground Truth', color='blue', linewidth=1.5, alpha=0.8)
            axes[i, 1].plot(time_axis, predictions_rollout_vel[:plot_len, joint_idx], label='Rollout Pred', color='red', linestyle='--', linewidth=1, alpha=0.8)
            axes[i, 1].plot(time_axis, single_step_pred_vel_segment[:plot_len, joint_idx], label='Single-Step Pred', color='green', linestyle=':', linewidth=1, alpha=0.8)
            axes[i, 1].set_ylabel(f'{joint_names[joint_idx]} Vel (rad/s)'); axes[i, 1].legend(); axes[i, 1].grid(True)

        axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')
        fig.suptitle('Single-Step vs Multi-step Rollout Prediction vs Ground Truth (Cos/Sin Encoded Cab)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.savefig(rollout_plot_save_path, dpi=300); plt.close(fig); print(f"Comparison plot saved to {rollout_plot_save_path}")
    else: print("Skipping rollout comparison plot due to empty prediction/ground truth arrays or insufficient data.")


    # --- 15. Calculate & Print Rollout Errors ---
    # ... (同上一版本, 增加空集检查) ...
    if (len(predictions_rollout_pos) > 0 and len(ground_truth_rollout_pos) > 0 and len(predictions_rollout_pos) == len(ground_truth_rollout_pos)):
        # ... (Error calculation logic remains the same) ...
        rollout_qpos_error = np.zeros_like(predictions_rollout_pos); rollout_qpos_error[:, cab_idx] = angle_difference(predictions_rollout_pos[:, cab_idx], ground_truth_rollout_pos[:, cab_idx])
        current_col = 0
        for idx in range(4):
             if idx != cab_idx: rollout_qpos_error[:, idx] = predictions_rollout_pos[:, idx] - ground_truth_rollout_pos[:, idx]
        rollout_qvel_error = predictions_rollout_vel - ground_truth_rollout_vel
        rollout_qpos_mae = np.mean(np.abs(rollout_qpos_error)); rollout_qvel_mae = np.mean(np.abs(rollout_qvel_error)); rollout_qpos_rmse = np.sqrt(np.mean(rollout_qpos_error**2)); rollout_qvel_rmse = np.sqrt(np.mean(rollout_qvel_error**2))
        print("\n--- Rollout Prediction Errors (Test Segment) ---"); print(f"Qpos MAE: {rollout_qpos_mae:.6f} rad ({np.rad2deg(rollout_qpos_mae):.4f} deg)"); print(f"Qvel MAE: {rollout_qvel_mae:.6f} rad/s"); print(f"Qpos RMSE: {rollout_qpos_rmse:.6f} rad ({np.rad2deg(rollout_qpos_rmse):.4f} deg)"); print(f"Qvel RMSE: {rollout_qvel_rmse:.6f} rad/s")

        # Rollout Error Plot (High DPI)
        plot_len_err = len(rollout_qpos_error)
        if plot_len_err > 0 and len(time_test_seq_end) >= plot_len_err:
            time_axis_err = time_test_seq_end[start_index_in_test_seq : start_index_in_test_seq + plot_len_err]; time_axis_err = time_axis_err - time_axis_err[0]
            fig_err, axes_err = plt.subplots(num_joints_to_plot, 2, figsize=(18, 3.5 * num_joints_to_plot), sharex=True)
            if num_joints_to_plot == 1: axes_err = np.array([[axes_err[0], axes_err[1]]])
            fig_err.suptitle('Rollout Prediction Error (Prediction - Ground Truth)')
            for i in range(num_joints_to_plot):
                 axes_err[i, 0].plot(time_axis_err, np.rad2deg(rollout_qpos_error[:, i]), label=f'{joint_names[i]} Pos Error', color='green', linewidth=1); axes_err[i, 0].axhline(0, color='gray', linestyle='--', linewidth=0.8); axes_err[i, 0].set_ylabel('Error (deg)'); axes_err[i, 0].legend(); axes_err[i, 0].grid(True)
                 axes_err[i, 1].plot(time_axis_err, rollout_qvel_error[:, i], label=f'{joint_names[i]} Vel Error', color='purple', linewidth=1); axes_err[i, 1].axhline(0, color='gray', linestyle='--', linewidth=0.8); axes_err[i, 1].set_ylabel('Error (rad/s)'); axes_err[i, 1].legend(); axes_err[i, 1].grid(True)
            axes_err[-1, 0].set_xlabel('Time (s)'); axes_err[-1, 1].set_xlabel('Time (s)')
            plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.savefig(rollout_error_plot_save_path, dpi=300); plt.close(fig_err); print(f"Rollout error plot saved to {rollout_error_plot_save_path}")
        else: print("Skipping rollout error plot due to insufficient data or time axis mismatch.")
    else: print("Skipping rollout error calculation and plotting due to missing data.")


    # --- 16. Plot Single Step Errors (Full Test Set) ---
    # ... (同上一版本, 增加空集检查, 使用 dpi=300) ...
    if test_seq_len > 0 and len(single_step_qpos_error_full) == test_seq_len:
        print("\nPlotting single-step errors...")
        num_joints_to_plot = 4; fig_ss_err, axes_ss_err = plt.subplots(num_joints_to_plot, 2, figsize=(18, 3.5 * num_joints_to_plot), sharex=True)
        if num_joints_to_plot == 1: axes_ss_err = np.array([[axes_ss_err[0], axes_ss_err[1]]])
        fig_ss_err.suptitle('Single-Step Prediction Error (Prediction - Ground Truth, Full Test Set)')
        time_axis_full_test = time_test_seq_end[:test_seq_len]; time_axis_full_test = time_axis_full_test - time_axis_full_test[0]
        for i in range(num_joints_to_plot):
            axes_ss_err[i, 0].plot(time_axis_full_test, np.rad2deg(single_step_qpos_error_full[:, i]), label=f'{joint_names[i]} Pos Error', color='orange', linewidth=1); axes_ss_err[i, 0].axhline(0, color='gray', linestyle='--', linewidth=0.8); axes_ss_err[i, 0].set_ylabel('Error (deg)'); axes_ss_err[i, 0].legend(); axes_ss_err[i, 0].grid(True)
            axes_ss_err[i, 1].plot(time_axis_full_test, single_step_qvel_error_full[:, i], label=f'{joint_names[i]} Vel Error', color='cyan', linewidth=1); axes_ss_err[i, 1].axhline(0, color='gray', linestyle='--', linewidth=0.8); axes_ss_err[i, 1].set_ylabel('Error (rad/s)'); axes_ss_err[i, 1].legend(); axes_ss_err[i, 1].grid(True)
        axes_ss_err[-1, 0].set_xlabel('Time (s)'); axes_ss_err[-1, 1].set_xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.savefig(single_step_error_plot_save_path, dpi=300); plt.close(fig_ss_err); print(f"Single-step error plot saved to {single_step_error_plot_save_path}")
    else: print("Skipping single-step error plotting due to missing data.")

    print("Script finished.")