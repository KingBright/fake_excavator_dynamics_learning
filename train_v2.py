# -*- coding: utf-8 -*-
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
import math
import random
import copy
import time
from tqdm import tqdm

# ==============================================================================
# LSTM Model Definition (保持不变)
# ==============================================================================
class ExcavatorLSTM(nn.Module):
    # Input size and output size will be changed when creating the instance
    def __init__(self, input_size=13, hidden_size=256, output_size=9, num_layers=3, dropout=0.1): # Default sizes updated
        super(ExcavatorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_c=None):
        if not isinstance(x, torch.FloatTensor):
             x = x.float().to(next(self.parameters()).device) # Ensure correct device
        out, (hn, cn) = self.lstm(x, h_c)
        out = self.fc(out[:, -1, :])
        return out, (hn, cn)

# ==============================================================================
# 自定义损失函数 (不再需要特殊处理周期性，使用标准MSE)
# ==============================================================================
# We can now use standard MSE loss, as the periodicity is handled by encoding.
# No custom loss class needed for this specific problem anymore,
# although one *could* be made to weight components differently.
# We will use nn.MSELoss directly in the training loop.

# ==============================================================================
# Helper Functions (create_sequences, angle_difference 保持不变)
# ==============================================================================
def create_sequences(input_features, output_labels, seq_length):
    # ... (代码同上一版本) ...
    xs, ys = [], []
    for i in range(len(input_features) - seq_length):
        x = input_features[i:(i + seq_length)]
        y = output_labels[i + seq_length] # Label is state AFTER sequence
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.empty((0, seq_length, input_features.shape[1])), np.empty((0, output_labels.shape[1]))
    return np.array(xs), np.array(ys)

def angle_difference(angle1, angle2):
    # ... (代码同上一版本) ...
    diff = angle1 - angle2
    return (diff + np.pi) % (2 * np.pi) - np.pi

# Removed wrap_angle_to_pi as we use raw angles before cos/sin transform

# ==============================================================================
# Main Training and Evaluation
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Excavator Dynamics LSTM (Cos/Sin Angles, Scaled Vel)') # Updated description
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .npz data file')
    parser.add_argument('--output_prefix', type=str, default='excavator_lstm_cossin_qvelscaled', help='Prefix for saving files') # Updated prefix
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
    # --periodic_angle_idx is no longer directly used by loss, but useful for identifying which qpos index is Cab
    parser.add_argument('--cab_angle_idx', type=int, default=0, help='Index of the Cab angle within the 4 controlled qpos (0-based)')

    args = parser.parse_args()

    # --- 设置随机种子, 创建结果目录 (保持不变) ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    SAVE_DIR = f"h{args.hidden_size}_l{args.num_layers}_s{args.seq_length}_lr{args.learning_rate}_wd{args.weight_decay:.0e}_bs{args.batch_size}_cossin"
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR); print(f"Created results directory: {SAVE_DIR}")
    else: print(f"Results directory already exists: {SAVE_DIR}")
    scaler_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_scaler_input.joblib")
    scaler_qvel_out_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_scaler_qvel_out.joblib")
    model_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_model_best.pth")
    loss_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_losses.png")
    rollout_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_rollout_vs_single_step.png")
    error_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_rollout_error.png")
    single_step_error_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_single_step_error.png")


    # --- 1. 加载数据 (保持不变) ---
    print(f"Loading data from {args.data_path}...")
    # ... (代码同上一版本) ...
    try:
        data = np.load(args.data_path); time_data = data['time']; qpos_all = data['qpos']; qvel_all = data['qvel']; control_signal = data['control_signal']
        print(f"Data loaded. Total steps: {len(time_data)}")
    except Exception as e: print(f"Error loading data: {e}"); exit()


    # --- 2. 特征选择与工程 (修改: 应用 Cos/Sin 编码) ---
    qpos_indices = args.qpos_indices; qvel_indices = args.qvel_indices
    cab_idx = args.cab_angle_idx # Which of the 4 qpos is the Cab?
    other_qpos_indices_rel = [i for i in range(4) if i != cab_idx]

    print(f"Using qpos indices: {qpos_indices}"); print(f"Using qvel indices: {qvel_indices}")
    print(f"Cab angle index relative to controlled qpos: {cab_idx}")
    try:
        qpos = qpos_all[:, qpos_indices].copy(); qvel = qvel_all[:, qvel_indices]
    except IndexError as e: print(f"Error selecting indices: {e}. Check indices."); exit()

    # No need to wrap Cab angle here, cos/sin handles periodicity naturally
    # qpos[:, cab_idx] = wrap_angle_to_pi(qpos[:, cab_idx]) # Removed wrapping

    num_steps = min(len(qpos), len(qvel), len(control_signal))

    # Prepare features (t=0 to T-2)
    control_feat = control_signal[:num_steps-1]
    qvel_feat = qvel[:num_steps-1]
    qpos_feat_all = qpos[:num_steps-1]
    # Cab angle features
    cab_angle_raw = qpos_feat_all[:, cab_idx]
    cab_cos_feat = np.cos(cab_angle_raw)
    cab_sin_feat = np.sin(cab_angle_raw)
    # Other qpos features (remain in radians)
    other_qpos_feat = qpos_feat_all[:, other_qpos_indices_rel]
    # Combine input features: ctrl, cos(cab), sin(cab), other_qpos, qvel
    input_features = np.concatenate(
        (control_feat, cab_cos_feat[:, None], cab_sin_feat[:, None], other_qpos_feat, qvel_feat), axis=1
    )
    input_size = input_features.shape[1]; print(f"Input feature size: {input_size}") # Should be 13

    # Prepare labels (t=1 to T-1)
    qpos_labels_all = qpos[1:num_steps]
    qvel_labels_all = qvel[1:num_steps]
    # Cab angle labels
    cab_angle_labels_raw = qpos_labels_all[:, cab_idx]
    cab_cos_labels = np.cos(cab_angle_labels_raw)
    cab_sin_labels = np.sin(cab_angle_labels_raw)
    # Other qpos labels (remain in radians)
    other_qpos_labels = qpos_labels_all[:, other_qpos_indices_rel]
    # Combine output labels: cos(cab), sin(cab), other_qpos, qvel
    output_labels = np.concatenate(
        (cab_cos_labels[:, None], cab_sin_labels[:, None], other_qpos_labels, qvel_labels_all), axis=1
    )
    output_size = output_labels.shape[1]; print(f"Output label size: {output_size}") # Should be 9
    time_data = time_data[1:num_steps] # Align time

    print(f"Input features shape: {input_features.shape}")
    print(f"Output labels shape: {output_labels.shape}")
    print(f"Time data shape: {time_data.shape}")


    # --- 3. 数据划分 (Train, Validation, Test) (保持不变) ---
    # ... (代码同上一版本) ...
    print("Splitting data into Train, Validation, Test sets...")
    X_train_val, X_test, y_train_val, y_test, time_train_val, time_test = train_test_split(
        input_features, output_labels, time_data,
        test_size=args.test_split, random_state=args.seed, shuffle=False
    )
    val_size_in_train_val = int(len(X_train_val) * (args.val_split / (1.0 - args.test_split)))
    X_train = X_train_val[:-val_size_in_train_val]
    X_val = X_train_val[-val_size_in_train_val:]
    y_train = y_train_val[:-val_size_in_train_val]
    y_val = y_train_val[-val_size_in_train_val:]
    time_train = time_train_val[:-val_size_in_train_val]
    time_val = time_train_val[-val_size_in_train_val:]
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")


    # --- 4. Preprocessing (Scale Inputs, Scale Output Velocities) ---
    print("Scaling input features (fit on train)...")
    scaler_input = StandardScaler(); X_train_scaled = scaler_input.fit_transform(X_train)
    X_val_scaled = scaler_input.transform(X_val)
    X_test_scaled = scaler_input.transform(X_test)
    joblib.dump(scaler_input, scaler_filename); print(f"Input feature scaler saved to {scaler_filename}")

    print("Scaling ONLY output velocities (qvel part of labels, fit on train)...")
    # Labels: [cos, sin, qp1, qp2, qp3, qv0, qv1, qv2, qv3] (Indices 5 to 8 are qvel)
    qvel_output_indices = list(range(output_size - 4, output_size)) # Indices of velocities in the 9-dim output

    y_train_non_vel = y_train[:, :qvel_output_indices[0]] # cos, sin, other_qpos
    y_train_vel = y_train[:, qvel_output_indices]
    y_val_non_vel = y_val[:, :qvel_output_indices[0]]
    y_val_vel = y_val[:, qvel_output_indices]
    y_test_non_vel = y_test[:, :qvel_output_indices[0]]
    y_test_vel = y_test[:, qvel_output_indices] # Original velocities stored

    scaler_qvel_out = StandardScaler()
    y_train_vel_scaled = scaler_qvel_out.fit_transform(y_train_vel)
    y_val_vel_scaled = scaler_qvel_out.transform(y_val_vel)
    y_test_vel_scaled = scaler_qvel_out.transform(y_test_vel) # Scaled velocities for labels
    joblib.dump(scaler_qvel_out, scaler_qvel_out_filename)
    print(f"Output qvel scaler saved to {scaler_qvel_out_filename}")

    # Combine processed labels: non-vel part (original) + qvel part (scaled)
    y_train_processed = np.hstack((y_train_non_vel, y_train_vel_scaled))
    y_val_processed = np.hstack((y_val_non_vel, y_val_vel_scaled))
    y_test_processed = np.hstack((y_test_non_vel, y_test_vel_scaled)) # Labels for training/val/test loaders

    # Store original (unprocessed) test labels for final evaluation/plotting
    y_test_original = y_test.copy()


    # --- 5. Sequence Creation (保持不变) ---
    # ... (代码同上一版本, 使用正确的 create_sequences) ...
    print(f"Creating sequences with length {args.seq_length}...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled[:-1], y_train_processed, args.seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled[:-1], y_val_processed, args.seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled[:-1], y_test_processed, args.seq_length)
    time_train_seq_end = time_train[args.seq_length:]
    time_val_seq_end = time_val[args.seq_length:]
    time_test_seq_end = time_test[args.seq_length:]
    train_seq_len = min(len(X_train_seq), len(time_train_seq_end))
    val_seq_len = min(len(X_val_seq), len(time_val_seq_end))
    test_seq_len = min(len(X_test_seq), len(time_test_seq_end))
    X_train_seq, y_train_seq = X_train_seq[:train_seq_len], y_train_seq[:train_seq_len]
    X_val_seq, y_val_seq = X_val_seq[:val_seq_len], y_val_seq[:val_seq_len]
    X_test_seq, y_test_seq = X_test_seq[:test_seq_len], y_test_seq[:test_seq_len]
    time_train_seq_end = time_train_seq_end[:train_seq_len]
    time_val_seq_end = time_val_seq_end[:val_seq_len]
    time_test_seq_end = time_test_seq_end[:test_seq_len]
    print(f"Train sequences shape: X={X_train_seq.shape}, y={y_train_seq.shape}, t={time_train_seq_end.shape}")
    print(f"Validation sequences shape: X={X_val_seq.shape}, y={y_val_seq.shape}, t={time_val_seq_end.shape}")
    print(f"Test sequences shape: X={X_test_seq.shape}, y={y_test_seq.shape}, t={time_test_seq_end.shape}")


    # --- 6. Prepare PyTorch DataLoaders (保持不变) ---
    # ... (代码同上一版本) ...
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
       print("Error: Not enough data to create sequences for train, validation, or test set after alignment.")
       exit()
    train_dataset = TensorDataset(torch.FloatTensor(X_train_seq).to(device), torch.FloatTensor(y_train_seq).to(device))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_seq).to(device), torch.FloatTensor(y_val_seq).to(device))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_seq).to(device), torch.FloatTensor(y_test_seq).to(device))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # --- 7. Model, Loss, Optimizer (修改: 使用标准 MSE Loss) ---
    model = ExcavatorLSTM(input_size, args.hidden_size, output_size, args.num_layers, args.dropout).to(device)
    # Use standard MSE Loss for all outputs (cos, sin, other_qpos, qvel_scaled)
    criterion = nn.MSELoss()
    print("Using standard nn.MSELoss.")
    # Optimizer remains the same
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Using Adam optimizer with lr={args.learning_rate} and weight_decay={args.weight_decay}")
    print("Model Summary:"); # print(model);
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"Total trainable parameters: {num_params}")


    # --- 8. Training Loop with Validation (修改: 使用新 criterion) ---
    print("Starting training...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    total_start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_train_loss = 0
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Train", leave=False)
        for batch_x, batch_y_processed in progress_bar_train: # batch_y contains processed labels
            optimizer.zero_grad()
            # Ensure model and data are on the same device
            # batch_x = batch_x.to(device)
            # batch_y_processed = batch_y_processed.to(device)
            outputs_processed, _ = model(batch_x) # Model outputs processed predictions
            loss = criterion(outputs_processed, batch_y_processed) # Use standard MSE
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            progress_bar_train.set_postfix(loss=f"{loss.item():.6f}")
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Val", leave=False)
            for batch_x_val, batch_y_val_processed in progress_bar_val:
                # batch_x_val = batch_x_val.to(device)
                # batch_y_val_processed = batch_y_val_processed.to(device)
                outputs_val_processed, _ = model(batch_x_val)
                loss_val = criterion(outputs_val_processed, batch_y_val_processed) # Use standard MSE
                epoch_val_loss += loss_val.item()
                progress_bar_val.set_postfix(loss=f"{loss_val.item():.6f}")
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        epoch_end_time = time.time(); epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {epoch_duration:.2f}s')

        # Checkpoint and Early Stopping (no change here)
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.6f} --> {avg_val_loss:.6f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_filename)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    total_end_time = time.time(); total_training_time = total_end_time - total_start_time
    print(f"Training finished. Total time: {total_training_time:.2f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")


    # --- 9. Load Best Model (保持不变) ---
    if os.path.exists(model_filename):
        print(f"Loading best model from {model_filename} for final evaluation...")
        model.load_state_dict(torch.load(model_filename, map_location=device))
    else:
        print("Warning: No best model file found. Evaluating model from last epoch.")


    # --- 10. Final Evaluation on Test Set (Single Step) (修改: 使用新 criterion) ---
    print("Evaluating model on Test Set (Single Step)...")
    model.eval()
    test_loss_final = 0
    all_test_preds_processed = [] # Store processed predictions (cos, sin, ..., qvel_scaled)
    # all_test_targets_processed = [] # We already have y_test_seq which are the targets
    with torch.no_grad():
        for batch_x_test, batch_y_test_processed in tqdm(test_loader, desc="Testing (Single Step)", leave=False):
            # batch_x_test = batch_x_test.to(device)
            # batch_y_test_processed = batch_y_test_processed.to(device)
            outputs_test_processed, _ = model(batch_x_test)
            loss_test = criterion(outputs_test_processed, batch_y_test_processed) # Use standard MSE
            test_loss_final += loss_test.item()
            all_test_preds_processed.append(outputs_test_processed.cpu().numpy())
            # all_test_targets_processed.append(batch_y_test_processed.cpu().numpy())

    avg_test_loss_final = test_loss_final / len(test_loader)
    print(f"Average Test Set Loss (Single Step, MSE): {avg_test_loss_final:.6f}")

    # Concatenate test predictions
    all_test_preds_processed = np.concatenate(all_test_preds_processed, axis=0) # Predictions matching y_test_seq


    # --- 11. Inverse Transform Single Step Predictions & Recover Angles ---
    print("Recovering angles and inverse transforming velocities for single-step predictions...")
    # Predictions: [pred_cos, pred_sin, pred_qp1, pred_qp2, pred_qp3, pred_qv0_s, ...]
    pred_cab_cos = all_test_preds_processed[:, 0]
    pred_cab_sin = all_test_preds_processed[:, 1]
    pred_other_qpos_orig = all_test_preds_processed[:, 2:5] # Indices 2, 3, 4
    pred_qvel_scaled = all_test_preds_processed[:, 5:] # Indices 5 to 8

    # Recover Cab angle from prediction
    single_step_pred_cab_angle_orig = np.arctan2(pred_cab_sin, pred_cab_cos)
    # Inverse transform velocities
    single_step_pred_qvel_orig = scaler_qvel_out.inverse_transform(pred_qvel_scaled)

    # Combine into a standard qpos/qvel format for analysis (relative order matters!)
    single_step_pred_qpos_orig = np.zeros((len(all_test_preds_processed), 4))
    single_step_pred_qpos_orig[:, cab_idx] = single_step_pred_cab_angle_orig
    single_step_pred_qpos_orig[:, other_qpos_indices_rel] = pred_other_qpos_orig

    # Get corresponding Ground Truth in original qpos/qvel scale
    # We need the original output_labels corresponding to the test sequences
    gt_start_idx_original = len(X_train) + len(X_val) + args.seq_length
    gt_end_idx_original = gt_start_idx_original + len(y_test_seq)
    # Select the original qpos/qvel from the unfiltered `output_labels` (which already has cos/sin removed)
    # Need to reconstruct original qpos/qvel from the initial data loading stage
    original_output_qpos = qpos[1:num_steps] # Original qpos labels
    original_output_qvel = qvel[1:num_steps] # Original qvel labels

    ground_truth_qpos_orig_for_test_seq = original_output_qpos[gt_start_idx_original:gt_end_idx_original, :]
    ground_truth_qvel_orig_for_test_seq = original_output_qvel[gt_start_idx_original:gt_end_idx_original, :]


    # --- 12. Multi-step Rollout Prediction (修改: 使用 cos/sin) ---
    print("Performing multi-step rollout on a test segment...")
    start_index_in_test_seq = 0
    if len(X_test_seq) <= start_index_in_test_seq:
        print("Not enough test sequences for rollout evaluation.")
        predictions_rollout_pos = np.array([])
        predictions_rollout_vel = np.array([])
        ground_truth_rollout_pos = np.array([])
        ground_truth_rollout_vel = np.array([])
        rollout_steps = 0
    else:
        rollout_steps = len(X_test_seq) - start_index_in_test_seq
        current_sequence_rollout = torch.FloatTensor(X_test_seq[start_index_in_test_seq:start_index_in_test_seq+1]).to(device)
        # Store predictions in ORIGINAL qpos/qvel format
        predictions_rollout_pos_list = []
        predictions_rollout_vel_list = []

        # Initial state for rollout comparison (first GT label corresponding to the sequences)
        initial_rollout_qpos_true = ground_truth_qpos_orig_for_test_seq[0]
        initial_rollout_qvel_true = ground_truth_qvel_orig_for_test_seq[0]
        predictions_rollout_pos_list.append(initial_rollout_qpos_true)
        predictions_rollout_vel_list.append(initial_rollout_qvel_true)

        current_h_c_rollout = None
        # Keep track of the last predicted state in the format needed for input reconstruction
        last_predicted_qpos = initial_rollout_qpos_true.copy()
        last_predicted_qvel = initial_rollout_qvel_true.copy()

        with torch.no_grad():
            for i in range(rollout_steps - 1): # Predict rollout_steps-1 steps
                # Model predicts: [cos, sin, qp1, qp2, qp3, qv0_s, ...]
                predicted_output_processed_tensor, current_h_c_rollout = model(current_sequence_rollout, current_h_c_rollout)
                predicted_output_processed = predicted_output_processed_tensor.cpu().numpy().flatten()

                # --- Deconstruct prediction, recover angle, inverse transform velocity ---
                pred_cab_cos_rollout = predicted_output_processed[0]
                pred_cab_sin_rollout = predicted_output_processed[1]
                pred_other_qpos_orig_rollout = predicted_output_processed[2:5]
                pred_qvel_scaled_rollout = predicted_output_processed[5:]

                next_qpos_pred_orig = np.zeros(4)
                next_qpos_pred_orig[cab_idx] = np.arctan2(pred_cab_sin_rollout, pred_cab_cos_rollout)
                next_qpos_pred_orig[other_qpos_indices_rel] = pred_other_qpos_orig_rollout

                next_qvel_pred_orig = scaler_qvel_out.inverse_transform(pred_qvel_scaled_rollout.reshape(1, -1)).flatten()
                # --- End deconstruction ---

                # Store the recovered qpos/qvel
                predictions_rollout_pos_list.append(next_qpos_pred_orig)
                predictions_rollout_vel_list.append(next_qvel_pred_orig)

                # Update last predicted state for next input
                last_predicted_qpos = next_qpos_pred_orig
                last_predicted_qvel = next_qvel_pred_orig

                # --- Construct the next input frame (using cos/sin for Cab) ---
                next_control_frame_idx_in_X_test = start_index_in_test_seq + args.seq_length + i
                if next_control_frame_idx_in_X_test >= len(X_test):
                   print(f"Rollout stopped early at step {i+1}: Reached end of available control signals in X_test.")
                   rollout_steps = i + 2
                   break

                next_control_signal = X_test[next_control_frame_idx_in_X_test, :4] # Get from original unscaled test features split

                # Use predicted state, but convert Cab angle back to cos/sin for input
                next_cab_angle = last_predicted_qpos[cab_idx]
                next_cab_cos = np.cos(next_cab_angle)
                next_cab_sin = np.sin(next_cab_angle)
                next_other_qpos = last_predicted_qpos[other_qpos_indices_rel]
                next_qvel = last_predicted_qvel

                # Combine into next input frame (unscaled, 13 dim)
                next_input_frame_unscaled = np.concatenate(
                    (next_control_signal, [next_cab_cos], [next_cab_sin], next_other_qpos, next_qvel)
                )

                # Scale the frame using the input scaler
                next_input_frame_scaled = scaler_input.transform(next_input_frame_unscaled.reshape(1, -1)).flatten()
                # --- End constructing next input frame ---

                # Update input sequence
                next_sequence_np = np.vstack((current_sequence_rollout.cpu().numpy()[0, 1:, :], next_input_frame_scaled))
                current_sequence_rollout = torch.FloatTensor(next_sequence_np).unsqueeze(0).to(device)

        predictions_rollout_pos = np.array(predictions_rollout_pos_list)
        predictions_rollout_vel = np.array(predictions_rollout_vel_list)

        # Ground Truth for the exact rollout period
        ground_truth_rollout_pos = ground_truth_qpos_orig_for_test_seq[:rollout_steps]
        ground_truth_rollout_vel = ground_truth_qvel_orig_for_test_seq[:rollout_steps]


    # --- 13. Get Single Step Predictions for the Rollout Segment (修改: 使用恢复后的角度) ---
    if rollout_steps > 0:
       # Use the recovered single_step_pred_qpos_orig
       single_step_pred_qpos_segment = single_step_pred_qpos_orig[start_index_in_test_seq : start_index_in_test_seq + rollout_steps]
       single_step_pred_vel_segment = single_step_pred_qvel_orig[start_index_in_test_seq : start_index_in_test_seq + rollout_steps]
    else:
       single_step_pred_qpos_segment = np.array([])
       single_step_pred_vel_segment = np.array([])


    # --- 14. Plotting Comparison (修改: Cab位姿使用恢复的角度) ---
    print("Plotting results...")
    # Plot Train/Validation Loss (保持不变)
    # ... (代码同上一版本) ...
    plt.figure(figsize=(10, 5)); plt.plot(train_losses, label='Training Loss'); plt.plot(val_losses, label='Validation Loss'); best_epoch_idx = np.argmin(val_losses); plt.axvline(x=best_epoch_idx, color='r', linestyle='--', label=f'Best Epoch ({best_epoch_idx+1})'); plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.ylim(bottom=0); plt.legend(); plt.grid(True); plt.savefig(loss_plot_save_path); plt.close(); print(f"Loss plot saved to {loss_plot_save_path}")


    # Plot Rollout vs Single Step vs Ground Truth
    if rollout_steps > 0:
        num_joints_to_plot = 4
        fig, axes = plt.subplots(num_joints_to_plot, 2, figsize=(16, 3 * num_joints_to_plot), sharex=True)
        if num_joints_to_plot == 1: axes = np.array([[axes[0], axes[1]]])

        time_axis = time_test_seq_end[start_index_in_test_seq : start_index_in_test_seq + rollout_steps]
        time_axis = time_axis - time_axis[0]

        joint_names = ['Cab', 'Boom', 'Arm', 'Bucket'] # Assuming this order matches qpos_indices [0, 1, 2, 3]

        for i in range(num_joints_to_plot):
            # Qpos plot - Use original angles for GT, single-step, rollout
            axes[i, 0].plot(time_axis, np.rad2deg(ground_truth_rollout_pos[:, i]), label='Ground Truth', color='blue', linewidth=1.5)
            axes[i, 0].plot(time_axis, np.rad2deg(predictions_rollout_pos[:, i]), label='Rollout Pred', color='red', linestyle='--', linewidth=1)
            axes[i, 0].plot(time_axis, np.rad2deg(single_step_pred_qpos_segment[:, i]), label='Single-Step Pred', color='green', linestyle=':', linewidth=1)
            axes[i, 0].set_ylabel(f'{joint_names[i]} Pos (deg)'); axes[i, 0].legend(); axes[i, 0].grid(True)
            # Qvel plot - Use original velocities for GT, single-step, rollout
            axes[i, 1].plot(time_axis, ground_truth_rollout_vel[:, i], label='Ground Truth', color='blue', linewidth=1.5)
            axes[i, 1].plot(time_axis, predictions_rollout_vel[:, i], label='Rollout Pred', color='red', linestyle='--', linewidth=1)
            axes[i, 1].plot(time_axis, single_step_pred_vel_segment[:, i], label='Single-Step Pred', color='green', linestyle=':', linewidth=1)
            axes[i, 1].set_ylabel(f'{joint_names[i]} Vel (rad/s)'); axes[i, 1].legend(); axes[i, 1].grid(True)

        axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')
        fig.suptitle('Single-Step vs Multi-step Rollout Prediction vs Ground Truth (Cos/Sin Encoded Cab)') # Updated title
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(rollout_plot_save_path); print(f"Comparison plot saved to {rollout_plot_save_path}"); plt.close(fig)
    else:
        print("Skipping rollout plot as no rollout steps were performed.")


    # --- 15. Calculate & Print Errors (修改: 使用恢复后的角度计算误差) ---
    if rollout_steps > 0:
        # Rollout Errors (Use recovered angles)
        rollout_qpos_error = np.zeros_like(predictions_rollout_pos)
        # Handle Cab angle difference properly
        rollout_qpos_error[:, cab_idx] = angle_difference(predictions_rollout_pos[:, cab_idx], ground_truth_rollout_pos[:, cab_idx])
        # Handle other angles
        rollout_qpos_error[:, other_qpos_indices_rel] = predictions_rollout_pos[:, other_qpos_indices_rel] - ground_truth_rollout_pos[:, other_qpos_indices_rel]
        rollout_qvel_error = predictions_rollout_vel - ground_truth_rollout_vel # Velocity error is standard subtraction
        # Calculate metrics...
        rollout_qpos_mae = np.mean(np.abs(rollout_qpos_error)); rollout_qvel_mae = np.mean(np.abs(rollout_qvel_error)); rollout_qpos_rmse = np.sqrt(np.mean(rollout_qpos_error**2)); rollout_qvel_rmse = np.sqrt(np.mean(rollout_qvel_error**2))
        print("\n--- Rollout Prediction Errors (Test Segment) ---"); print(f"Qpos MAE: {rollout_qpos_mae:.6f} rad ({np.rad2deg(rollout_qpos_mae):.4f} deg)"); print(f"Qvel MAE: {rollout_qvel_mae:.6f} rad/s"); print(f"Qpos RMSE: {rollout_qpos_rmse:.6f} rad ({np.rad2deg(rollout_qpos_rmse):.4f} deg)"); print(f"Qvel RMSE: {rollout_qvel_rmse:.6f} rad/s")


        # Single Step Errors (Use recovered angles)
        single_step_qpos_error = np.zeros_like(single_step_pred_qpos_segment)
        single_step_qpos_error[:, cab_idx] = angle_difference(single_step_pred_qpos_segment[:, cab_idx], ground_truth_rollout_pos[:, cab_idx])
        single_step_qpos_error[:, other_qpos_indices_rel] = single_step_pred_qpos_segment[:, other_qpos_indices_rel] - ground_truth_rollout_pos[:, other_qpos_indices_rel]
        single_step_qvel_error = single_step_pred_vel_segment - ground_truth_rollout_vel
        # Calculate metrics...
        single_step_qpos_mae = np.mean(np.abs(single_step_qpos_error)); single_step_qvel_mae = np.mean(np.abs(single_step_qvel_error)); single_step_qpos_rmse = np.sqrt(np.mean(single_step_qpos_error**2)); single_step_qvel_rmse = np.sqrt(np.mean(single_step_qvel_error**2))
        print("\n--- Single-Step Prediction Errors (Test Segment) ---"); print(f"Qpos MAE: {single_step_qpos_mae:.6f} rad ({np.rad2deg(single_step_qpos_mae):.4f} deg)"); print(f"Qvel MAE: {single_step_qvel_mae:.6f} rad/s"); print(f"Qpos RMSE: {single_step_qpos_rmse:.6f} rad ({np.rad2deg(single_step_qpos_rmse):.4f} deg)"); print(f"Qvel RMSE: {single_step_qvel_rmse:.6f} rad/s")


        # --- Error Plot (Rollout) (修改: 使用恢复后的角度误差) ---
        fig_err, axes_err = plt.subplots(num_joints_to_plot, 2, figsize=(16, 3 * num_joints_to_plot), sharex=True)
        # ... (Plotting code uses rollout_qpos_error which is calculated correctly now) ...
        if num_joints_to_plot == 1: axes_err = np.array([[axes_err[0], axes_err[1]]])
        fig_err.suptitle('Rollout Prediction Error (Prediction - Ground Truth)')
        for i in range(num_joints_to_plot):
             axes_err[i, 0].plot(time_axis, np.rad2deg(rollout_qpos_error[:, i]), label=f'{joint_names[i]} Pos Error', color='green'); axes_err[i, 0].axhline(0, color='gray', linestyle='--', linewidth=0.8); axes_err[i, 0].set_ylabel('Error (deg)'); axes_err[i, 0].legend(); axes_err[i, 0].grid(True)
             axes_err[i, 1].plot(time_axis, rollout_qvel_error[:, i], label=f'{joint_names[i]} Vel Error', color='purple'); axes_err[i, 1].axhline(0, color='gray', linestyle='--', linewidth=0.8); axes_err[i, 1].set_ylabel('Error (rad/s)'); axes_err[i, 1].legend(); axes_err[i, 1].grid(True)
        axes_err[-1, 0].set_xlabel('Time (s)'); axes_err[-1, 1].set_xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(error_plot_save_path); print(f"Rollout error plot saved to {error_plot_save_path}"); plt.close(fig_err)


    print("Script finished.")