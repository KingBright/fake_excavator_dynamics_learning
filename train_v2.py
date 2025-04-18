# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import joblib # To save/load scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Import MinMaxScaler just in case, though not used in recommended path
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
    def __init__(self, input_size=12, hidden_size=256, output_size=8, num_layers=3, dropout=0.1):
        super(ExcavatorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_c=None):
        out, (hn, cn) = self.lstm(x, h_c)
        out = self.fc(out[:, -1, :])
        return out, (hn, cn)

# ==============================================================================
# 自定义损失函数 (处理角度周期性)
# ==============================================================================
class PeriodicAngleMSELoss(nn.Module):
    # --- 修改: 移除 qpos_scaler, 因为 qpos 不再归一化 ---
    def __init__(self, angle_indices=[0], non_angle_indices=[1, 2, 3],
                 # qpos_scaler=None, # Removed
                 weight_angle=1.0, weight_other=1.0):
        """
        自定义损失函数，特殊处理周期性角度的误差。
        假设 qpos 输入是原始弧度值，qvel 输入是归一化值。
        """
        super().__init__()
        self.angle_indices = angle_indices
        self.non_angle_indices = non_angle_indices
        # self.qpos_scaler = qpos_scaler # Removed
        self.weight_angle = weight_angle
        self.weight_other = weight_other
        self.mse_loss = nn.MSELoss(reduction='none') # Calculate element-wise squared error

    def forward(self, y_pred, y_true):
        """
        计算损失。
        Args:
            y_pred (Tensor): 预测值, 形状 (batch, 8)。qpos部分是原始弧度, qvel部分是归一化值。
            y_true (Tensor): 真实值, 形状 (batch, 8)。qpos部分是原始弧度, qvel部分是归一化值。
        """
        qpos_pred_orig = y_pred[:, :4] # Predictions are already in original radians
        qvel_pred_scaled = y_pred[:, 4:] # Velocity predictions are scaled
        qpos_true_orig = y_true[:, :4] # True positions are in original radians
        qvel_true_scaled = y_true[:, 4:] # True velocities are scaled

        # --- 计算速度部分的 MSE Loss (在归一化空间计算) ---
        loss_qvel = self.mse_loss(qvel_pred_scaled, qvel_true_scaled).mean()

        # --- 计算非周期性角度的 MSE Loss (在原始空间计算) ---
        if self.non_angle_indices:
            loss_qpos_non_periodic = self.mse_loss(qpos_pred_orig[:, self.non_angle_indices],
                                                   qpos_true_orig[:, self.non_angle_indices]).mean()
        else:
            loss_qpos_non_periodic = torch.tensor(0.0, device=y_pred.device)

        # --- 计算周期性角度的 Loss (在原始空间计算) ---
        loss_qpos_periodic = torch.tensor(0.0, device=y_pred.device)
        if self.angle_indices:
            # qpos 已经是原始角度，直接计算最短角度差
            angle_diff = qpos_pred_orig[:, self.angle_indices] - qpos_true_orig[:, self.angle_indices]
            wrapped_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi
            # 计算平方误差并求平均
            loss_qpos_periodic = torch.mean(wrapped_diff**2)

        # 加权组合损失 (现在所有角度误差都在原始空间算，速度误差在归一化空间算)
        # 可以考虑后续对速度误差项进行反归一化后再加权，但目前先保持这样
        total_loss = (self.weight_angle * (loss_qpos_periodic + loss_qpos_non_periodic) +
                      self.weight_other * loss_qvel)

        return total_loss

# ==============================================================================
# Helper Functions (保持不变)
# ==============================================================================
def create_sequences(input_features, output_labels, seq_length):
    xs, ys = [], []
    for i in range(len(input_features) - seq_length):
        x = input_features[i:(i + seq_length)]
        y = output_labels[i + seq_length -1] # Get the label corresponding to the END of the sequence
        xs.append(x)
        ys.append(y)
    if not xs: # Handle case where data is shorter than seq_length
        return np.empty((0, seq_length, input_features.shape[1])), np.empty((0, output_labels.shape[1]))
    return np.array(xs), np.array(ys)

def angle_difference(angle1, angle2):
    diff = angle1 - angle2
    return (diff + np.pi) % (2 * np.pi) - np.pi

def wrap_angle_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# ==============================================================================
# Main Training and Evaluation
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Excavator Dynamics LSTM (Raw Angles, Scaled Vel)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .npz data file')
    parser.add_argument('--output_prefix', type=str, default='excavator_lstm_qvelscaled', help='Prefix for saving files') # Changed default prefix
    parser.add_argument('--seq_length', type=int, default=50, help='Input sequence length') # Changed default
    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty)')
    # --- 修改: 调整 test_split 为 val_split, 并增加 test_split ---
    parser.add_argument('--val_split', type=float, default=0.15, help='Fraction of data for validation')
    parser.add_argument('--test_split', type=float, default=0.15, help='Fraction of data for final testing')
    # --- 结束修改 ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--qpos_indices', type=int, nargs=4, default=[0, 1, 2, 3], help='Indices of controlled joint positions')
    parser.add_argument('--qvel_indices', type=int, nargs=4, default=[0, 1, 2, 3], help='Indices of controlled joint velocities')
    parser.add_argument('--periodic_angle_idx', type=int, default=0, help='Index of periodic angle within controlled qpos (0-based)')
    # --- 移除 --scale_output 参数 ---
    # parser.add_argument('--scale_output', action='store_true', help='Scale output features (qpos, qvel) separately') # Removed

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 创建结果目录 ---
    # Adjusted naming convention
    SAVE_DIR = f"h{args.hidden_size}_l{args.num_layers}_s{args.seq_length}_lr{args.learning_rate}_wd{args.weight_decay:.0e}_bs{args.batch_size}"
    # Removed scaledout flag
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR); print(f"Created results directory: {SAVE_DIR}")
    else: print(f"Results directory already exists: {SAVE_DIR}")

    # --- 更新文件保存路径 ---
    scaler_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_scaler_input.joblib")
    # scaler_qpos_out_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_scaler_qpos_out.joblib") # Removed qpos scaler
    scaler_qvel_out_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_scaler_qvel_out.joblib") # Keep qvel scaler
    model_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_model_best.pth") # Save best model
    loss_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_losses.png") # Combined loss plot
    rollout_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_rollout_vs_ground_truth.png") # Simplified name
    error_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_rollout_error.png")

    # --- 1. Load Data ---
    print(f"Loading data from {args.data_path}...")
    try:
        data = np.load(args.data_path); time_data = data['time']; qpos_all = data['qpos']; qvel_all = data['qvel']; control_signal = data['control_signal']
        print(f"Data loaded. Total steps: {len(time_data)}")
    except Exception as e: print(f"Error loading data: {e}"); exit()

    # --- 2. Feature Selection and Engineering ---
    qpos_indices = args.qpos_indices; qvel_indices = args.qvel_indices
    print(f"Using qpos indices: {qpos_indices}"); print(f"Using qvel indices: {qvel_indices}")
    try:
        qpos = qpos_all[:, qpos_indices].copy(); qvel = qvel_all[:, qvel_indices]
    except IndexError as e: print(f"Error selecting indices: {e}. Check indices."); exit()

    # Wrap 周期性角度到 [-pi, pi] (用于输入和输出标签)
    periodic_idx_in_controlled = args.periodic_angle_idx
    print(f"Wrapping angle at index {periodic_idx_in_controlled} to [-pi, pi]...")
    qpos[:, periodic_idx_in_controlled] = wrap_angle_to_pi(qpos[:, periodic_idx_in_controlled])

    # 输入特征: control (4), qpos (4, wrapped), qvel (4) -> 12 维
    num_steps = min(len(qpos), len(qvel), len(control_signal))
    input_features = np.hstack((control_signal[:num_steps], qpos[:num_steps], qvel[:num_steps]))

    # 输出标签: next qpos (4维, wrapped) + next qvel (4维) -> 8维
    output_qpos_labels = qpos[1:num_steps+1]
    output_qvel_labels = qvel[1:num_steps+1]
    output_labels = np.hstack((output_qpos_labels, output_qvel_labels))

    input_features = input_features[:len(output_labels)] # Ensure input and output lengths match
    time_data = time_data[:len(output_labels)] # Align time data as well

    print(f"Input features shape: {input_features.shape}"); print(f"Output labels shape: {output_labels.shape}")
    input_size = input_features.shape[1]; output_size = output_labels.shape[1]

    # --- 3. Data Splitting (Train, Validation, Test) ---
    print("Splitting data into Train, Validation, Test sets...")
    if args.val_split + args.test_split >= 1.0:
        print("Error: Sum of validation and test splits must be less than 1.")
        exit()

    # First split into Train+Val and Test
    X_train_val, X_test, y_train_val, y_test, time_train_val, time_test = train_test_split(
        input_features, output_labels, time_data,
        test_size=args.test_split, random_state=args.seed, shuffle=False # Keep sequential order
    )

    # Calculate split index for Train vs Val within the remaining data
    val_size_in_train_val = int(len(X_train_val) * (args.val_split / (1.0 - args.test_split)))

    # Split Train+Val into Train and Val
    X_train = X_train_val[:-val_size_in_train_val]
    X_val = X_train_val[-val_size_in_train_val:]
    y_train = y_train_val[:-val_size_in_train_val]
    y_val = y_train_val[-val_size_in_train_val:]
    time_train = time_train_val[:-val_size_in_train_val]
    time_val = time_train_val[-val_size_in_train_val:]


    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    # --- 4. Preprocessing (Scaling) ---
    # 标准化输入特征 (Fit only on Train data)
    print("Scaling input features (fit on train)...")
    scaler_input = StandardScaler(); X_train_scaled = scaler_input.fit_transform(X_train)
    X_val_scaled = scaler_input.transform(X_val) # Transform Val
    X_test_scaled = scaler_input.transform(X_test) # Transform Test
    joblib.dump(scaler_input, scaler_filename); print(f"Input feature scaler saved to {scaler_filename}")

    # --- 修改: 仅归一化输出标签中的速度部分 ---
    print("Scaling ONLY output velocities (qvel, fit on train)...")
    y_train_qpos = y_train[:, :4]; y_train_qvel = y_train[:, 4:]
    y_val_qpos = y_val[:, :4];     y_val_qvel = y_val[:, 4:]
    y_test_qpos = y_test[:, :4];    y_test_qvel = y_test[:, 4:]

    scaler_qvel_out = StandardScaler()
    y_train_qvel_scaled = scaler_qvel_out.fit_transform(y_train_qvel)
    y_val_qvel_scaled = scaler_qvel_out.transform(y_val_qvel) # Transform Val
    y_test_qvel_scaled = scaler_qvel_out.transform(y_test_qvel) # Transform Test
    joblib.dump(scaler_qvel_out, scaler_qvel_out_filename)
    print(f"Output qvel scaler saved to {scaler_qvel_out_filename}")

    # 组合处理后的标签: qpos (原始) + qvel (归一化)
    y_train_processed = np.hstack((y_train_qpos, y_train_qvel_scaled))
    y_val_processed = np.hstack((y_val_qpos, y_val_qvel_scaled))
    y_test_processed = np.hstack((y_test_qpos, y_test_qvel_scaled))
    # --- 结束修改 ---

    # --- 5. Sequence Creation ---
    print(f"Creating sequences with length {args.seq_length}...")
    # Use scaled inputs and processed outputs (qpos_orig, qvel_scaled)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_processed, args.seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_processed, args.seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_processed, args.seq_length)
    print(f"Train sequences shape: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"Validation sequences shape: X={X_val_seq.shape}, y={y_val_seq.shape}")
    print(f"Test sequences shape: X={X_test_seq.shape}, y={y_test_seq.shape}")

    # --- 6. Prepare PyTorch DataLoaders ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    print(f"Using device: {device}")

    # Check if sequence creation resulted in empty arrays
    if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
       print("Error: Not enough data to create sequences for train, validation, or test set.")
       exit()

    train_dataset = TensorDataset(torch.FloatTensor(X_train_seq).to(device), torch.FloatTensor(y_train_seq).to(device))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_seq).to(device), torch.FloatTensor(y_val_seq).to(device))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_seq).to(device), torch.FloatTensor(y_test_seq).to(device))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) # No shuffle for validation/test
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 7. Model, Loss, Optimizer ---
    model = ExcavatorLSTM(input_size, args.hidden_size, output_size, args.num_layers, args.dropout).to(device)
    periodic_idx_list = [args.periodic_angle_idx]; non_periodic_idx_list = [i for i in range(4) if i != args.periodic_angle_idx]
    # --- 修改: 移除传入损失函数的 qpos_scaler ---
    criterion = PeriodicAngleMSELoss(angle_indices=periodic_idx_list,
                                     non_angle_indices=non_periodic_idx_list)
                                     # qpos_scaler=None) # Removed
    print(f"Using PeriodicAngleMSELoss for angle index {periodic_idx_list}")
    print("Loss assumes qpos is in radians, qvel is scaled.")
    # --- 结束修改 ---
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Using Adam optimizer with lr={args.learning_rate} and weight_decay={args.weight_decay}")
    print("Model Summary:"); print(model); num_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"Total trainable parameters: {num_params}")

    # --- 8. Training Loop with Validation ---
    print("Starting training...")
    train_losses, val_losses = [], [] # Store losses
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10 # How many epochs to wait for improvement before stopping

    total_start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Train", leave=False)
        for batch_x, batch_y in progress_bar_train:
            optimizer.zero_grad()
            outputs, _ = model(batch_x) # batch_y contains qpos_orig, qvel_scaled
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            progress_bar_train.set_postfix(loss=f"{loss.item():.6f}")
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Val", leave=False)
            for batch_x_val, batch_y_val in progress_bar_val:
                outputs_val, _ = model(batch_x_val) # batch_y_val contains qpos_orig, qvel_scaled
                loss_val = criterion(outputs_val, batch_y_val)
                epoch_val_loss += loss_val.item()
                progress_bar_val.set_postfix(loss=f"{loss_val.item():.6f}")
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        epoch_end_time = time.time(); epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {epoch_duration:.2f}s')

        # --- Checkpoint and Early Stopping based on Validation Loss ---
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
        # --- End Checkpoint ---

    total_end_time = time.time(); total_training_time = total_end_time - total_start_time
    print(f"Training finished. Total time: {total_training_time:.2f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # --- 9. Load Best Model (Saved based on Validation Loss) ---
    if os.path.exists(model_filename):
        print(f"Loading best model from {model_filename} for final evaluation...")
        model.load_state_dict(torch.load(model_filename, map_location=device))
    else:
        print("Warning: No best model file found. Evaluating model from last epoch.")

    # --- 10. Final Evaluation on Test Set ---
    print("Evaluating model on Test Set...")
    model.eval()
    test_loss_final = 0
    all_test_preds_processed = [] # Store predictions (qpos_orig, qvel_scaled)
    all_test_targets_processed = [] # Store targets (qpos_orig, qvel_scaled)
    with torch.no_grad():
        for batch_x_test, batch_y_test in tqdm(test_loader, desc="Testing", leave=False):
            outputs_test, _ = model(batch_x_test) # batch_y_test contains qpos_orig, qvel_scaled
            loss_test = criterion(outputs_test, batch_y_test)
            test_loss_final += loss_test.item()
            all_test_preds_processed.append(outputs_test.cpu().numpy())
            all_test_targets_processed.append(batch_y_test.cpu().numpy())

    avg_test_loss_final = test_loss_final / len(test_loader)
    print(f"Average Test Set Loss (Final Model, Custom Loss): {avg_test_loss_final:.6f}")

    # Concatenate test predictions and targets
    all_test_preds_processed = np.concatenate(all_test_preds_processed, axis=0)
    all_test_targets_processed = np.concatenate(all_test_targets_processed, axis=0)

    # --- 11. Inverse Transform Test Predictions (Only Velocities) ---
    print("Inverse transforming predicted velocities for analysis...")
    pred_qpos_test_orig = all_test_preds_processed[:, :4]
    pred_qvel_test_scaled = all_test_preds_processed[:, 4:]
    true_qpos_test_orig = all_test_targets_processed[:, :4]
    # true_qvel_test_scaled = all_test_targets_processed[:, 4:] # We already have the scaler

    # Inverse transform predictions
    pred_qvel_test_orig = scaler_qvel_out.inverse_transform(pred_qvel_test_scaled)

    # Get original true velocities (from the initial split `y_test`)
    # Need to align `y_test` with the sequences used in `test_loader`
    # `y_test_seq` corresponds to labels for `X_test_seq`
    # We need the original `y_test` that corresponds to `y_test_seq`
    # Find the starting index of the test set in the original data
    test_start_index = len(X_train) + len(X_val)
    # The labels in y_test_seq start from index `test_start_index + seq_length - 1` in the original `output_labels`
    test_label_start_original_idx = test_start_index + args.seq_length - 1
    test_label_end_original_idx = test_label_start_original_idx + len(y_test_seq)

    true_qvel_test_orig_full = y_test[:, 4:] # Get original qvel from the initial split
    # Select the portion corresponding to the sequences evaluated
    true_qvel_test_orig = true_qvel_test_orig_full[args.seq_length-1:]

    # --- 12. Multi-step Rollout Prediction on Test Set ---
    print("Performing multi-step rollout on a test segment...")
    # Use the start of the *test* sequences for rollout
    start_index_in_test_seq = 0
    if len(X_test_seq) <= start_index_in_test_seq:
        print("Not enough test sequences for rollout evaluation.")
    else:
        rollout_steps = len(X_test_seq) - start_index_in_test_seq # Rollout over the available test sequences
        current_sequence_rollout = torch.FloatTensor(X_test_seq[start_index_in_test_seq:start_index_in_test_seq+1]).to(device)

        # Store predictions in ORIGINAL scale
        predictions_rollout_pos = []
        predictions_rollout_vel = []

        # Get the *initial state* for the rollout from the *original* test data
        # The first sequence X_test_seq[0] corresponds to input features from X_test[0] to X_test[seq_length-1]
        # The *first prediction* corresponds to the state at y_test[seq_length-1]
        # We need the state *before* the first prediction to start the rollout comparison fairly.
        # Let's use the *first actual ground truth label* from the test set as the initial state for comparison plots.
        initial_rollout_qpos_true = y_test_qpos[args.seq_length - 1]
        initial_rollout_qvel_true = y_test_qvel[args.seq_length - 1] # Original, unscaled

        predictions_rollout_pos.append(initial_rollout_qpos_true)
        predictions_rollout_vel.append(initial_rollout_qvel_true)

        current_h_c_rollout = None

        with torch.no_grad():
            for i in range(1, rollout_steps):
                # Model predicts: qpos_orig, qvel_scaled
                predicted_output_processed_tensor, current_h_c_rollout = model(current_sequence_rollout, current_h_c_rollout)
                predicted_output_processed = predicted_output_processed_tensor.cpu().numpy().flatten()

                # --- Deconstruct and inverse transform velocity ---
                next_qpos_pred_orig = predicted_output_processed[:4]
                next_qvel_pred_scaled = predicted_output_processed[4:].reshape(1, -1)
                next_qvel_pred_orig = scaler_qvel_out.inverse_transform(next_qvel_pred_scaled).flatten()
                # --- End deconstruction ---

                # Wrap predicted periodic angle (already in original scale)
                next_qpos_pred_orig[periodic_idx_list] = wrap_angle_to_pi(next_qpos_pred_orig[periodic_idx_list])

                predictions_rollout_pos.append(next_qpos_pred_orig)
                predictions_rollout_vel.append(next_qvel_pred_orig)

                # --- Construct the next input frame ---
                # Get the *actual* control signal for the *next* time step from the original test features X_test
                # The current sequence `current_sequence_rollout` ends at time step `start_index_in_test_seq + i - 1` within X_test_seq.
                # This corresponds to original features up to index `test_start_index + args.seq_length + i - 2`.
                # The *next* control signal needed is at original index `test_start_index + args.seq_length + i - 1`.
                next_control_original_idx = test_start_index + args.seq_length + i - 1
                if next_control_original_idx >= len(input_features): # Check bounds against original input_features
                   print(f"Rollout stopped early at step {i}: Reached end of available control signals.")
                   rollout_steps = i # Adjust rollout steps if we run out of controls
                   break

                # Use original feature array X_test (unscaled) to get control signal
                next_control_signal = X_test[args.seq_length + i - 1, :4] # Index relative to start of X_test

                # Use the predicted qpos (orig) and qvel (orig) from the previous step
                next_qpos_for_input = next_qpos_pred_orig
                next_qvel_for_input = next_qvel_pred_orig

                # Combine into next input frame (unscaled)
                next_input_frame_unscaled = np.concatenate((next_control_signal, next_qpos_for_input, next_qvel_for_input))

                # Scale the frame using the input scaler
                next_input_frame_scaled = scaler_input.transform(next_input_frame_unscaled.reshape(1, -1)).flatten()
                # --- End constructing next input frame ---

                # Update input sequence
                next_sequence_np = np.vstack((current_sequence_rollout.cpu().numpy()[0, 1:, :], next_input_frame_scaled))
                current_sequence_rollout = torch.FloatTensor(next_sequence_np).unsqueeze(0).to(device)

        predictions_rollout_pos = np.array(predictions_rollout_pos)
        predictions_rollout_vel = np.array(predictions_rollout_vel)

        # --- Ground Truth for the Rollout Segment (Original Scale) ---
        # Get the ground truth states corresponding to the rollout predictions
        # Rollout predictions start from the state *after* the first input sequence X_test_seq[0]
        gt_start_idx_in_y_test = args.seq_length - 1
        gt_end_idx_in_y_test = gt_start_idx_in_y_test + rollout_steps

        if gt_end_idx_in_y_test > len(y_test): # Adjust if rollout was stopped early
            gt_end_idx_in_y_test = len(y_test)
            # Also adjust prediction length if needed (should match due to loop break condition)
            predictions_rollout_pos = predictions_rollout_pos[:gt_end_idx_in_y_test - gt_start_idx_in_y_test]
            predictions_rollout_vel = predictions_rollout_vel[:gt_end_idx_in_y_test - gt_start_idx_in_y_test]


        ground_truth_rollout_pos = y_test[gt_start_idx_in_y_test : gt_end_idx_in_y_test, :4]
        ground_truth_rollout_vel = y_test[gt_start_idx_in_y_test : gt_end_idx_in_y_test, 4:] # These are original scale velocities

        # --- Plotting Comparison (Train/Val Loss, Rollout) ---
        print("Plotting results...")
        # Plot Train/Validation Loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.axvline(x=np.argmin(val_losses), color='r', linestyle='--', label=f'Best Epoch ({np.argmin(val_losses)+1})')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Custom Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(loss_plot_save_path); plt.close()
        print(f"Loss plot saved to {loss_plot_save_path}")

        # Plot Rollout Comparison
        num_joints_to_plot = 4
        fig, axes = plt.subplots(num_joints_to_plot, 2, figsize=(16, 3 * num_joints_to_plot), sharex=True)
        if num_joints_to_plot == 1: axes = np.array([[axes[0], axes[1]]])

        # Adjust time axis using time_test
        time_axis_start_idx = gt_start_idx_in_y_test # Index within time_test
        time_axis_end_idx = time_axis_start_idx + len(predictions_rollout_pos) # Should match gt length now
        time_axis = time_test[time_axis_start_idx:time_axis_end_idx] - time_test[time_axis_start_idx] # Start time axis at 0


        joint_names = ['Cab', 'Boom', 'Arm', 'Bucket']
        for i in range(num_joints_to_plot):
            # Qpos plot
            axes[i, 0].plot(time_axis, np.rad2deg(ground_truth_rollout_pos[:, i]), label='Ground Truth', color='blue', linewidth=1.5)
            axes[i, 0].plot(time_axis, np.rad2deg(predictions_rollout_pos[:, i]), label='Rollout Pred', color='red', linestyle='--', linewidth=1)
            axes[i, 0].set_ylabel(f'{joint_names[i]} Pos (deg)'); axes[i, 0].legend(); axes[i, 0].grid(True)
            # Qvel plot
            axes[i, 1].plot(time_axis, ground_truth_rollout_vel[:, i], label='Ground Truth', color='blue', linewidth=1.5)
            axes[i, 1].plot(time_axis, predictions_rollout_vel[:, i], label='Rollout Pred', color='red', linestyle='--', linewidth=1)
            axes[i, 1].set_ylabel(f'{joint_names[i]} Vel (rad/s)'); axes[i, 1].legend(); axes[i, 1].grid(True)

        axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')
        fig.suptitle('Multi-step Rollout Prediction vs Ground Truth (Test Set)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(rollout_plot_save_path); print(f"Rollout plot saved to {rollout_plot_save_path}"); plt.close(fig)

        # --- Calculate & Print Rollout Errors (Original Scale) ---
        rollout_qpos_error = np.zeros_like(predictions_rollout_pos)
        rollout_qpos_error[:, non_periodic_idx_list] = predictions_rollout_pos[:, non_periodic_idx_list] - ground_truth_rollout_pos[:, non_periodic_idx_list]
        rollout_qpos_error[:, periodic_idx_list] = angle_difference(predictions_rollout_pos[:, periodic_idx_list], ground_truth_rollout_pos[:, periodic_idx_list])
        rollout_qvel_error = predictions_rollout_vel - ground_truth_rollout_vel

        rollout_qpos_mae = np.mean(np.abs(rollout_qpos_error))
        rollout_qvel_mae = np.mean(np.abs(rollout_qvel_error))
        rollout_qpos_rmse = np.sqrt(np.mean(rollout_qpos_error**2))
        rollout_qvel_rmse = np.sqrt(np.mean(rollout_qvel_error**2))

        print("\n--- Rollout Prediction Errors (Test Set) ---")
        print(f"Qpos MAE: {rollout_qpos_mae:.6f} rad ({np.rad2deg(rollout_qpos_mae):.4f} deg)")
        print(f"Qvel MAE: {rollout_qvel_mae:.6f} rad/s")
        print(f"Qpos RMSE: {rollout_qpos_rmse:.6f} rad ({np.rad2deg(rollout_qpos_rmse):.4f} deg)")
        print(f"Qvel RMSE: {rollout_qvel_rmse:.6f} rad/s")

        # --- Error Plot (Rollout, Original Scale) ---
        fig_err, axes_err = plt.subplots(num_joints_to_plot, 2, figsize=(16, 3 * num_joints_to_plot), sharex=True)
        if num_joints_to_plot == 1: axes_err = np.array([[axes_err[0], axes_err[1]]])
        fig_err.suptitle('Rollout Prediction Error (Prediction - Ground Truth)')
        for i in range(num_joints_to_plot):
             axes_err[i, 0].plot(time_axis, np.rad2deg(rollout_qpos_error[:, i]), label=f'{joint_names[i]} Pos Error', color='green')
             axes_err[i, 0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
             axes_err[i, 0].set_ylabel('Error (deg)')
             axes_err[i, 0].legend()
             axes_err[i, 0].grid(True)

             axes_err[i, 1].plot(time_axis, rollout_qvel_error[:, i], label=f'{joint_names[i]} Vel Error', color='purple')
             axes_err[i, 1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
             axes_err[i, 1].set_ylabel('Error (rad/s)')
             axes_err[i, 1].legend()
             axes_err[i, 1].grid(True)
        axes_err[-1, 0].set_xlabel('Time (s)'); axes_err[-1, 1].set_xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(error_plot_save_path); print(f"Rollout error plot saved to {error_plot_save_path}"); plt.close(fig_err)

    print("Script finished.")