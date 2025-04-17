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
from tqdm import tqdm # --- 恢复: 导入 tqdm ---

# ==============================================================================
# LSTM Model Definition
# ==============================================================================
class ExcavatorLSTM(nn.Module):
    # (保持不变)
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
    # (保持不变)
    def __init__(self, angle_indices=[0], non_angle_indices=[1, 2, 3], weight_angle=1.0, weight_other=1.0):
        super().__init__()
        self.angle_indices = angle_indices
        self.non_angle_indices = non_angle_indices
        self.weight_angle = weight_angle
        self.weight_other = weight_other
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true):
        qpos_pred = y_pred[:, :4]; qvel_pred = y_pred[:, 4:]
        qpos_true = y_true[:, :4]; qvel_true = y_true[:, 4:]
        loss_qvel = self.mse_loss(qvel_pred, qvel_true).mean()
        if self.non_angle_indices:
            loss_qpos_non_periodic = self.mse_loss(qpos_pred[:, self.non_angle_indices], qpos_true[:, self.non_angle_indices]).mean()
        else: loss_qpos_non_periodic = torch.tensor(0.0, device=y_pred.device)
        loss_qpos_periodic = torch.tensor(0.0, device=y_pred.device)
        if self.angle_indices:
            angle_diff = qpos_pred[:, self.angle_indices] - qpos_true[:, self.angle_indices]
            wrapped_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi
            loss_qpos_periodic = torch.mean(wrapped_diff**2)
        total_loss = (self.weight_angle * (loss_qpos_periodic + loss_qpos_non_periodic) + self.weight_other * loss_qvel)
        return total_loss

# ==============================================================================
# Helper Functions
# ==============================================================================
def create_sequences(input_features, output_labels, seq_length):
    # (保持不变)
    xs, ys = [], []
    for i in range(len(input_features) - seq_length):
        x = input_features[i:(i + seq_length)]
        y = output_labels[i + seq_length -1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def angle_difference(angle1, angle2):
    # (保持不变)
    diff = angle1 - angle2
    return (diff + np.pi) % (2 * np.pi) - np.pi

# ==============================================================================
# Main Training and Evaluation
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and Evaluate Excavator Dynamics Model (LSTM, Raw Angle Input, Custom Loss)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .npz data file from simulation')
    parser.add_argument('--output_prefix', type=str, default='excavator_lstm', help='Prefix for saving files within the results directory')
    parser.add_argument('--seq_length', type=int, default=150, help='Input sequence length')
    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate between LSTM layers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--test_split', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--qpos_indices', type=int, nargs=4, default=[0, 1, 2, 3], help='Indices of controlled joint positions in qpos array')
    parser.add_argument('--qvel_indices', type=int, nargs=4, default=[0, 1, 2, 3], help='Indices of controlled joint velocities in qvel array')
    parser.add_argument('--periodic_angle_idx', type=int, default=0, help='Index of the periodic angle (e.g., cab) within the 4 controlled qpos (0-based)')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 创建结果目录 ---
    SAVE_DIR = f"h{args.hidden_size}_l{args.num_layers}_s{args.seq_length}"
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR); print(f"Created results directory: {SAVE_DIR}")
    else: print(f"Results directory already exists: {SAVE_DIR}")

    # --- 更新文件保存路径 ---
    scaler_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_scaler.joblib")
    model_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_model.pth")
    loss_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_training_loss.png")
    rollout_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_rollout_vs_single_step.png")
    error_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_rollout_error.png")

    # --- 1. Load Data ---
    # (保持不变)
    print(f"Loading data from {args.data_path}...")
    try:
        data = np.load(args.data_path); time_data = data['time']; qpos_all = data['qpos']; qvel_all = data['qvel']; control_signal = data['control_signal']
        print(f"Data loaded. Total steps: {len(time_data)}")
    except Exception as e: print(f"Error loading data: {e}"); exit()

    # --- 2. Feature Selection and Engineering ---
    # (保持不变)
    qpos_indices = args.qpos_indices; qvel_indices = args.qvel_indices
    print(f"Using qpos indices: {qpos_indices}"); print(f"Using qvel indices: {qvel_indices}")
    try: qpos = qpos_all[:, qpos_indices]; qvel = qvel_all[:, qvel_indices]
    except IndexError as e: print(f"Error selecting indices: {e}. Check indices."); exit()
    num_steps = min(len(qpos), len(qvel), len(control_signal))
    input_features = np.hstack((control_signal[:num_steps], qpos[:num_steps], qvel[:num_steps]))
    output_qpos_labels = qpos[1:num_steps+1]; output_qvel_labels = qvel[1:num_steps+1]
    output_labels = np.hstack((output_qpos_labels, output_qvel_labels))
    input_features = input_features[:len(output_labels)]
    print(f"Input features shape: {input_features.shape}"); print(f"Output labels shape: {output_labels.shape}")
    input_size = input_features.shape[1]; output_size = output_labels.shape[1]

    # --- 3. Data Splitting and Preprocessing ---
    # (保持不变)
    print("Splitting data and scaling input features...")
    X_train, X_test, y_train, y_test = train_test_split(input_features, output_labels, test_size=args.test_split, random_state=args.seed, shuffle=False)
    scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_filename); print(f"Input feature scaler saved to {scaler_filename}")

    # --- 4. Sequence Creation ---
    # (保持不变)
    print(f"Creating sequences with length {args.seq_length}...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, args.seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, args.seq_length)
    print(f"Train sequences shape: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"Test sequences shape: X={X_test_seq.shape}, y={y_test_seq.shape}")

    # --- 5. Prepare PyTorch DataLoaders ---
    # (保持不变)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    print(f"Using device: {device}")
    train_dataset = TensorDataset(torch.FloatTensor(X_train_seq).to(device), torch.FloatTensor(y_train_seq).to(device))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_seq).to(device), torch.FloatTensor(y_test_seq).to(device))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 6. Model, Loss, Optimizer ---
    # (保持不变)
    model = ExcavatorLSTM(input_size, args.hidden_size, output_size, args.num_layers, args.dropout).to(device)
    periodic_idx = [args.periodic_angle_idx]; non_periodic_idx = [i for i in range(4) if i != args.periodic_angle_idx]
    criterion = PeriodicAngleMSELoss(angle_indices=periodic_idx, non_angle_indices=non_periodic_idx)
    print(f"Using PeriodicAngleMSELoss for angle index {periodic_idx}")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print("Model Summary:"); print(model); num_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"Total trainable parameters: {num_params}")

    # --- 7. Training Loop (修改: 恢复 tqdm) ---
    print("Starting training...")
    train_losses = []
    best_loss = float('inf')
    epochs_no_improve = 0
    total_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        # --- 修改: 恢复 tqdm 进度条 ---
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Train", leave=False)
        for batch_x, batch_y in progress_bar:
        # --- 结束修改 ---
            batch_x, batch_y = batch_x.to(device), batch_y.to(device) # Move data to device
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # --- 修改: 恢复 progress_bar.set_postfix ---
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            # --- 结束修改 ---

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # 打印 Epoch 总结信息 (包含耗时)
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.6f}, Time: {epoch_duration:.2f}s')

        # Simple early stopping based on training loss improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_filename)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= 10:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"Training finished. Total time: {total_training_time:.2f}s")
    # --- 结束训练循环修改 ---

    # --- 8. Save Model (加载最佳模型) ---
    # (保持不变)
    if os.path.exists(model_filename):
        print(f"Loading best model from {model_filename} for evaluation...")
        model.load_state_dict(torch.load(model_filename, map_location=device))
    else:
        print("Warning: No best model file found. Evaluating model from last epoch.")

    # --- 9. Evaluation ---
    # (保持不变)
    print("Evaluating model...")
    model.eval(); test_loss_single_step = 0
    all_single_step_preds = []; all_single_step_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs, _ = model(batch_x); loss = criterion(outputs, batch_y); test_loss_single_step += loss.item()
            all_single_step_preds.append(outputs.cpu().numpy()); all_single_step_targets.append(batch_y.cpu().numpy())
    avg_test_loss_single_step = test_loss_single_step / len(test_loader)
    print(f"Average Test Set Loss (Single Step Prediction, Custom Loss): {avg_test_loss_single_step:.6f}")

    # --- 10. Multi-step Rollout Prediction & Single-step Prediction on Rollout Segment ---
    # (保持不变)
    print("Performing multi-step rollout and single-step prediction on a test segment...")
    start_index = 0
    if len(X_test_seq) <= start_index: print("Not enough test sequences for evaluation.")
    else:
        rollout_steps = len(X_test_seq) - start_index
        current_sequence_rollout = torch.FloatTensor(X_test_seq[start_index:start_index+1]).to(device)
        predictions_rollout_pos = []; predictions_rollout_vel = []
        last_predicted_qpos_rollout = y_test_seq[start_index, :4].flatten(); last_predicted_qvel_rollout = y_test_seq[start_index, 4:].flatten()
        predictions_rollout_pos.append(last_predicted_qpos_rollout); predictions_rollout_vel.append(last_predicted_qvel_rollout)
        current_h_c_rollout = None
        with torch.no_grad():
            for i in range(1, rollout_steps):
                predicted_output_tensor, current_h_c_rollout = model(current_sequence_rollout, current_h_c_rollout)
                predicted_output = predicted_output_tensor.cpu().numpy().flatten()
                next_qpos_pred = predicted_output[:4]; next_qvel_pred = predicted_output[4:]
                predictions_rollout_pos.append(next_qpos_pred); predictions_rollout_vel.append(next_qvel_pred)
                next_control_idx = start_index + args.seq_length + i - 1
                if next_control_idx >= len(X_test): break
                next_control_signal = X_test[next_control_idx, :4]
                next_qpos = next_qpos_pred; next_qvel = next_qvel_pred
                next_input_frame_unscaled = np.concatenate((next_control_signal, next_qpos, next_qvel))
                next_input_frame_scaled = scaler.transform(next_input_frame_unscaled.reshape(1, -1)).flatten()
                next_sequence_np = np.vstack((current_sequence_rollout.cpu().numpy()[0, 1:, :], next_input_frame_scaled))
                current_sequence_rollout = torch.FloatTensor(next_sequence_np).unsqueeze(0).to(device)
        predictions_rollout_pos = np.array(predictions_rollout_pos); predictions_rollout_vel = np.array(predictions_rollout_vel)

        predictions_single_step_pos = []; predictions_single_step_vel = []
        with torch.no_grad():
             for i in range(start_index, start_index + len(predictions_rollout_pos)):
                  if i >= len(X_test_seq): break
                  input_seq_tensor = torch.FloatTensor(X_test_seq[i:i+1]).to(device)
                  pred_output_tensor, _ = model(input_seq_tensor)
                  pred_output = pred_output_tensor.cpu().numpy().flatten()
                  predictions_single_step_pos.append(pred_output[:4]); predictions_single_step_vel.append(pred_output[4:])
        predictions_single_step_pos = np.array(predictions_single_step_pos); predictions_single_step_vel = np.array(predictions_single_step_vel)

        ground_truth_rollout_pos = y_test_seq[start_index : start_index + len(predictions_rollout_pos), :4]
        ground_truth_rollout_vel = y_test_seq[start_index : start_index + len(predictions_rollout_vel), 4:]

        # --- 11. Plotting Comparison ---
        # (保持不变, 除了保存路径)
        print("Plotting comparison results...")
        plt.figure(figsize=(10, 5)); plt.plot(train_losses, label='Training Loss'); plt.title('Training Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Custom Loss'); plt.legend(); plt.grid(True)
        plt.savefig(loss_plot_save_path); plt.close()

        num_joints_to_plot = 4
        fig, axes = plt.subplots(num_joints_to_plot, 2, figsize=(16, 3 * num_joints_to_plot), sharex=True)
        if num_joints_to_plot == 1: axes = np.array([[axes[0], axes[1]]])
        dt_est = time_data[1]-time_data[0] if len(time_data)>1 else 0.02
        time_axis = np.arange(len(predictions_rollout_pos)) * dt_est
        joint_names = ['Cab', 'Boom', 'Arm', 'Bucket']
        for i in range(num_joints_to_plot):
            axes[i, 0].plot(time_axis, np.rad2deg(ground_truth_rollout_pos[:, i]), label='Ground Truth', color='blue', linewidth=1.5)
            axes[i, 0].plot(time_axis, np.rad2deg(predictions_rollout_pos[:, i]), label='Rollout Pred', color='red', linestyle='--', linewidth=1)
            axes[i, 0].plot(time_axis, np.rad2deg(predictions_single_step_pos[:, i]), label='Single-Step Pred', color='green', linestyle=':', linewidth=1)
            axes[i, 0].set_ylabel(f'{joint_names[i]} Pos (deg)'); axes[i, 0].legend(); axes[i, 0].grid(True)
            axes[i, 1].plot(time_axis, ground_truth_rollout_vel[:, i], label='Ground Truth', color='blue', linewidth=1.5)
            axes[i, 1].plot(time_axis, predictions_rollout_vel[:, i], label='Rollout Pred', color='red', linestyle='--', linewidth=1)
            axes[i, 1].plot(time_axis, predictions_single_step_vel[:, i], label='Single-Step Pred', color='green', linestyle=':', linewidth=1)
            axes[i, 1].set_ylabel(f'{joint_names[i]} Vel (rad/s)'); axes[i, 1].legend(); axes[i, 1].grid(True)
        axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')
        fig.suptitle('Single-Step vs Multi-step Rollout Prediction vs Ground Truth (LSTM - Raw Angle Input, Custom Loss)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(rollout_plot_save_path); print(f"Comparison plot saved to {rollout_plot_save_path}"); plt.close(fig)

        # --- 12. Calculate & Print Errors for Both Modes ---
        # (保持不变, 除了绘图保存路径)
        rollout_qpos_error = np.zeros_like(predictions_rollout_pos); rollout_qpos_error[:, non_periodic_idx] = predictions_rollout_pos[:, non_periodic_idx] - ground_truth_rollout_pos[:, non_periodic_idx]; rollout_qpos_error[:, periodic_idx] = angle_difference(predictions_rollout_pos[:, periodic_idx], ground_truth_rollout_pos[:, periodic_idx])
        rollout_qvel_error = predictions_rollout_vel - ground_truth_rollout_vel
        rollout_qpos_mae = np.mean(np.abs(rollout_qpos_error)); rollout_qvel_mae = np.mean(np.abs(rollout_qvel_error)); rollout_qpos_rmse = np.sqrt(np.mean(rollout_qpos_error**2)); rollout_qvel_rmse = np.sqrt(np.mean(rollout_qvel_error**2))
        print("\n--- Rollout Prediction Errors ---"); print(f"Qpos MAE: {rollout_qpos_mae:.6f} rad"); print(f"Qvel MAE: {rollout_qvel_mae:.6f} rad/s"); print(f"Qpos RMSE: {rollout_qpos_rmse:.6f} rad"); print(f"Qvel RMSE: {rollout_qvel_rmse:.6f} rad/s")

        single_step_qpos_error = np.zeros_like(predictions_single_step_pos); single_step_qpos_error[:, non_periodic_idx] = predictions_single_step_pos[:, non_periodic_idx] - ground_truth_rollout_pos[:, non_periodic_idx]; single_step_qpos_error[:, periodic_idx] = angle_difference(predictions_single_step_pos[:, periodic_idx], ground_truth_rollout_pos[:, periodic_idx])
        single_step_qvel_error = predictions_single_step_vel - ground_truth_rollout_vel
        single_step_qpos_mae = np.mean(np.abs(single_step_qpos_error)); single_step_qvel_mae = np.mean(np.abs(single_step_qvel_error)); single_step_qpos_rmse = np.sqrt(np.mean(single_step_qpos_error**2)); single_step_qvel_rmse = np.sqrt(np.mean(single_step_qvel_error**2))
        print("\n--- Single-Step Prediction Errors (on rollout segment) ---"); print(f"Qpos MAE: {single_step_qpos_mae:.6f} rad"); print(f"Qvel MAE: {single_step_qvel_mae:.6f} rad/s"); print(f"Qpos RMSE: {single_step_qpos_rmse:.6f} rad"); print(f"Qvel RMSE: {single_step_qvel_rmse:.6f} rad/s")

        # --- Error Plot ---
        fig_err, axes_err = plt.subplots(num_joints_to_plot, 2, figsize=(16, 3 * num_joints_to_plot), sharex=True)
        if num_joints_to_plot == 1: axes_err = np.array([[axes_err[0], axes_err[1]]])
        fig_err.suptitle('Rollout Prediction Error (Prediction - Ground Truth)')
        for i in range(num_joints_to_plot):
             axes_err[i, 0].plot(time_axis, np.rad2deg(rollout_qpos_error[:, i]), label=f'{joint_names[i]} Pos Error', color='green'); axes_err[i, 0].axhline(0, color='gray', linestyle='--', linewidth=0.8); axes_err[i, 0].set_ylabel('Error (deg)'); axes_err[i, 0].legend(); axes_err[i, 0].grid(True)
             axes_err[i, 1].plot(time_axis, rollout_qvel_error[:, i], label=f'{joint_names[i]} Vel Error', color='purple'); axes_err[i, 1].axhline(0, color='gray', linestyle='--', linewidth=0.8); axes_err[i, 1].set_ylabel('Error (rad/s)'); axes_err[i, 1].legend(); axes_err[i, 1].grid(True)
        axes_err[-1, 0].set_xlabel('Time (s)'); axes_err[-1, 1].set_xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(error_plot_save_path); print(f"Rollout error plot saved to {error_plot_save_path}"); plt.close(fig_err) # Use updated path

    print("Script finished.")

