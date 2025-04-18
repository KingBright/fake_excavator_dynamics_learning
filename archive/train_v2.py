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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import random
import copy
import time
from tqdm import tqdm

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
# 自定义损失函数 (处理角度周期性, 可选处理归一化角度)
# ==============================================================================
class PeriodicAngleMSELoss(nn.Module):
    # (修改: 增加 qpos_scaler 参数)
    def __init__(self, angle_indices=[0], non_angle_indices=[1, 2, 3],
                 qpos_scaler=None, # --- 新增: 传入 qpos 输出 scaler ---
                 weight_angle=1.0, weight_other=1.0):
        """
        自定义损失函数，特殊处理周期性角度的误差。
        如果传入了 qpos_scaler，则先对角度进行逆变换再计算周期误差。
        """
        super().__init__()
        self.angle_indices = angle_indices
        self.non_angle_indices = non_angle_indices
        self.qpos_scaler = qpos_scaler # Store the scaler
        self.weight_angle = weight_angle
        self.weight_other = weight_other
        self.mse_loss = nn.MSELoss(reduction='none') # Calculate element-wise squared error

    def forward(self, y_pred, y_true):
        """
        计算损失。
        Args:
            y_pred (Tensor): 预测值 (可能已归一化), 形状 (batch, 8)
            y_true (Tensor): 真实值 (可能已归一化), 形状 (batch, 8)
        """
        qpos_pred_maybe_scaled = y_pred[:, :4]
        qvel_pred_scaled = y_pred[:, 4:] # Assume qvel is always scaled if output scaling is used
        qpos_true_maybe_scaled = y_true[:, :4]
        qvel_true_scaled = y_true[:, 4:] # Assume qvel is always scaled if output scaling is used

        # --- 计算速度部分的 MSE Loss (在归一化空间计算) ---
        loss_qvel = self.mse_loss(qvel_pred_scaled, qvel_true_scaled).mean()

        # --- 计算非周期性角度的 MSE Loss (在归一化空间计算) ---
        if self.non_angle_indices:
            loss_qpos_non_periodic = self.mse_loss(qpos_pred_maybe_scaled[:, self.non_angle_indices],
                                                   qpos_true_maybe_scaled[:, self.non_angle_indices]).mean()
        else:
            loss_qpos_non_periodic = torch.tensor(0.0, device=y_pred.device)

        # --- 计算周期性角度的 Loss (在原始空间计算) ---
        loss_qpos_periodic = torch.tensor(0.0, device=y_pred.device)
        if self.angle_indices:
            # 如果提供了 scaler，先逆变换回原始角度尺度
            if self.qpos_scaler is not None:
                 with torch.no_grad(): # Inverse transform should not require gradients
                     # Need to move to CPU for sklearn scaler
                     qpos_pred_np = qpos_pred_maybe_scaled.detach().cpu().numpy()
                     qpos_true_np = qpos_true_maybe_scaled.detach().cpu().numpy()
                     qpos_pred_orig = torch.from_numpy(self.qpos_scaler.inverse_transform(qpos_pred_np)).to(y_pred.device)
                     qpos_true_orig = torch.from_numpy(self.qpos_scaler.inverse_transform(qpos_true_np)).to(y_pred.device)
            else:
                 # 如果没有 scaler，则假定输入已经是原始角度
                 qpos_pred_orig = qpos_pred_maybe_scaled
                 qpos_true_orig = qpos_true_maybe_scaled

            # 计算原始角度空间中的最短角度差
            angle_diff = qpos_pred_orig[:, self.angle_indices] - qpos_true_orig[:, self.angle_indices]
            wrapped_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi
            # 计算平方误差并求平均
            loss_qpos_periodic = torch.mean(wrapped_diff**2)

        # 加权组合损失
        # 注意：周期性角度误差是在原始空间算的，其他是在归一化空间算的，直接加权可能不理想
        # 更好的方式可能是分别计算原始空间的误差再加权，或者调整权重
        # 这里暂时保持简单加权，如果效果不好需要调整此部分
        total_loss = (self.weight_angle * (loss_qpos_periodic + loss_qpos_non_periodic) +
                      self.weight_other * loss_qvel)

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

def wrap_angle_to_pi(angle):
    # (保持不变)
    return (angle + np.pi) % (2 * np.pi) - np.pi

# ==============================================================================
# Main Training and Evaluation
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and Evaluate Excavator Dynamics Model (LSTM, Raw Angle Input, Custom Loss, Output Scaling, Regularization)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .npz data file from simulation')
    parser.add_argument('--output_prefix', type=str, default='excavator_lstm', help='Prefix for saving files within the results directory')
    parser.add_argument('--seq_length', type=int, default=150, help='Input sequence length')
    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate between LSTM layers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam optimizer')
    # --- 新增: L2 正则化权重 ---
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty) for Adam optimizer')
    # --- 结束新增 ---
    parser.add_argument('--test_split', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--qpos_indices', type=int, nargs=4, default=[0, 1, 2, 3], help='Indices of controlled joint positions in qpos array')
    parser.add_argument('--qvel_indices', type=int, nargs=4, default=[0, 1, 2, 3], help='Indices of controlled joint velocities in qvel array')
    parser.add_argument('--periodic_angle_idx', type=int, default=0, help='Index of the periodic angle (e.g., cab) within the 4 controlled qpos (0-based)')
    # --- 新增: 是否进行输出归一化 ---
    parser.add_argument('--scale_output', action='store_true', help='Scale output features (qpos, qvel) separately')
    # --- 结束新增 ---

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 创建结果目录 ---
    SAVE_DIR = f"h{args.hidden_size}_l{args.num_layers}_s{args.seq_length}"
    if args.scale_output: SAVE_DIR += "_scaledout"
    if args.weight_decay > 0: SAVE_DIR += f"_wd{args.weight_decay:.0e}"
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR); print(f"Created results directory: {SAVE_DIR}")
    else: print(f"Results directory already exists: {SAVE_DIR}")

    # --- 更新文件保存路径 ---
    scaler_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_scaler_input.joblib") # Input scaler
    scaler_qpos_out_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_scaler_qpos_out.joblib") # Output qpos scaler
    scaler_qvel_out_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_scaler_qvel_out.joblib") # Output qvel scaler
    model_filename = os.path.join(SAVE_DIR, f"{args.output_prefix}_model.pth")
    loss_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_training_loss.png")
    rollout_plot_save_path = os.path.join(SAVE_DIR, f"{args.output_prefix}_rollout_vs_single_step.png")
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

    # 输入特征: control (4), qpos (4), qvel (4) -> 12 维
    num_steps = min(len(qpos), len(qvel), len(control_signal))
    input_features = np.hstack((control_signal[:num_steps], qpos[:num_steps], qvel[:num_steps]))

    # 输出标签: next qpos (4维) + next qvel (4维) -> 8维
    output_qpos_labels = qpos[1:num_steps+1]
    output_qvel_labels = qvel[1:num_steps+1]
    output_labels = np.hstack((output_qpos_labels, output_qvel_labels))

    input_features = input_features[:len(output_labels)]
    print(f"Input features shape: {input_features.shape}"); print(f"Output labels shape: {output_labels.shape}")
    input_size = input_features.shape[1]; output_size = output_labels.shape[1]

    # --- 3. Data Splitting and Preprocessing ---
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        input_features, output_labels, test_size=args.test_split, random_state=args.seed, shuffle=False
    )

    # 标准化输入特征
    print("Scaling input features...")
    scaler_input = StandardScaler(); X_train_scaled = scaler_input.fit_transform(X_train); X_test_scaled = scaler_input.transform(X_test)
    joblib.dump(scaler_input, scaler_filename); print(f"Input feature scaler saved to {scaler_filename}")

    # --- 新增: 输出归一化 ---
    y_train_scaled = y_train.copy() # Start with original scale
    y_test_scaled = y_test.copy()
    scaler_qpos_out = None
    scaler_qvel_out = None

    if args.scale_output:
        print("Scaling output features (qpos and qvel separately)...")
        y_train_qpos = y_train[:, :4]; y_train_qvel = y_train[:, 4:]
        y_test_qpos = y_test[:, :4];   y_test_qvel = y_test[:, 4:]

        scaler_qpos_out = StandardScaler()
        y_train_qpos_scaled = scaler_qpos_out.fit_transform(y_train_qpos)
        y_test_qpos_scaled = scaler_qpos_out.transform(y_test_qpos)
        joblib.dump(scaler_qpos_out, scaler_qpos_out_filename)
        print(f"Output qpos scaler saved to {scaler_qpos_out_filename}")

        scaler_qvel_out = StandardScaler()
        y_train_qvel_scaled = scaler_qvel_out.fit_transform(y_train_qvel)
        y_test_qvel_scaled = scaler_qvel_out.transform(y_test_qvel)
        joblib.dump(scaler_qvel_out, scaler_qvel_out_filename)
        print(f"Output qvel scaler saved to {scaler_qvel_out_filename}")

        # Combine scaled outputs
        y_train_scaled = np.hstack((y_train_qpos_scaled, y_train_qvel_scaled))
        y_test_scaled = np.hstack((y_test_qpos_scaled, y_test_qvel_scaled))
    else:
        print("Output features will not be scaled.")
    # --- 结束新增 ---

    # --- 4. Sequence Creation ---
    print(f"Creating sequences with length {args.seq_length}...")
    # Use scaled inputs and potentially scaled outputs
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, args.seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, args.seq_length)
    print(f"Train sequences shape: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"Test sequences shape: X={X_test_seq.shape}, y={y_test_seq.shape}")

    # --- 5. Prepare PyTorch DataLoaders ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    print(f"Using device: {device}")
    train_dataset = TensorDataset(torch.FloatTensor(X_train_seq).to(device), torch.FloatTensor(y_train_seq).to(device))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_seq).to(device), torch.FloatTensor(y_test_seq).to(device))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 6. Model, Loss, Optimizer ---
    model = ExcavatorLSTM(input_size, args.hidden_size, output_size, args.num_layers, args.dropout).to(device)
    periodic_idx_list = [args.periodic_angle_idx]; non_periodic_idx_list = [i for i in range(4) if i != args.periodic_angle_idx]
    # --- 修改: 将 qpos_scaler 传入损失函数 (如果使用了输出缩放) ---
    criterion = PeriodicAngleMSELoss(angle_indices=periodic_idx_list,
                                     non_angle_indices=non_periodic_idx_list,
                                     qpos_scaler=scaler_qpos_out if args.scale_output else None)
    print(f"Using PeriodicAngleMSELoss for angle index {periodic_idx_list}")
    if args.scale_output: print("Loss function will use inverse scaling for periodic angle error.")
    # --- 结束修改 ---
    # --- 修改: 在优化器中加入 weight_decay ---
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Using Adam optimizer with lr={args.learning_rate} and weight_decay={args.weight_decay}")
    # --- 结束修改 ---
    print("Model Summary:"); print(model); num_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"Total trainable parameters: {num_params}")

    # --- 7. Training Loop ---
    # (保持不变, 使用 tqdm)
    print("Starting training...")
    train_losses = []; best_loss = float('inf'); epochs_no_improve = 0
    total_start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start_time = time.time(); model.train(); epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Train", leave=False)
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device) # y is potentially scaled
            optimizer.zero_grad(); outputs, _ = model(batch_x); loss = criterion(outputs, batch_y) # loss calculated potentially on scaled data (except wrapped angle)
            loss.backward(); optimizer.step(); epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
        avg_loss = epoch_loss / len(train_loader); train_losses.append(avg_loss)
        epoch_end_time = time.time(); epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.6f}, Time: {epoch_duration:.2f}s')
        if avg_loss < best_loss: best_loss = avg_loss; epochs_no_improve = 0; torch.save(model.state_dict(), model_filename)
        else: epochs_no_improve += 1
        if epochs_no_improve >= 10: print(f"Early stopping triggered at epoch {epoch+1}"); break
    total_end_time = time.time(); total_training_time = total_end_time - total_start_time
    print(f"Training finished. Total time: {total_training_time:.2f}s")

    # --- 8. Load Best Model ---
    if os.path.exists(model_filename): print(f"Loading best model from {model_filename} for evaluation..."); model.load_state_dict(torch.load(model_filename, map_location=device))
    else: print("Warning: No best model file found. Evaluating model from last epoch.")

    # --- 9. Evaluation ---
    print("Evaluating model..."); model.eval(); test_loss_single_step = 0
    all_single_step_preds_scaled = []; all_single_step_targets_scaled = []
    with torch.no_grad():
        for batch_x, batch_y_scaled in test_loader: # Use scaled targets from loader
            outputs_scaled, _ = model(batch_x); loss = criterion(outputs_scaled, batch_y_scaled); test_loss_single_step += loss.item()
            all_single_step_preds_scaled.append(outputs_scaled.cpu().numpy()); all_single_step_targets_scaled.append(batch_y_scaled.cpu().numpy())
    avg_test_loss_single_step = test_loss_single_step / len(test_loader)
    print(f"Average Test Set Loss (Single Step Prediction, Custom Loss on potentially scaled data): {avg_test_loss_single_step:.6f}")

    # --- 10. Multi-step Rollout Prediction & Single-step Prediction on Rollout Segment ---
    print("Performing multi-step rollout and single-step prediction on a test segment...")
    start_index = 0
    if len(X_test_seq) <= start_index: print("Not enough test sequences for evaluation.")
    else:
        rollout_steps = len(X_test_seq) - start_index
        current_sequence_rollout = torch.FloatTensor(X_test_seq[start_index:start_index+1]).to(device)
        # Store predictions in ORIGINAL scale
        predictions_rollout_pos = []
        predictions_rollout_vel = []
        # Get the first target state in ORIGINAL scale
        first_target_unscaled = y_test[start_index] # Get from original unscaled test data
        last_predicted_qpos = first_target_unscaled[:4].flatten()
        last_predicted_qvel = first_target_unscaled[4:].flatten()
        predictions_rollout_pos.append(last_predicted_qpos)
        predictions_rollout_vel.append(last_predicted_qvel)
        current_h_c_rollout = None

        with torch.no_grad():
            for i in range(1, rollout_steps):
                # Model predicts SCALED state
                predicted_output_scaled_tensor, current_h_c_rollout = model(current_sequence_rollout, current_h_c_rollout)
                predicted_output_scaled = predicted_output_scaled_tensor.cpu().numpy().flatten()

                # --- 修改: Inverse transform 预测结果 ---
                pred_qpos_scaled = predicted_output_scaled[:4].reshape(1, -1)
                pred_qvel_scaled = predicted_output_scaled[4:].reshape(1, -1)

                if args.scale_output:
                    next_qpos_pred = scaler_qpos_out.inverse_transform(pred_qpos_scaled).flatten()
                    next_qvel_pred = scaler_qvel_out.inverse_transform(pred_qvel_scaled).flatten()
                else:
                    next_qpos_pred = pred_qpos_scaled.flatten()
                    next_qvel_pred = pred_qvel_scaled.flatten()
                # --- 结束修改 ---

                # Wrap 预测出的周期性角度 (在原始尺度上)
                next_qpos_pred[periodic_idx_list] = wrap_angle_to_pi(next_qpos_pred[periodic_idx_list])

                predictions_rollout_pos.append(next_qpos_pred); predictions_rollout_vel.append(next_qvel_pred)

                # --- 修改: 构建下一个输入帧 ---
                # 获取真实的下一时刻控制信号 (来自原始未缩放的 X_test)
                next_control_idx = start_index + args.seq_length + i - 1
                if next_control_idx >= len(X_test): break
                next_control_signal = X_test[next_control_idx, :4]

                # 使用预测的 qpos (wrap 后的) 和 qvel (已 inverse transform)
                next_qpos = next_qpos_pred
                next_qvel = next_qvel_pred

                # 组合成下一帧的输入特征 (12维, 原始尺度)
                next_input_frame_unscaled = np.concatenate((next_control_signal, next_qpos, next_qvel))
                # 标准化 (使用输入 scaler)
                next_input_frame_scaled = scaler_input.transform(next_input_frame_unscaled.reshape(1, -1)).flatten()
                # --- 结束修改 ---

                # 更新输入序列
                next_sequence_np = np.vstack((current_sequence_rollout.cpu().numpy()[0, 1:, :], next_input_frame_scaled))
                current_sequence_rollout = torch.FloatTensor(next_sequence_np).unsqueeze(0).to(device)

        predictions_rollout_pos = np.array(predictions_rollout_pos); predictions_rollout_vel = np.array(predictions_rollout_vel)

        # --- Single-Step Prediction for the same segment ---
        predictions_single_step_pos = []; predictions_single_step_vel = []
        with torch.no_grad():
             # 使用 test_loader 更方便，但需要找到对应的 batch
             # 或者直接用 X_test_seq 循环
             for i in range(start_index, start_index + len(predictions_rollout_pos)):
                  if i >= len(X_test_seq): break
                  input_seq_tensor = torch.FloatTensor(X_test_seq[i:i+1]).to(device)
                  pred_output_scaled_tensor, _ = model(input_seq_tensor)
                  pred_output_scaled = pred_output_scaled_tensor.cpu().numpy().flatten()
                  # Inverse transform
                  pred_qpos_s = pred_output_scaled[:4].reshape(1, -1)
                  pred_qvel_s = pred_output_scaled[4:].reshape(1, -1)
                  if args.scale_output:
                      pred_qpos = scaler_qpos_out.inverse_transform(pred_qpos_s).flatten()
                      pred_qvel = scaler_qvel_out.inverse_transform(pred_qvel_s).flatten()
                  else:
                      pred_qpos = pred_qpos_s.flatten()
                      pred_qvel = pred_qvel_s.flatten()
                  # Wrap angle
                  pred_qpos[periodic_idx_list] = wrap_angle_to_pi(pred_qpos[periodic_idx_list])
                  predictions_single_step_pos.append(pred_qpos); predictions_single_step_vel.append(pred_qvel)

        predictions_single_step_pos = np.array(predictions_single_step_pos); predictions_single_step_vel = np.array(predictions_single_step_vel)

        # --- Ground Truth for the segment (原始尺度) ---
        ground_truth_rollout_pos = y_test[start_index : start_index + len(predictions_rollout_pos), :4]
        ground_truth_rollout_vel = y_test[start_index : start_index + len(predictions_rollout_vel), 4:]

        # --- 11. Plotting Comparison (使用原始尺度数据) ---
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
            # Qpos plot
            axes[i, 0].plot(time_axis, np.rad2deg(ground_truth_rollout_pos[:, i]), label='Ground Truth', color='blue', linewidth=1.5)
            axes[i, 0].plot(time_axis, np.rad2deg(predictions_rollout_pos[:, i]), label='Rollout Pred', color='red', linestyle='--', linewidth=1)
            axes[i, 0].plot(time_axis, np.rad2deg(predictions_single_step_pos[:, i]), label='Single-Step Pred', color='green', linestyle=':', linewidth=1)
            axes[i, 0].set_ylabel(f'{joint_names[i]} Pos (deg)'); axes[i, 0].legend(); axes[i, 0].grid(True)
            # Qvel plot
            axes[i, 1].plot(time_axis, ground_truth_rollout_vel[:, i], label='Ground Truth', color='blue', linewidth=1.5)
            axes[i, 1].plot(time_axis, predictions_rollout_vel[:, i], label='Rollout Pred', color='red', linestyle='--', linewidth=1)
            axes[i, 1].plot(time_axis, predictions_single_step_vel[:, i], label='Single-Step Pred', color='green', linestyle=':', linewidth=1)
            axes[i, 1].set_ylabel(f'{joint_names[i]} Vel (rad/s)'); axes[i, 1].legend(); axes[i, 1].grid(True)
        axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')
        fig.suptitle('Single-Step vs Multi-step Rollout Prediction vs Ground Truth (LSTM - Raw Angle Input, Custom Loss)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(rollout_plot_save_path); print(f"Comparison plot saved to {rollout_plot_save_path}"); plt.close(fig)

        # --- 12. Calculate & Print Errors for Both Modes (在原始尺度计算) ---
        rollout_qpos_error = np.zeros_like(predictions_rollout_pos); rollout_qpos_error[:, non_periodic_idx_list] = predictions_rollout_pos[:, non_periodic_idx_list] - ground_truth_rollout_pos[:, non_periodic_idx_list]; rollout_qpos_error[:, periodic_idx_list] = angle_difference(predictions_rollout_pos[:, periodic_idx_list], ground_truth_rollout_pos[:, periodic_idx_list])
        rollout_qvel_error = predictions_rollout_vel - ground_truth_rollout_vel
        rollout_qpos_mae = np.mean(np.abs(rollout_qpos_error)); rollout_qvel_mae = np.mean(np.abs(rollout_qvel_error)); rollout_qpos_rmse = np.sqrt(np.mean(rollout_qpos_error**2)); rollout_qvel_rmse = np.sqrt(np.mean(rollout_qvel_error**2))
        print("\n--- Rollout Prediction Errors ---"); print(f"Qpos MAE: {rollout_qpos_mae:.6f} rad"); print(f"Qvel MAE: {rollout_qvel_mae:.6f} rad/s"); print(f"Qpos RMSE: {rollout_qpos_rmse:.6f} rad"); print(f"Qvel RMSE: {rollout_qvel_rmse:.6f} rad/s")

        single_step_qpos_error = np.zeros_like(predictions_single_step_pos); single_step_qpos_error[:, non_periodic_idx_list] = predictions_single_step_pos[:, non_periodic_idx_list] - ground_truth_rollout_pos[:, non_periodic_idx_list]; single_step_qpos_error[:, periodic_idx_list] = angle_difference(predictions_single_step_pos[:, periodic_idx_list], ground_truth_rollout_pos[:, periodic_idx_list])
        single_step_qvel_error = predictions_single_step_vel - ground_truth_rollout_vel
        single_step_qpos_mae = np.mean(np.abs(single_step_qpos_error)); single_step_qvel_mae = np.mean(np.abs(single_step_qvel_error)); single_step_qpos_rmse = np.sqrt(np.mean(single_step_qpos_error**2)); single_step_qvel_rmse = np.sqrt(np.mean(single_step_qvel_error**2))
        print("\n--- Single-Step Prediction Errors (on rollout segment) ---"); print(f"Qpos MAE: {single_step_qpos_mae:.6f} rad"); print(f"Qvel MAE: {single_step_qvel_mae:.6f} rad/s"); print(f"Qpos RMSE: {single_step_qpos_rmse:.6f} rad"); print(f"Qvel RMSE: {single_step_qvel_rmse:.6f} rad/s")

        # --- Error Plot (在原始尺度绘制) ---
        fig_err, axes_err = plt.subplots(num_joints_to_plot, 2, figsize=(16, 3 * num_joints_to_plot), sharex=True)
        if num_joints_to_plot == 1: axes_err = np.array([[axes_err[0], axes_err[1]]])
        fig_err.suptitle('Rollout Prediction Error (Prediction - Ground Truth)')
        for i in range(num_joints_to_plot):
             axes_err[i, 0].plot(time_axis, np.rad2deg(rollout_qpos_error[:, i]), label=f'{joint_names[i]} Pos Error', color='green'); axes_err[i, 0].axhline(0, color='gray', linestyle='--', linewidth=0.8); axes_err[i, 0].set_ylabel('Error (deg)'); axes_err[i, 0].legend(); axes_err[i, 0].grid(True)
             axes_err[i, 1].plot(time_axis, rollout_qvel_error[:, i], label=f'{joint_names[i]} Vel Error', color='purple'); axes_err[i, 1].axhline(0, color='gray', linestyle='--', linewidth=0.8); axes_err[i, 1].set_ylabel('Error (rad/s)'); axes_err[i, 1].legend(); axes_err[i, 1].grid(True)
        axes_err[-1, 0].set_xlabel('Time (s)'); axes_err[-1, 1].set_xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(error_plot_save_path); print(f"Rollout error plot saved to {error_plot_save_path}"); plt.close(fig_err)

    print("Script finished.")

