# -*- coding: utf-8 -*-
import numpy as np
import math # For exp and sin

class VelocityCoordinator:
    """
    根据控制信号和系统总“努力度”限制来协调计算目标关节速度。
    增加了非对称的加速/减速响应延迟，并加入了停止时的阻尼振荡效果。
    (移除了触发振荡的最小速度阈值)
    """
    def __init__(self,
                 num_joints: int,
                 dt: float,
                 max_velocities: list or np.ndarray,
                 effort_factors: list or np.ndarray,
                 max_total_effort: float,
                 accel_response_time: float = 0.1,
                 decel_response_time: float = 0.2,
                 # --- 振荡参数 ---
                 oscillation_frequency: float = 15.0,
                 oscillation_decay_rate: float = 5.0,
                 initial_amplitude_scale: float = 0.3,
                 oscillation_stop_threshold: float = 0.01,
                 # min_velocity_for_oscillation: float = 0.05, # --- 移除此参数 ---
                 zero_control_threshold: float = 1.0,
                 # --- 结束 ---
                 control_signal_range=(-1000.0, 1000.0)
                 ):
        """
        初始化速度协调器。
        """
        if not (len(max_velocities) == len(effort_factors) == num_joints):
            raise ValueError("Length of parameter lists must match num_joints")

        self.num_joints = num_joints
        self.dt = dt
        self.max_velocities = np.array(max_velocities)
        self.effort_factors = np.array(effort_factors)
        self.max_total_effort = max_total_effort
        self.accel_response_time = accel_response_time
        self.decel_response_time = decel_response_time
        self.control_min, self.control_max = control_signal_range
        self.control_span = self.control_max - self.control_min

        # --- 振荡参数 ---
        self.osc_freq = oscillation_frequency
        self.osc_decay = oscillation_decay_rate
        self.osc_amp_scale = initial_amplitude_scale
        self.osc_stop_thresh = oscillation_stop_threshold
        # self.min_vel_osc = min_velocity_for_oscillation # --- 移除 ---
        self.zero_ctrl_thresh = zero_control_threshold
        # --- 结束 ---

        self.control_to_velocity_ratio = self.max_velocities / (0.5 * self.control_span)

        # 计算 alphas
        if self.accel_response_time <= 1e-6: self.alpha_accel = 1.0
        else: self.alpha_accel = 1.0 - math.exp(-self.dt / self.accel_response_time)
        if self.decel_response_time <= 1e-6: self.alpha_decel = 1.0
        else: self.alpha_decel = 1.0 - math.exp(-self.dt / self.decel_response_time)

        # 初始化状态
        self.last_output_velocities = np.zeros(self.num_joints)
        self.is_oscillating = np.zeros(self.num_joints, dtype=bool)
        self.t_osc = np.zeros(self.num_joints)
        self.initial_osc_velocity_mag = np.zeros(self.num_joints)
        self.last_control_signal_was_zero = np.ones(self.num_joints, dtype=bool)

        print("VelocityCoordinator (Asymmetric Delay + Oscillation - No Min Vel Trigger) initialized:")
        # (省略打印信息)


    def calculate_target_velocities(self,
                                    control_signals: np.ndarray
                                    ) -> np.ndarray:
        """
        计算目标速度，包含流量限制、非对称延迟和停止振荡。
        """
        # 1. 控制信号 -> 期望速度
        mid_control = (self.control_max + self.control_min) / 2.0
        desired_velocities = (control_signals - mid_control) * self.control_to_velocity_ratio
        desired_velocities = np.clip(desired_velocities, -self.max_velocities, self.max_velocities)

        # 2. 努力度限制 -> 理想目标速度
        effort_demands = np.abs(desired_velocities) * self.effort_factors
        total_effort_demand = np.sum(effort_demands)
        ideal_target_velocities = desired_velocities
        if total_effort_demand > self.max_total_effort and total_effort_demand > 1e-6:
            effort_scale_factor = self.max_total_effort / total_effort_demand
            ideal_target_velocities = desired_velocities * effort_scale_factor

        # 3. 应用非对称延迟滤波 -> 基础目标速度
        is_accelerating = np.abs(ideal_target_velocities) >= np.abs(self.last_output_velocities)
        alphas = np.where(is_accelerating, self.alpha_accel, self.alpha_decel)
        base_target_velocities = alphas * ideal_target_velocities + (1.0 - alphas) * self.last_output_velocities

        # 4. 处理停止振荡逻辑
        final_output_velocities = base_target_velocities.copy()
        current_control_is_zero = np.abs(control_signals) < self.zero_ctrl_thresh

        for j in range(self.num_joints):
            # --- 修改: 移除速度阈值判断 ---
            # 检测是否需要开始振荡: 当前控制为0 & 上一步控制非0
            start_oscillation = (current_control_is_zero[j] and
                                 not self.last_control_signal_was_zero[j] and
                                 np.abs(self.last_output_velocities[j]) > 1e-4) # 加一个极小值判断避免静止时误触发

            if start_oscillation:
                self.is_oscillating[j] = True
                self.t_osc[j] = 0.0
                self.initial_osc_velocity_mag[j] = np.abs(self.last_output_velocities[j])
            # --- 结束修改 ---

            if self.is_oscillating[j]:
                self.t_osc[j] += self.dt
                current_amplitude = (self.initial_osc_velocity_mag[j] *
                                     self.osc_amp_scale *
                                     math.exp(-self.osc_decay * self.t_osc[j]))

                if current_amplitude < self.osc_stop_thresh:
                    self.is_oscillating[j] = False
                else:
                    osc_term = current_amplitude * math.sin(self.osc_freq * self.t_osc[j])
                    final_output_velocities[j] = base_target_velocities[j] + osc_term

            if not current_control_is_zero[j]:
                 self.is_oscillating[j] = False # 如果控制恢复，停止振荡

        # 5. 更新状态
        self.last_output_velocities = final_output_velocities.copy()
        self.last_control_signal_was_zero = current_control_is_zero

        # 6. 返回最终目标速度
        return final_output_velocities

