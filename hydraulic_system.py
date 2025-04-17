# -*- coding: utf-8 -*-
import numpy as np

class RuleBasedHydraulics:
    """
    基于规则的简化液压系统模型。
    - 控制信号映射到期望力矩。
    - 基于控制信号估算流量需求，超限时缩放期望力矩。
    - 当控制信号为零时，施加较大的制动阻尼。
    - 应用力矩变化率和最大力矩限制。
    """
    def __init__(self,
                 num_joints: int,
                 max_torques: list or np.ndarray,         # 各关节最大力矩 (Nm)
                 flow_demand_factors: list or np.ndarray, # 控制信号到流量需求的转换系数 (L/min per control unit) - 需要估算!
                 pump_max_flow: float,                    # 泵最大总流量 (L/min)
                 braking_damping_coeff: list or np.ndarray, # 控制为零时的制动阻尼系数 (Nm/(rad/s)) - 需要调整!
                 control_signal_range=(-1000.0, 1000.0), # 输入控制信号范围
                 torque_rise_time=0.02,                  # 力矩上升时间 (s)
                 zero_control_threshold=1.0              # 判断控制为零的阈值 (信号绝对值)
                 ):
        """
        初始化基于规则的液压模型。
        """
        if not (len(max_torques) == len(flow_demand_factors) == len(braking_damping_coeff) == num_joints):
            raise ValueError("Length of parameter lists must match num_joints")

        self.num_joints = num_joints
        self.max_torques = np.array(max_torques)
        self.flow_demand_factors = np.array(flow_demand_factors)
        self.pump_max_flow = pump_max_flow
        self.control_min = control_signal_range[0]
        self.control_max = control_signal_range[1]
        self.control_span = self.control_max - self.control_min
        self.torque_rise_time = torque_rise_time
        self.braking_damping_coeff = np.array(braking_damping_coeff) # 新增
        self.zero_control_threshold = zero_control_threshold       # 新增

        # 控制信号到期望力矩的映射比例
        self.control_to_torque_ratio = self.max_torques / (0.5 * self.control_span)

        # 计算最大力矩变化率
        if self.torque_rise_time <= 1e-6:
             self.max_torque_rate = np.full(self.num_joints, np.inf)
        else:
             self.max_torque_rate = self.max_torques / self.torque_rise_time

        print("RuleBasedHydraulics (with Braking Logic) initialized:")
        print(f"  Max Torques: {self.max_torques}")
        print(f"  Flow Demand Factors: {self.flow_demand_factors}")
        print(f"  Pump Max Flow: {self.pump_max_flow}")
        print(f"  Max Torque Rate: {self.max_torque_rate}")
        print(f"  Braking Damping Coeff: {self.braking_damping_coeff}")


    def calculate_torques(self,
                          control_signals: np.ndarray, # shape (num_joints,)
                          current_velocities: np.ndarray, # shape (num_joints,) - Re-added!
                          previous_torques: np.ndarray, # shape (num_joints,)
                          dt: float # Simulation timestep
                          ) -> np.ndarray:
        """
        根据控制信号和当前状态计算下一个时间步的关节力矩。
        """
        # 1. 将控制信号映射到期望力矩
        mid_control = (self.control_max + self.control_min) / 2.0
        desired_torques = (control_signals - mid_control) * self.control_to_torque_ratio
        desired_torques = np.clip(desired_torques, -self.max_torques, self.max_torques)

        # 2. 根据控制信号估算流量需求
        flow_demands = np.abs(control_signals) * self.flow_demand_factors
        total_flow_demand = np.sum(flow_demands)

        # 3. 如果流量超限，计算力矩缩放因子
        torque_scale_factor = 1.0
        if total_flow_demand > self.pump_max_flow and total_flow_demand > 1e-6:
            torque_scale_factor = self.pump_max_flow / total_flow_demand

        # 4. 计算可达目标力矩 (根据控制信号是否为零区分处理)
        achievable_target_torques = np.zeros_like(desired_torques)
        is_control_zero = np.abs(control_signals) < self.zero_control_threshold

        # 对控制信号不为零的关节，计算流量限制后的期望力矩
        non_zero_mask = ~is_control_zero
        achievable_target_torques[non_zero_mask] = desired_torques[non_zero_mask] * torque_scale_factor

        # 对控制信号为零的关节，计算制动力矩
        zero_mask = is_control_zero
        braking_torques = -self.braking_damping_coeff[zero_mask] * current_velocities[zero_mask]
        # 限制制动力矩不超过最大力矩
        achievable_target_torques[zero_mask] = np.clip(braking_torques, -self.max_torques[zero_mask], self.max_torques[zero_mask])

        # 5. 应用力矩变化率限制 (模拟响应延迟)
        max_torque_change = self.max_torque_rate * dt
        torque_diff = achievable_target_torques - previous_torques
        clamped_diff = np.clip(torque_diff, -max_torque_change, max_torque_change)
        next_torques = previous_torques + clamped_diff

        # 6. 应用最终的绝对最大力矩限制
        next_torques = np.clip(next_torques, -self.max_torques, self.max_torques)

        return next_torques

