# -*- coding: utf-8 -*-
import numpy as np
import argparse
import random
import os

# --- 修改: 结合动作、平滑和保持时间生成控制信号 ---
def generate_action_based_controls(num_steps, dt, num_joints=4,
                                   min_hold_time=0.5, max_hold_time=3.0,
                                   pause_time=0.2,
                                   smooth_alpha=0.1, # 平滑系数 (0 < alpha <= 1)
                                   control_range=(-1000.0, 1000.0)):
    """
    生成基于“动作”的平滑随机控制信号。
    每个动作随机选择关节，设定目标控制量，并保持一段时间。
    动作之间有停顿。控制量的变化通过指数平滑进行。

    Args:
        num_steps (int): 总的仿真步数。
        dt (float): 仿真时间步长 (秒)。
        num_joints (int): 控制的关节数量 (默认为 4)。
        min_hold_time (float): 一个动作的目标控制量保持的最短时间 (秒)。
        max_hold_time (float): 一个动作的目标控制量保持的最长时间 (秒)。
        pause_time (float): 动作之间的停顿时间 (秒)。
        smooth_alpha (float): 指数平滑系数 alpha (0 < alpha <= 1)。值越小越平滑。
        control_range (tuple): 控制信号的范围 (最小值, 最大值)。

    Returns:
        np.ndarray: 控制信号数组，形状为 (num_steps, num_joints)。
    """
    controls = np.zeros((num_steps, num_joints))
    current_smooth_controls = np.zeros(num_joints) # 滤波器的当前状态
    current_step = 0
    min_val, max_val = control_range
    pause_steps = max(1, int(round(pause_time / dt)))

    print(f"Generating {num_steps} steps of action-based smooth controls (dt={dt:.3f}s)...")
    print(f"Action hold time range: [{min_hold_time:.2f}s, {max_hold_time:.2f}s]")
    print(f"Pause time: {pause_time:.2f}s ({pause_steps} steps)")
    print(f"Smooth alpha: {smooth_alpha:.3f}")
    print(f"Control value range: [{min_val:.1f}, {max_val:.1f}]")

    if not (0 < smooth_alpha <= 1.0):
        raise ValueError("smooth_alpha must be between 0 (exclusive) and 1 (inclusive)")

    while current_step < num_steps:
        # --- Action Phase ---
        # 1. 决定本次动作驱动几个关节
        num_active_joints = random.randint(1, num_joints)
        active_indices = random.sample(range(num_joints), num_active_joints)

        # 2. 为活动关节生成目标控制量，非活动关节目标为0
        target_controls = np.zeros(num_joints)
        for idx in active_indices:
            target_controls[idx] = random.uniform(min_val, max_val)

        # 3. 决定本次动作持续时间
        hold_duration = random.uniform(min_hold_time, max_hold_time)
        action_steps = max(1, int(round(hold_duration / dt)))
        action_end_step = min(current_step + action_steps, num_steps)

        # print(f"Step {current_step}: Action - Joints {active_indices}, Targets {target_controls[active_indices]}, Steps {action_end_step - current_step}") # Debug

        # 4. 在动作持续时间内，平滑地趋近目标控制量
        for step in range(current_step, action_end_step):
            current_smooth_controls = (smooth_alpha * target_controls +
                                       (1.0 - smooth_alpha) * current_smooth_controls)
            controls[step, :] = current_smooth_controls
        current_step = action_end_step

        # --- Pause Phase ---
        if current_step < num_steps:
            pause_end_step = min(current_step + pause_steps, num_steps)
            target_controls = np.zeros(num_joints) # 停顿期间目标为0

            # print(f"Step {current_step}: Pause - Steps {pause_end_step - current_step}") # Debug

            # 5. 在停顿时间内，平滑地趋近于 0
            for step in range(current_step, pause_end_step):
                current_smooth_controls = (smooth_alpha * target_controls +
                                           (1.0 - smooth_alpha) * current_smooth_controls)
                controls[step, :] = current_smooth_controls
            current_step = pause_end_step

        # 简单的进度显示
        if 'last_print_percent' not in locals(): locals()['last_print_percent'] = -1
        percent_done = (current_step / num_steps) * 100
        if percent_done // 10 > locals()['last_print_percent'] // 10:
             print(f"  Generated {current_step}/{num_steps} steps ({percent_done:.0f}%)...")
             locals()['last_print_percent'] = percent_done


    print(f"Finished generating {num_steps} steps.")
    return controls
# --- 结束修改 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate action-based smooth random control signals.') # Updated description
    parser.add_argument('--duration', type=float, required=True, help='Total duration of the control sequence in seconds')
    parser.add_argument('--dt', type=float, default=0.02, help='Simulation timestep in seconds (must match simulator)')
    parser.add_argument('--output_file', type=str, default='action_smooth_random_controls.npz', help='Output .npz file name') # Changed default name
    # --- 恢复 hold time 参数 ---
    parser.add_argument('--min_hold', type=float, default=0.5, help='Minimum time (s) for an action phase')
    parser.add_argument('--max_hold', type=float, default=3.0, help='Maximum time (s) for an action phase')
    # --- 恢复/添加 smooth_alpha 参数 ---
    parser.add_argument('--smooth_alpha', type=float, default=0.1, help='Smoothing factor alpha (0 < alpha <= 1). Smaller values mean smoother/slower transitions.')
    # --- 新增 pause_time 参数 ---
    parser.add_argument('--pause_time', type=float, default=0.2, help='Pause time (s) between actions')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed for reproducibility')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    if args.min_hold <= 0 or args.max_hold <= 0 or args.min_hold > args.max_hold:
        print("Error: Invalid hold time arguments.")
        exit()
    if args.pause_time < 0:
         print("Error: Pause time cannot be negative.")
         exit()
    if not (0 < args.smooth_alpha <= 1.0):
        print("Error: --smooth_alpha must be between 0 (exclusive) and 1 (inclusive).")
        exit()

    num_steps = int(round(args.duration / args.dt))
    if num_steps <= 0:
        print("Error: Duration and dt result in zero or negative steps.")
        exit()

    # --- 调用新的生成函数 ---
    random_controls = generate_action_based_controls(
        num_steps=num_steps,
        dt=args.dt,
        min_hold_time=args.min_hold,
        max_hold_time=args.max_hold,
        pause_time=args.pause_time,
        smooth_alpha=args.smooth_alpha
    )
    # --- 结束调用 ---

    # 保存到 .npz 文件
    try:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        np.savez(args.output_file, control_signal=random_controls) # 仍然使用 'control_signal' 键名
        print(f"Action-based smooth random control signals saved to {args.output_file}")
        print(f"  Shape: {random_controls.shape}")
    except Exception as e:
        print(f"Error saving data to {args.output_file}: {e}")

