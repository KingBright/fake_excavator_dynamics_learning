# -*- coding: utf-8 -*-
import mujoco
import numpy as np
import os
import argparse
import math # For exp
import joblib
from sklearn.preprocessing import StandardScaler
import pygame # For gamepad
from tqdm import tqdm
import time
import glfw # For MuJoCo Viewer window
import matplotlib.pyplot as plt

# --- Import the velocity coordinator ---
try:
    # Assumes velocity_coordination.py contains VelocityCoordinator class
    # (with oscillation fix, WITHOUT min_velocity_for_oscillation param)
    from velocity_coordination import VelocityCoordinator
except ImportError:
    print("Warning: velocity_coordination.py not found. Using placeholder.")
    class VelocityCoordinator: # Placeholder
        def __init__(self, *args, **kwargs): self.num_joints = 4; print("Using Placeholder VelocityCoordinator!")
        def calculate_target_velocities(self, *args, **kwargs): return np.zeros(self.num_joints)


# --- 控制映射字典 ---
DEFAULT_GAMEPAD_MAPPING = {
    'cab':    {'type': 'axis', 'index': 1, 'scale': 1000, 'comment': 'Left Stick Y: Up=CCW, Down=CW'},
    'arm':    {'type': 'axis', 'index': 0, 'scale': 1000, 'comment': 'Left Stick X: Left=Out, Right=In'},
    'boom':   {'type': 'axis', 'index': 3, 'scale': -1000, 'comment': 'Right Stick Y: Up=Lift, Down=Lower'},
    'bucket': {'type': 'axis', 'index': 2, 'scale': -1000, 'comment': 'Right Stick X: Left=Curl, Right=Dump'},
}
# --- 结束控制映射 ---

# --- 计算 Effort Factors 的函数 ---
def calculate_effort_factors(model: mujoco.MjModel) -> np.ndarray:
    """
    根据模型中的质量和质心信息估算各关节的努力度因子。
    """
    # (函数内容保持不变)
    try:
        body_names = ['cab', 'boom', 'arm', 'bucket']
        body_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names}
        if any(id < 0 for id in body_ids.values()):
            print(f"Warning: Could not find all bodies for effort calculation: {body_ids}")
            return np.ones(4)

        m_cab = model.body_mass[body_ids['cab']]
        m_boom = model.body_mass[body_ids['boom']]
        m_arm = model.body_mass[body_ids['arm']]
        m_bucket = model.body_mass[body_ids['bucket']]

        com_cab = model.body_ipos[body_ids['cab']]
        com_boom = model.body_ipos[body_ids['boom']]
        com_arm = model.body_ipos[body_ids['arm']]
        com_bucket = model.body_ipos[body_ids['bucket']]

        izz_cab = model.body_inertia[body_ids['cab']][2]

        effort_cab = izz_cab
        effort_boom = (m_boom + m_arm + m_bucket) * np.linalg.norm(com_boom)
        effort_arm = (m_arm + m_bucket) * np.linalg.norm(com_arm)
        effort_bucket = m_bucket * np.linalg.norm(com_bucket)

        factors = np.array([effort_cab, effort_boom, effort_arm, effort_bucket])
        factors = np.maximum(factors, 1e-6)
        normalized_factors = factors / np.max(factors)

        print(f"Calculated Raw Effort Factors: Cab={factors[0]:.1f}, Boom={factors[1]:.1f}, Arm={factors[2]:.1f}, Bucket={factors[3]:.1f}")
        print(f"Normalized Effort Factors: {normalized_factors}")
        return normalized_factors

    except Exception as e:
        print(f"Error calculating effort factors: {e}")
        return np.ones(4)
# --- 结束 Effort Factors 计算函数 ---


def get_gamepad_input(joystick, mapping):
    """读取手柄输入，应用死区和分段线性映射，返回控制信号 [-1000, 1000]。"""
    # (函数内容保持不变)
    if joystick is None or not pygame.joystick.get_init() or not joystick.get_init():
         return np.zeros(4)
    try:
        pygame.event.pump()
        controls = np.zeros(4)
        joint_names = ['cab', 'boom', 'arm', 'bucket']
        deadzone = 0.2
        for i, name in enumerate(joint_names):
            map_info = mapping.get(name)
            if map_info and map_info['type'] == 'axis':
                axis_val = joystick.get_axis(map_info['index'])
                control_val = 0.0
                scale_abs = abs(map_info.get('scale', 1000.0))
                if axis_val > deadzone:
                    control_val = ((axis_val - deadzone) / (1.0 - deadzone)) * scale_abs
                elif axis_val < -deadzone:
                    control_val = ((axis_val + deadzone) / (1.0 - deadzone)) * scale_abs
                if map_info.get('scale', 1000.0) < 0:
                    control_val *= -1.0
                controls[i] = np.clip(control_val, -scale_abs, scale_abs)
    except pygame.error as e:
        print(f"Warn: Pygame error reading joystick: {e}")
        return np.zeros(4)
    return controls


def load_scripted_controls(filepath):
    """从 .npz 文件加载控制信号"""
    # (函数内容保持不变)
    try:
        data = np.load(filepath)
        controls = data['control_signal']
        original_qpos = data.get('qpos')
        original_qvel = data.get('qvel')
        original_set_vel = data.get('set_velocity')
        if 'target_velocity' in data and original_set_vel is None:
             original_set_vel = data['target_velocity']
        if controls.ndim != 2 or controls.shape[1] != 4: raise ValueError(f"Invalid shape for controls: {controls.shape}")
        print(f"Loaded replay data from {filepath}, shape: {controls.shape}")
        return controls, original_qpos, original_qvel, original_set_vel
    except Exception as e:
        print(f"Error loading replay data: {e}")
        return None, None, None, None

# --- MuJoCo Viewer 全局变量 ---
button_left = False; button_middle = False; button_right = False
lastx = 0; lasty = 0; paused = False; slow_motion_factor = 1.0
mj_model = None; mj_data = None; cam = None; scene = None; opt = None; context = None
active_control_mode = 'gamepad'; has_script_file = False; joystick = None

# --- GLFW 回调函数 (keyboard, mouse_button, mouse_move, scroll) ---
# (保持不变)
def keyboard(window, key, scancode, act, mods):
    global paused, slow_motion_factor, mj_model, mj_data, active_control_mode, has_script_file, joystick
    if active_control_mode == 'replay' and key in [glfw.KEY_G, glfw.KEY_S, glfw.KEY_Z]:
        print("Cannot switch mode during replay.")
        return
    if act == glfw.PRESS:
        if key == glfw.KEY_SPACE: paused = not paused
        elif key == glfw.KEY_LEFT_BRACKET: slow_motion_factor = max(0.1, slow_motion_factor * 0.5); print(f"Slow motion factor: {slow_motion_factor}")
        elif key == glfw.KEY_RIGHT_BRACKET: slow_motion_factor = min(10.0, slow_motion_factor * 2.0); print(f"Slow motion factor: {slow_motion_factor}")
        elif key == glfw.KEY_BACKSPACE:
            if mj_model is not None and mj_data is not None:
                 mujoco.mj_resetData(mj_model, mj_data); mujoco.mj_forward(mj_model, mj_data); print("Simulation reset.")
            else: print("Cannot reset: model or data not available.")
        elif key == glfw.KEY_G:
            print("Switching to Gamepad control mode.")
            active_control_mode = 'gamepad'
            if joystick is None and pygame.joystick.get_count() > 0:
                 try: joystick = pygame.joystick.Joystick(0); joystick.init(); print("Gamepad initialized on switch.")
                 except pygame.error as e: print(f"Error initializing gamepad on switch: {e}"); joystick = None
        elif key == glfw.KEY_S:
            if has_script_file: print("Switching to Scripted control mode."); active_control_mode = 'scripted'
            else: print("Cannot switch to Scripted mode: No script file provided via --script_path.")
        elif key == glfw.KEY_Z: print("Switching to Zero control mode."); active_control_mode = 'zero'
def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right, lastx, lasty
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    lastx, lasty = glfw.get_cursor_pos(window)
def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right, mj_model, mj_data, cam, scene
    if mj_model is None or mj_data is None or cam is None or scene is None: return
    if not (button_left or button_middle or button_right): return
    dx = xpos - lastx; dy = ypos - lasty; lastx = xpos; lasty = ypos
    width, height = glfw.get_window_size(window)
    if height == 0: return
    mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
    action = None
    if button_right: action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
    elif button_left: action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    else: action = mujoco.mjtMouse.mjMOUSE_ZOOM
    mujoco.mjv_moveCamera(mj_model, action, dx/height, dy/height, scene, cam)
def scroll(window, xoffset, yoffset):
    global mj_model, scene, cam
    if mj_model is None or scene is None or cam is None: return
    mujoco.mjv_moveCamera(mj_model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scene, cam)
# --- 结束回调函数 ---


# --- 主模拟函数 (修正函数签名和 VelocityCoordinator 初始化调用) ---
def simulate_excavator_with_viewer(xml_path, duration=30.0, initial_control_mode='gamepad',
                                   script_path=None, replay_file=None,
                                   gamepad_mapping=DEFAULT_GAMEPAD_MAPPING,
                                   # --- 速度协调器参数 ---
                                   max_velocities = [0.5, 0.4, 0.6, 0.8],
                                   effort_factors = [1.0, 1.0, 1.0, 1.0],
                                   max_total_effort = 1.5,
                                   accel_response_time = 0.1,
                                   decel_response_time = 0.2,
                                   # --- 振荡参数 ---
                                   oscillation_frequency = 15.0,
                                   oscillation_decay_rate = 5.0,
                                   initial_amplitude_scale = 0.3,
                                   oscillation_stop_threshold = 0.01,
                                   # min_velocity_for_oscillation = 0.05, # <<< 彻底移除此参数
                                   zero_control_threshold = 1.0,
                                   # --- 其他参数 ---
                                   real_time_sim=False,
                                   no_display=False,
                                   output_filename="excavator_osc_delay_data.npz",
                                   plot_comparison=True
                                   ):
    """
    Simulates the excavator using VelocityCoordinator with asymmetric delay and stopping oscillation,
    and DIRECTLY SETTING qvel. Includes joint limit handling.
    WARNING: Physically unrealistic.
    """
    global mj_model, mj_data, cam, scene, opt, context, active_control_mode, has_script_file, joystick

    # --- Set initial control mode ---
    active_control_mode = initial_control_mode
    has_script_file = script_path is not None and os.path.exists(script_path)

    # --- Load Replay Data if in Replay Mode ---
    replay_control_signals = None
    original_qpos = None
    original_qvel = None
    original_set_vel = None
    if active_control_mode == 'replay':
        if replay_file is None or not os.path.exists(replay_file):
            print(f"Error: Replay mode selected but replay file '{replay_file}' not found or invalid.")
            return None
        print(f"--- Replay Mode Activated: Loading data from {replay_file} ---")
        replay_control_signals, original_qpos, original_qvel, original_set_vel = load_scripted_controls(replay_file)
        if replay_control_signals is None:
            print("Failed to load replay data.")
            return None
        # Estimate duration based on replay data length (will be recalculated with actual dt)
        duration = len(replay_control_signals) * 0.02 # Estimate using default dt first
        print(f"Replay duration estimated to {duration:.2f}s based on loaded data.")

    # --- Load the Model ---
    try:
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
        dt = mj_model.opt.timestep # Get actual dt
        if active_control_mode == 'replay': # Recalculate duration with actual dt
             duration = len(replay_control_signals) * dt
             print(f"Replay duration recalculated to {duration:.2f}s with actual dt={dt:.4f}s.")
        print(f"Model loaded: {xml_path} (dt={dt:.4f}s)")
        if mj_model.nu > 0:
             print("Warning: Actuators found in XML, but they will be ignored by this script.")
    except Exception as e: print(f"Error loading model: {e}"); return None

    # --- 计算 Effort Factors ---
    effort_factors_calculated = calculate_effort_factors(mj_model)

    # --- 获取关节 DoF 和 QPos 索引 ---
    controlled_joint_names = ['cab', 'boom', 'arm', 'bucket']
    try:
        num_controlled_joints = len(controlled_joint_names)
        joint_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in controlled_joint_names]
        if any(idx < 0 for idx in joint_ids): raise ValueError(f"Joints not found: {[n for n, i in zip(controlled_joint_names, joint_ids) if i < 0]}")
        dof_indices = [mj_model.jnt_dofadr[jid] for jid in joint_ids]
        qpos_indices = [mj_model.jnt_qposadr[jid] for jid in joint_ids]
        print(f"Controlled DoF indices (for qvel): {dof_indices}")
        print(f"Controlled QPos indices (for qpos): {qpos_indices}")
        if not (len(max_velocities) == len(effort_factors_calculated) == num_controlled_joints):
             raise ValueError("Length of max_velocities must match number of controlled joints")
    except Exception as e: print(f"Error finding joint indices: {e}"); return None


    # --- 初始化速度协调模块 (修正: 移除 min_velocity_for_oscillation) ---
    velocity_coordinator = VelocityCoordinator(
        num_joints=num_controlled_joints,
        dt=dt,
        max_velocities=max_velocities,
        effort_factors=effort_factors_calculated,
        max_total_effort=max_total_effort,
        accel_response_time=accel_response_time,
        decel_response_time=decel_response_time,
        oscillation_frequency=oscillation_frequency,
        oscillation_decay_rate=oscillation_decay_rate,
        initial_amplitude_scale=initial_amplitude_scale,
        oscillation_stop_threshold=oscillation_stop_threshold,
        # min_velocity_for_oscillation=min_velocity_for_oscillation, # <<< 彻底移除此行
        zero_control_threshold=zero_control_threshold
    )
    print(f"Using Velocity Coordinator (Asymmetric Delay + Oscillation - No Min Vel Trigger):")
    # (省略打印信息)

    # --- Setup Control Input (excluding replay) ---
    # (Control setup logic remains the same)
    joystick = None; scripted_controls = None; script_step = 0
    if initial_control_mode == 'gamepad':
        try:
            print("Initializing Pygame for potential gamepad use...")
            pygame.init(); pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                joystick = pygame.joystick.Joystick(0); joystick.init()
                print(f"Gamepad available: {joystick.get_name()}")
            else: print("No gamepad detected at start.")
        except Exception as e: print(f"Error initializing pygame: {e}")
    if script_path and os.path.exists(script_path):
        if active_control_mode != 'replay':
            scripted_controls, _, _, _ = load_scripted_controls(script_path)
            if scripted_controls is None: return None
            script_duration = scripted_controls.shape[0] * dt
            if duration > script_duration: print(f"Warning: Duration longer than script. Simulating for {script_duration:.2f}s."); duration = script_duration
    elif initial_control_mode == 'scripted' and active_control_mode != 'replay':
         print(f"Error: Control mode 'scripted' requested but script file '{script_path}' not found or invalid.")
         return None


    # --- Initialize MuJoCo Visualization ---
    # (Visualization init remains the same)
    if not no_display:
        print("Initializing GLFW for visualization...")
        if not glfw.init(): print("Could not initialize GLFW"); return None
        window = glfw.create_window(1200, 900, f"Excavator Simulation ({active_control_mode.upper()})", None, None)
        if not window: glfw.terminate(); print("Could not create GLFW window"); return None
        glfw.make_context_current(window); glfw.swap_interval(1)
        cam = mujoco.MjvCamera(); opt = mujoco.MjvOption()
        scene = mujoco.MjvScene(mj_model, maxgeom=10000)
        context = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)
        glfw.set_key_callback(window, keyboard); glfw.set_cursor_pos_callback(window, mouse_move)
        glfw.set_mouse_button_callback(window, mouse_button); glfw.set_scroll_callback(window, scroll)
        mujoco.mjv_defaultCamera(cam); cam.azimuth = 90; cam.elevation = -15
        cam.distance = 25.0; cam.lookat = np.array([2.0, 0, 1.5])
        print(f"Initial camera set: distance={cam.distance}, azimuth={cam.azimuth}, elevation={cam.elevation}")


    # --- Simulation Setup ---
    times, qpos_data, qvel_data, set_vel_data, control_signal_data = [], [], [], [], []
    mjstep_times = []

    # --- Simulation Loop with Viewer ---
    print(f"Starting simulation: duration={duration:.2f}s, initial_mode='{active_control_mode}', real_time={real_time_sim}")
    if active_control_mode == 'replay':
         print("--- REPLAYING RECORDED CONTROL SIGNALS ---")
    else:
         print(">>> WARNING: Directly setting joint velocities - PHYSICALLY UNREALISTIC! <<<")
         if not no_display:
            print("Press [Space] to pause/resume. Press [G] Gamepad, [S] Scripted, [Z] Zero to switch modes.")
    if not no_display:
        print("Press [ or ] to change speed.")

    current_ctrl_signal = np.zeros(num_controlled_joints)
    target_velocities = np.zeros(num_controlled_joints)
    limit_epsilon = 1e-5

    while no_display or not glfw.window_should_close(window):
        sim_loop_start_time = time.perf_counter()
        step_time_ms = 0.0

        sim_continue = (mj_data.time < duration - dt/2)
        if active_control_mode == 'replay':
            sim_continue = script_step < len(replay_control_signals)

        if not paused and sim_continue:
            step_start_time = time.perf_counter()

            # --- Get Control Signal ---
            # (Control signal logic remains the same)
            if active_control_mode == 'gamepad':
                if joystick is None or not joystick.get_init():
                    if pygame.joystick.get_count() > 0:
                        try: joystick = pygame.joystick.Joystick(0); joystick.init(); print("Gamepad (re)connected.")
                        except pygame.error: joystick = None; current_ctrl_signal = np.zeros(num_controlled_joints)
                        else: current_ctrl_signal = get_gamepad_input(joystick, gamepad_mapping)
                    else: current_ctrl_signal = np.zeros(num_controlled_joints)
                else: current_ctrl_signal = get_gamepad_input(joystick, gamepad_mapping)
            elif active_control_mode == 'scripted':
                if scripted_controls is not None and script_step < len(scripted_controls):
                    current_ctrl_signal = scripted_controls[script_step]; script_step += 1
                else:
                    current_ctrl_signal = np.zeros(num_controlled_joints)
                    if scripted_controls is not None and script_step == len(scripted_controls): print("Script finished."); script_step += 1
            elif active_control_mode == 'replay':
                 current_ctrl_signal = replay_control_signals[script_step]; script_step += 1
            else: current_ctrl_signal = np.zeros(num_controlled_joints)

            # --- Calculate Target Velocities using Coordinator ---
            target_velocities = velocity_coordinator.calculate_target_velocities(current_ctrl_signal)
            # --- End Velocity Calculation ---

            # --- 关节限位处理 ---
            final_velocities_to_set = target_velocities.copy()
            for i in range(num_controlled_joints):
                joint_name = controlled_joint_names[i]
                if joint_name == 'cab': continue
                joint_id = joint_ids[i]; qpos_idx = qpos_indices[i]
                if mj_model.jnt_limited[joint_id]:
                    lower_limit, upper_limit = mj_model.jnt_range[joint_id]
                    current_pos = mj_data.qpos[qpos_idx]
                    calculated_vel = target_velocities[i]
                    if current_pos <= lower_limit + limit_epsilon and calculated_vel < 0:
                        final_velocities_to_set[i] = 0.0
                    elif current_pos >= upper_limit - limit_epsilon and calculated_vel > 0:
                        final_velocities_to_set[i] = 0.0
            # --- 结束关节限位处理 ---


            # --- Directly Set Joint Velocities (Bypass Physics) ---
            mj_data.qvel[dof_indices] = final_velocities_to_set
            # --- End Direct Setting ---

            # --- Clear other forces/controls ---
            mj_data.qfrc_applied[:] = 0
            mj_data.ctrl[:] = 0

            # Step Simulation
            t_mjstep_start = time.perf_counter()
            try: mujoco.mj_step(mj_model, mj_data)
            except Exception as e: print(f"Error during mj_step: {e}"); break
            t_mjstep_end = time.perf_counter()
            step_time_ms = (t_mjstep_end - t_mjstep_start) * 1000
            mjstep_times.append(t_mjstep_end - t_mjstep_start)

            # Record Data
            times.append(mj_data.time); qpos_data.append(mj_data.qpos.copy()); qvel_data.append(mj_data.qvel.copy())
            set_vel_data.append(final_velocities_to_set.copy())
            control_signal_data.append(current_ctrl_signal.copy())


        # --- Visualization ---
        # (Visualization code remains the same)
        if not no_display:
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            glfw.get_framebuffer_size(window)
            viewport.width, viewport.height = glfw.get_framebuffer_size(window)
            mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            # Overlays
            status = "PAUSED" if paused else f"Speed: {1.0/slow_motion_factor:.1f}x"
            mode_str = active_control_mode.upper()
            if active_control_mode != 'replay': mode_str += " (G/S/Z)"
            top_left_text = (f"Time: {mj_data.time:.2f} s\n"
                            f"Mode: {mode_str}\n"
                            f"{status}\n"
                            f"mj_step: {step_time_ms:.1f} ms")
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, top_left_text, None, context)
            current_qpos_controlled = mj_data.qpos[qpos_indices]
            current_qvel_controlled = mj_data.qvel[dof_indices]
            qpos_texts = [f"Pos_{name}: {np.rad2deg(pos): 6.1f} deg" for name, pos in zip(controlled_joint_names, current_qpos_controlled)]
            qvel_texts = [f"ActVel_{name}: {vel: 6.2f}" for name, vel in zip(controlled_joint_names, current_qvel_controlled)]
            bottom_left_text = "Joint State\n" + "\n".join(qpos_texts) + "\n" + "\n".join(qvel_texts)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, bottom_left_text, None, context)
            ctrl_texts = [f"CtrlSig_{name}: {sig: 6.0f}" for name, sig in zip(controlled_joint_names, current_ctrl_signal)]
            set_vel_texts = [f"SetVel_{name}: {tv: 6.2f}" for name, tv in zip(controlled_joint_names, final_velocities_to_set)]
            bottom_right_text = ("Control Signal\n" + "\n".join(ctrl_texts) + "\n\n" +
                                "Set Velocity (rad/s)\n" + "\n".join(set_vel_texts) )
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, viewport, bottom_right_text, None, context)

            glfw.swap_buffers(window)
            glfw.poll_events()

        # Real-time Speed Control
        if real_time_sim and not paused:
            sim_loop_end_time = time.perf_counter()
            time_elapsed = sim_loop_end_time - sim_loop_start_time
            time_to_sleep = (dt / slow_motion_factor) - time_elapsed
            if time_to_sleep > 0: time.sleep(time_to_sleep)

        if not sim_continue and not paused:
            if active_control_mode == 'scripted': print("Script finished."); break

            if active_control_mode == 'replay': print("Replay finished."); break
            pass # Keep window open

    # --- Cleanup & Reporting ---
    if not no_display:
        glfw.terminate()
    if joystick: joystick.quit()
    pygame.quit()
    print("Simulation loop finished.")
    if mjstep_times: print(f"Avg mj_step Time: {np.mean(mjstep_times)*1000:.3f} ms")

    # --- Convert lists to arrays ---
    times = np.array(times); qpos_data = np.array(qpos_data); qvel_data = np.array(qvel_data)
    set_vel_data = np.array(set_vel_data); control_signal_data = np.array(control_signal_data)

    # --- Save Data ---
    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        np.savez(output_filename, time=times, qpos=qpos_data, qvel=qvel_data,
                 set_velocity=set_vel_data, control_signal=control_signal_data)
        print(f"Simulation data saved to {output_filename}")
    except Exception as e: print(f"Error saving data: {e}")

    # --- Replay 一致性比较 ---
    if active_control_mode == 'replay' and original_qpos is not None and original_qvel is not None:
        print("\n--- Replay Consistency Check ---")
        min_len = min(len(qpos_data), len(original_qpos))
        if min_len == 0: print("No data recorded during replay for comparison.")
        else:
            qpos_replay_comp = qpos_data[:min_len]; qvel_replay_comp = qvel_data[:min_len]
            qpos_orig_comp = original_qpos[:min_len]; qvel_orig_comp = original_qvel[:min_len]
            qpos_mae = np.mean(np.abs(qpos_replay_comp[:, qpos_indices] - qpos_orig_comp[:, qpos_indices]))
            qvel_mae = np.mean(np.abs(qvel_replay_comp[:, dof_indices] - qvel_orig_comp[:, dof_indices]))
            qpos_rmse = np.sqrt(np.mean((qpos_replay_comp[:, qpos_indices] - qpos_orig_comp[:, qpos_indices])**2))
            qvel_rmse = np.sqrt(np.mean((qvel_replay_comp[:, dof_indices] - qvel_orig_comp[:, dof_indices])**2))
            print(f"Compared {min_len} steps."); print(f"Qpos MAE: {qpos_mae:.6f} rad"); print(f"Qvel MAE: {qvel_mae:.6f} rad/s"); print(f"Qpos RMSE: {qpos_rmse:.6f} rad"); print(f"Qvel RMSE: {qvel_rmse:.6f} rad/s")
            if qpos_mae > 1e-4 or qvel_mae > 1e-3: print("Warning: Significant difference detected.")
            else: print("Consistency check passed.")
            if plot_comparison:
                print("Generating comparison plot...")
                fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True); fig.suptitle(f'Replay vs Original Trajectory\nFile: {os.path.basename(replay_file)}')
                time_axis_comp = times[:min_len]
                for i in range(num_controlled_joints):
                    joint_name = controlled_joint_names[i]
                    axes[i, 0].plot(time_axis_comp, np.rad2deg(qpos_orig_comp[:, qpos_indices[i]]), label='Original', color='blue', alpha=0.7); axes[i, 0].plot(time_axis_comp, np.rad2deg(qpos_replay_comp[:, qpos_indices[i]]), label='Replay', color='red', linestyle='--', alpha=0.7); axes[i, 0].set_ylabel(f'{joint_name} Pos (deg)'); axes[i, 0].legend(); axes[i, 0].grid(True)
                    axes[i, 1].plot(time_axis_comp, qvel_orig_comp[:, dof_indices[i]], label='Original', color='blue', alpha=0.7); axes[i, 1].plot(time_axis_comp, qvel_replay_comp[:, dof_indices[i]], label='Replay', color='red', linestyle='--', alpha=0.7); axes[i, 1].set_ylabel(f'{joint_name} Vel (rad/s)'); axes[i, 1].legend(); axes[i, 1].grid(True)
                axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)'); plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plot_filename = output_filename.replace(".npz", "_comparison.png"); plt.savefig(plot_filename); print(f"Comparison plot saved to {plot_filename}"); plt.close(fig)
    # --- 结束 Replay 比较 ---

    # --- Return recorded data ---
    return times, qpos_data, qvel_data, set_vel_data, control_signal_data


# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description='Simulate Excavator by Directly Setting Joint Velocities using VelocityCoordinator (with Asymmetric Delay + Oscillation + Limit Handling + Replay)')
    parser.add_argument('--xml', type=str, default="excavator_modified.xml", help='Path to the MuJoCo XML model file (Actuators ignored)')
    parser.add_argument('--duration', type=float, default=60.0, help='Simulation duration in seconds (ignored in replay mode)')
    parser.add_argument('--initial_control_mode', type=str, default='gamepad', choices=['gamepad', 'scripted', 'zero', 'replay'], help='Initial control input source')
    parser.add_argument('--script_path', type=str, default=None, help='Path to .npz file containing "controls" array (required if starting/switching to scripted mode)')
    parser.add_argument('--replay_file', type=str, default=None, help='Path to .npz file to replay (required for replay mode)')
    parser.add_argument('--output', type=str, default="excavator_direct_vel_osc_delay_data.npz", help='Output .npz file name')
    parser.add_argument('--real_time', action='store_true', default=False, help='Attempt to run simulation in real-time')
    parser.add_argument('--max_vel', type=float, nargs=4, default=[0.4, 0.3, 0.4, 0.6], metavar=('V_CAB', 'V_BOOM', 'V_ARM', 'V_BUCKET'), help='Max target velocities (rad/s) for joints')
    parser.add_argument('--max_effort', type=float, default=1.5, help='Max total effort for velocity coordination')
    parser.add_argument('--accel_time', type=float, default=0.1, help='Velocity acceleration response time constant (tau) in seconds')
    parser.add_argument('--decel_time', type=float, default=0.2, help='Velocity deceleration response time constant (tau) in seconds')
    parser.add_argument('--osc_freq', type=float, default=15.0, help='Stopping oscillation frequency (rad/s)')
    parser.add_argument('--osc_decay', type=float, default=5.0, help='Stopping oscillation decay rate (1/s)')
    parser.add_argument('--osc_scale', type=float, default=0.3, help='Stopping oscillation initial amplitude scale (relative to stopping velocity)')
    parser.add_argument('--no_plot', action='store_true', default=False, help='Disable comparison plot generation in replay mode')
    parser.add_argument('--no_display', action='store_true', default=False, help='Disable gui display')

    # Removed min_velocity_for_oscillation arg
    args = parser.parse_args()
    # --- End Argument Parser ---

    # --- Run Simulation ---
    if not os.path.exists(args.xml): print(f"Error: XML file not found at '{args.xml}'")
    else:
        if args.initial_control_mode == 'replay' and args.replay_file is None:
             print("Error: --initial_control_mode is 'replay' but --replay_file was not provided.")
             exit()
        if args.initial_control_mode == 'scripted' and (args.script_path is None or not os.path.exists(args.script_path)):
             print("Error: --initial_control_mode is 'scripted' but --script_path is missing or invalid.")
             exit()

        # --- Load model once to calculate effort factors ---
        try:
            temp_model_for_effort = mujoco.MjModel.from_xml_path(args.xml)
            effort_factors_calculated = calculate_effort_factors(temp_model_for_effort)
            del temp_model_for_effort
        except Exception as e:
            print(f"Failed to calculate effort factors from model: {e}")
            effort_factors_calculated = np.ones(4)

        # --- 设置速度协调器参数 ---
        velocity_coord_params_for_sim = {
            "max_velocities": args.max_vel,
            "effort_factors": effort_factors_calculated.tolist(),
            "max_total_effort": args.max_effort,
            "accel_response_time": args.accel_time,
            "decel_response_time": args.decel_time,
            "oscillation_frequency": args.osc_freq,
            "oscillation_decay_rate": args.osc_decay,
            "initial_amplitude_scale": args.osc_scale,
            # Removed min_velocity_for_oscillation
        }
        print(f"Using Velocity Coordinator Params: {velocity_coord_params_for_sim}")
        # --- 结束参数设置 ---

        # Call the main simulation function
        sim_data_tuple = simulate_excavator_with_viewer(
            xml_path=args.xml,
            duration=args.duration,
            initial_control_mode=args.initial_control_mode,
            script_path=args.script_path,
            replay_file=args.replay_file, # Pass replay file path
            real_time_sim=args.real_time,
            no_display=args.no_display, # Pass no_display paramete
            output_filename=args.output,
            plot_comparison=(not args.no_plot), # Pass plot flag
            # Pass parameters explicitly
            max_velocities=velocity_coord_params_for_sim["max_velocities"],
            effort_factors=velocity_coord_params_for_sim["effort_factors"],
            max_total_effort=velocity_coord_params_for_sim["max_total_effort"],
            accel_response_time=velocity_coord_params_for_sim["accel_response_time"],
            decel_response_time=velocity_coord_params_for_sim["decel_response_time"],
            oscillation_frequency=velocity_coord_params_for_sim["oscillation_frequency"],
            oscillation_decay_rate=velocity_coord_params_for_sim["oscillation_decay_rate"],
            initial_amplitude_scale=velocity_coord_params_for_sim["initial_amplitude_scale"]
            # Removed min_velocity_for_oscillation
        )

        # --- Data Saving Logic (Moved inside function) ---
        if sim_data_tuple is not None:
             print("\nSimulation function finished successfully.")
        else:
             print("\nSimulation function failed.")

        print("\nMain script finished.")

