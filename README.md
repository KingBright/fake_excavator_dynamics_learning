
# MuJoCo 挖掘机模拟脚本 (速度协调 + 直接设置 qvel)

## 目的

该 Python 脚本旨在使用 MuJoCo 物理引擎模拟一个虚拟挖掘机的动态行为。其核心控制逻辑为：

1. 使用内置的 `VelocityCoordinator` 类来处理控制信号（来自游戏手柄或脚本）。
2. `VelocityCoordinator` 根据控制信号、最大速度限制、估算的关节“努力度”(`effort_factors`)以及系统总“努力度”上限(`max_total_effort`)，计算出协调后的目标关节角速度。
3. 该计算过程还包含了**非对称的响应延迟**（不同的加速和减速时间常数）和**停止时的阻尼振荡**效果，以模拟更真实的动态响应。
4. 最终计算出的目标角速度被**直接赋值**给 MuJoCo 的 `mj_data.qvel` 变量，**强制设定**关节的实际速度，**完全绕过了 MuJoCo 的执行器 (actuator) 和相关的力/力矩计算**。

脚本运行仿真，记录下关节的位置、速度、控制信号以及**直接设定的目标速度**等数据，并将这些数据保存到 `.npz` 文件中，通常用于后续机器学习模型（如 RNN）的训练数据准备。

**重要警告**: 这种直接修改 `mj_data.qvel` 的方法**不符合物理规律**，它覆盖了 MuJoCo 基于力、质量和惯性的速度积分过程。虽然可以直观地让关节速度匹配计算出的目标值，但在存在复杂接触、约束或需要精确动力学模拟的场景下，可能导致**不真实的行为和潜在的不稳定性**。

## 依赖项

* **mujoco**: MuJoCo 官方 Python 绑定 (`pip install mujoco`)
* **numpy**: 用于数值计算 (`pip install numpy`)
* **pygame**: 用于读取游戏手柄输入 (`pip install pygame`)
* **argparse**: (Python 标准库) 用于处理命令行参数。
* **math**: (Python 标准库) 用于指数和正弦函数。

## 关键组件

1. **主仿真脚本 (`excavator_interactive_simulator.py` 或你的文件名)**:
   * 包含 `VelocityCoordinator` 类的定义。
   * 包含 `calculate_effort_factors` 函数，用于根据模型估算努力度系数。
   * 包含 `get_gamepad_input` 函数，处理手柄输入（含死区和映射）。
   * 包含 `simulate_excavator_with_viewer` 函数，负责加载模型、初始化、运行仿真主循环（调用 `VelocityCoordinator` 计算速度并直接设置 `mj_data.qvel`）、处理可视化、记录和保存数据。
   * 包含 `if __name__ == "__main__":` 部分，处理命令行参数并启动仿真。
2. **MuJoCo XML 模型 (`excavator_modified.xml` 或你的文件名)**:
   * 定义挖掘机的刚体结构、几何形状、关节约束。
   * **必须包含**准确的**惯性参数** (`<inertial>` 标签中的 `mass`, `pos`, `diaginertia`)。
   * **必须移除或注释掉** `<actuator>` 部分，因为此脚本不使用它们。
   * 包含传感器定义 (`<sensor>`)。

## 运行前准备

1. **配置 MuJoCo XML 文件**:
   * **惯性参数**: 确保为主要运动部件（`cab`, `boom`, `arm`, `bucket`）提供了尽可能准确的 `mass` (千克), `pos` (质心相对位置, **米**), 和 `diaginertia` (主转动惯量)。这些参数的准确性对仿真行为至关重要。
2. **配置 Python 脚本参数**:
   * **命令行参数**: 可以通过命令行方便地调整许多参数（见下文）。
   * **脚本内默认值**: 如果不使用命令行参数，脚本会使用 `argparse` 中定义的默认值。你可以直接修改这些默认值。关键参数包括：
     * `max_velocities`: 各关节最大目标速度。
     * `max_total_effort`: 速度协调器的总努力度上限。
     * `accel_response_time`, `decel_response_time`: 非对称响应延迟的时间常数。
     * `oscillation_frequency`, `oscillation_decay_rate`, `initial_amplitude_scale`: 停止振荡效果的参数。

## 运行脚本

在终端中使用 `python` 命令运行主仿真脚本。

**基本用法:**

```bash
python your_main_script_name.py --xml <path_to_your_xml> [options]
```

**主要命令行参数:**

* `--xml <path>`: **必需**。指定修改好的 MuJoCo XML 文件路径。(默认: `"excavator_modified.xml"`)
* `--duration <seconds>`: 模拟的总时长（秒）。(默认: `60.0`)
* `--initial_control_mode <mode>`: 初始控制模式 (`gamepad`, `scripted`, `zero`)。(默认: `gamepad`)
* `--script_path <path>`: 如果初始模式或运行时切换到 `scripted`，需要指定包含控制信号的 `.npz` 文件路径。
* `--output <path>`: 指定保存模拟数据的输出 `.npz` 文件名。(默认: `"excavator_direct_vel_osc_delay_data.npz"`)
* `--real_time`: 尝试以接近实时速度运行（如果计算允许）。
* `--max_vel <V_CAB> <V_BOOM> <V_ARM> <V_BUCKET>`: 设置各关节最大速度 (rad/s)。(默认: `0.5 0.4 0.6 0.8`)
* `--max_effort <value>`: 设置速度协调器的最大总努力度。(默认: `1.5`)
* `--accel_time <seconds>`: 设置加速响应时间常数。(默认: `0.1`)
* `--decel_time <seconds>`: 设置减速响应时间常数。(默认: `0.2`)
* `--osc_freq <rad/s>`: 设置停止振荡频率。(默认: `15.0`)
* `--osc_decay <1/s>`: 设置停止振荡衰减率。(默认: `5.0`)
* `--osc_scale <factor>`: 设置停止振荡初始幅度比例。(默认: `0.3`)

**示例:**

```bash
# 使用手柄控制，设置不同的响应时间和振荡参数
python your_main_script_name.py --xml excavator_modified.xml --accel_time 0.05 --decel_time 0.15 --osc_freq 20 --osc_decay 8 --osc_scale 0.2
```

## 控制细节

* **游戏手柄**: 映射在 `DEFAULT_GAMEPAD_MAPPING` 定义，输入处理在 `get_gamepad_input`（包含死区和映射逻辑）。
* **脚本控制**: 需要提供 `(N, 4)` 形状的 `controls` 数组在 `.npz` 文件中。
* **运行时切换**: 可按 `G` (Gamepad), `S` (Scripted,需提供文件), `Z` (Zero) 切换模式。

## 输出文件

脚本运行结束后，生成一个 `.npz` 文件，包含：

* `time`: (N,) 时间戳。
* `qpos`: (N, nq) 关节位置。
* `qvel`: (N, nv) 关节**实际**速度（在 `mj_step` 之后记录）。
* `set_velocity`: (N, 4) 在每个 `mj_step` **之前**，脚本**直接设置**给 `mj_data.qvel` 的目标速度值。
* `control_signal`: (N, 4) 原始的 `[-1000, 1000]` 控制信号。
