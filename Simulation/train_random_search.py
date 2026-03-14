# train_random_search_english.py
"""
Random Search for fine-tuning keyframe gait (output Δq)
- 3-second startup stabilization period
- Transition_steps=1000
- IMU stability limits (penalty starts at 10°, termination at 40°)
- Random search instead of policy gradient
"""

import os
import time
import math
import random
import numpy as np
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend to avoid font issues
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning)  # Suppress font warnings

# --------------------------
# Hyperparameters
# --------------------------
XML_PATH = "fufu.xml"

# Random Search parameters
POPULATION_SIZE = 20  # 种群大小
GENERATIONS = 500  # 迭代代数
MUTATION_RATE = 0.1  # 变异率
MUTATION_SCALE = 0.05  # 变异幅度
ELITISM_COUNT = 4  # 精英保留数量

# --------------------------
# Training Process
# --------------------------
max_episode_steps = 12000  # 1200 transition_steps

# --------------------------
# Action Constraints
# --------------------------
DELTA_Q_LIMIT = 0.25  # rad, 保持不变

# --------------------------
# IMU Limits (基于实际数据: 加速度最大27.6, 角速度最大0.36)
# --------------------------
ACC_CLIP = 30.0  # 恢复到 30（从 25）
GYRO_CLIP = 5.0  # 大幅放宽（从 0.5）⚠️ 关键修改
IMU_ACCEL_TERMINATE = 60.0  # 放宽终止（从 45）
IMU_GYRO_TERMINATE = 10.0  # 大幅放宽（从 1.0）⚠️ 关键修改

# --------------------------
# Posture Limits (放宽以便学习)
# --------------------------
MAX_ROLL_PITCH = 0.175  # ~10°, 终止阈值（从 5° 放宽）
MAX_TILT = 0.0873  # ~5°, 惩罚阈值（从 1° 放宽）⚠️ 关键修改
TILT_PENALTY_SCALE = 0.1  # 减小惩罚（从 0.2）

# --------------------------
# Reward Weights
# --------------------------
WEIGHT_FORWARD = 1.5  # 强调前进速度
WEIGHT_IMU_PENALTY = 0.3  # IMU 惩罚权重
WEIGHT_ENERGY = 5e-4  # 能量惩罚
WEIGHT_JERK = 1e-4  # Jerk 惩罚

# Testing / Saving
model_save_path = "random_search_best.pt"
test_interval = 50  # Test with visualization every N updates
test_duration_steps = 12000  # Steps during testing
render_on_test = False

# ------------- keyframe section -------------
keyframes_degrees = {
    'o1': [0, 0, 0, 0, 0, 0, 0, 0],
    'o2': [-27.5, 0, 32.5, 0, -5, 0, 0, 0],
    'o3': [-22.5, -5, 20, 27, 2.5, -22, 28, -28],
    'o4': [-18, -19.5, 31.5, 60, -13.5, -40.5, 22, -22],
    'o5': [0, -39, 0, 70, 0, -31, 0, 0],
    'o6': [-14, -36, 45, 30, -31, 6, 28, -28],
    'o7': [-19.5, -18, 60, 31.5, -40.5, -13.5, 22, -22]
}
keyframes_radians = {k: [math.radians(x) for x in v] for k, v in keyframes_degrees.items()}
joint_order = [
    'right_hip_joint', 'left_hip_joint',
    'right_knee_joint', 'left_knee_joint',
    'right_ankle_joint', 'left_ankle_joint',
    'right_hair_joint', 'left_hair_joint'
]
animation_sequence = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7']
transition_steps = 1000  # Keep 1000 unchanged

# actuator mapping
actuator_mapping = {
    'right_hip_joint': 'right_hip_servo',
    'left_hip_joint': 'left_hip_servo',
    'right_knee_joint': 'right_knee_servo',
    'left_knee_joint': 'left_knee_servo',
    'right_ankle_joint': 'right_ankle_servo',
    'left_ankle_joint': 'left_ankle_servo',
    'right_hair_joint': 'right_hair_servo',
    'left_hair_joint': 'left_hair_servo'
}


# -----------------------
# Environment wrapper
# -----------------------
class KeyframeBaseline:
    """Improved Keyframe interpolator with 3-second startup delay"""

    def __init__(self, keyframes, joint_order, anim_seq, transition_steps, start_delay=3.0):
        self.keyframes = keyframes
        self.joint_order = joint_order
        self.anim_seq = anim_seq
        self.transition_steps = transition_steps  # Keep 1000 unchanged
        self.start_delay = start_delay
        self.start_time = None
        self.started = False
        self.reset()

    def reset(self):
        self.current_frame_index = 0
        self.transition_progress = 0
        self.step_count = 0
        self.start_time = None
        self.started = False

    def step(self):
        # Initialize start time
        if self.start_time is None:
            self.start_time = time.time()

        # Check if startup delay has passed
        if not self.started and time.time() - self.start_time >= self.start_delay:
            self.started = True

        # If not started yet, maintain o1 posture
        if not self.started:
            targets = dict(zip(self.joint_order, self.keyframes[self.anim_seq[0]]))
            return targets

        # Normal interpolation logic
        current_key = self.anim_seq[self.current_frame_index]
        if self.current_frame_index == len(self.anim_seq) - 1:
            next_key = 'o2'
        else:
            next_key = self.anim_seq[self.current_frame_index + 1]

        cur = self.keyframes[current_key]
        nxt = self.keyframes[next_key]

        if self.transition_progress < self.transition_steps:
            alpha = self.transition_progress / self.transition_steps
            t = alpha * alpha * (3 - 2 * alpha)  # ease_in_out
            targets = [cur[i] + (nxt[i] - cur[i]) * t for i in range(len(cur))]
        else:
            targets = nxt

        # Update state
        self.step_count += 1
        self.transition_progress += 1
        if self.transition_progress >= self.transition_steps:
            self.transition_progress = 0
            self.current_frame_index += 1
            if self.current_frame_index >= len(self.anim_seq):
                self.current_frame_index = 1

        return dict(zip(self.joint_order, targets))


class MujocoWalkEnv:
    """MuJoCo environment wrapper"""

    def __init__(self, xml_path, render=False, start_delay=3.0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render = render
        self.viewer = None

        if render:
            try:
                from mujoco.viewer import launch_passive
                self.viewer = launch_passive(self.model, self.data)
            except Exception as e:
                print("Failed to launch viewer:", e)
                self.render = False
                self.viewer = None

        # ID mappings
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "main_body")
        # sensor ids
        self.accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")

        # joint and actuator ids
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_order]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_mapping[j]) for j in
                             joint_order]

        # keyframe baseline (with 3-second delay)
        self.kf = KeyframeBaseline(keyframes_radians, joint_order, animation_sequence,
                                   transition_steps, start_delay=start_delay)

        # internal state
        self.prev_accel = np.zeros(3)
        self.prev_gyro = np.zeros(3)
        # reset sim
        self.reset()

    def reset(self):
        # Full reset to initial XML state
        mujoco.mj_resetData(self.model, self.data)
        self.kf.reset()
        self.prev_accel[:] = 0.0
        self.prev_gyro[:] = 0.0

        # Step a few times to settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Return initial observation
        return self._get_obs()

    def render_frame(self):
        if self.render and self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.render and self.viewer is not None:
            self.viewer.close()

    def _read_imu(self):
        # Read from sensordata
        try:
            base_idx = self.model.sensor_adr[self.accel_id]
            accel = np.array(self.data.sensordata[base_idx:base_idx + 3], dtype=np.float64)
            base_idx = self.model.sensor_adr[self.gyro_id]
            gyro = np.array(self.data.sensordata[base_idx:base_idx + 3], dtype=np.float64)
            base_idx = self.model.sensor_adr[self.quat_id]
            quat = np.array(self.data.sensordata[base_idx:base_idx + 4], dtype=np.float64)
        except Exception as e:
            # Fallback if index fails due to version differences
            accel = np.zeros(3)
            gyro = np.zeros(3)
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        # Force normalize quaternion and unify sign
        qnorm = np.linalg.norm(quat) + 1e-12
        quat = quat / qnorm
        if quat[0] < 0:
            quat = -quat
        return accel, gyro, quat

    def _get_base_pos_vel(self):
        # Try to read body linear velocity and position
        try:
            pos = np.array(self.data.xpos[self.body_id].copy()) if hasattr(self.data, "xpos") and len(
                self.data.xpos) > self.body_id else np.zeros(3)
            if hasattr(self.data, "xvelp"):
                vel = np.array(self.data.xvelp[self.body_id].copy())
            else:
                vel = np.zeros(3)
        except Exception:
            pos = np.zeros(3)
            vel = np.zeros(3)
        return pos, vel

    def _get_obs(self):
        """
        Return current observation
        """
        # 1) Current keyframe target
        kf_targets_dict = self.kf.step()
        kf_vec = np.array([kf_targets_dict[j] for j in joint_order], dtype=np.float32)

        # 2) Full qpos/qvel
        qpos = np.array(self.data.qpos, dtype=np.float32)
        qvel = np.array(self.data.qvel, dtype=np.float32)

        # 3) IMU
        accel, gyro, quat = self._read_imu()

        # --- low-pass filter ---
        tau = 0.09
        dt = self.model.opt.timestep
        alpha = math.exp(-dt / tau)
        accel_f = alpha * self.prev_accel + (1 - alpha) * accel
        gyro_f = alpha * self.prev_gyro + (1 - alpha) * gyro
        self.prev_accel = accel_f
        self.prev_gyro = gyro_f

        # --- clipping & normalization ---
        accel_c = np.clip(accel_f, -ACC_CLIP, ACC_CLIP) / ACC_CLIP
        gyro_c = np.clip(gyro_f, -GYRO_CLIP, GYRO_CLIP) / GYRO_CLIP

        # 4) Concatenate into observation
        obs = np.concatenate([
            kf_vec,  # keyframe target
            qpos,
            qvel,
            accel_c,
            gyro_c,
            quat
        ]).astype(np.float32)

        return obs

    def step(self, action_delta):
        """
        Execute one simulation step
        """
        delta = np.clip(action_delta, -DELTA_Q_LIMIT, DELTA_Q_LIMIT)

        # 1) Get current keyframe target
        kf_targets_dict = self.kf.step()
        for i, jname in enumerate(joint_order):
            actuator_id = self.actuator_ids[i]
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = kf_targets_dict[jname] + float(delta[i])

        # 2) Step simulation
        mujoco.mj_step(self.model, self.data)

        # 3) Read new observation
        obs = self._get_obs()

        # 4) Reward calculation
        pos, vel = self._get_base_pos_vel()

        # Assume robot moves in -X direction
        forward_vel = -vel[0]  # Move in -X direction

        reward_forward = WEIGHT_FORWARD * forward_vel

        # Energy consumption penalty
        energy = np.sum(np.abs(delta))

        # IMU reading
        accel, gyro, quat = self._read_imu()
        accel_norm = np.linalg.norm(accel)
        gyro_norm = np.linalg.norm(gyro)

        # Calculate torso tilt angle
        w, x, y, z = quat
        # Calculate tilt angle relative to initial upright posture
        tilt = np.arccos(min(1.0, max(-1.0, 1 - 2 * (x * x + y * y))))

        # Posture penalty: starts at 10° (0.1745 rad)
        tilt_penalty = 0.0
        if tilt > MAX_TILT:
            tilt_penalty = TILT_PENALTY_SCALE * (tilt - MAX_TILT) ** 2

        # IMU limit penalty
        accel_penalty = 0.0
        gyro_penalty = 0.0
        if accel_norm > IMU_ACCEL_TERMINATE:
            accel_penalty = WEIGHT_IMU_PENALTY * (accel_norm - IMU_ACCEL_TERMINATE)
        if gyro_norm > IMU_GYRO_TERMINATE:
            gyro_penalty = WEIGHT_IMU_PENALTY * (gyro_norm - IMU_GYRO_TERMINATE)

        # Combined reward
        reward = reward_forward - WEIGHT_ENERGY * energy - tilt_penalty - accel_penalty - gyro_penalty

        # Termination condition (only terminate on severe tilt)
        done = False
        info = {}
        if tilt > MAX_ROLL_PITCH:  # ~40° termination
            done = True
            info['termination'] = 'tilt_exceed'
            reward -= 10.0  # Termination penalty

        # Debug information
        info.update({
            'forward_vel': forward_vel,
            'tilt_deg': math.degrees(tilt),
            'accel_norm': accel_norm,
            'gyro_norm': gyro_norm
        })

        return obs, float(reward), done, info


# -------------------------
# Random Search Individual
# -------------------------
class RandomSearchIndividual:
    """个体表示一个完整的动作序列"""

    def __init__(self, seq_length, action_dim):
        self.seq_length = seq_length  # 序列长度（步数）
        self.action_dim = action_dim  # 动作维度（关节数）

        # 初始化动作序列：seq_length × action_dim
        # 使用较小的初始值，避免过大扰动
        self.actions = np.random.uniform(-0.05, 0.05, (seq_length, action_dim))
        self.fitness = -np.inf  # 适应度（奖励）

    def mutate(self, mutation_rate, mutation_scale):
        """变异操作"""
        mask = np.random.rand(*self.actions.shape) < mutation_rate
        mutation = np.random.randn(*self.actions.shape) * mutation_scale
        self.actions[mask] += mutation[mask]

        # 确保在限制范围内
        self.actions = np.clip(self.actions, -DELTA_Q_LIMIT, DELTA_Q_LIMIT)

    def crossover(self, other):
        """交叉操作：两点交叉"""
        child = RandomSearchIndividual(self.seq_length, self.action_dim)

        # 选择两个交叉点
        point1 = np.random.randint(0, self.seq_length)
        point2 = np.random.randint(point1, self.seq_length)

        # 交叉
        child.actions[:point1] = self.actions[:point1]
        child.actions[point1:point2] = other.actions[point1:point2]
        child.actions[point2:] = self.actions[point2:]

        return child

    def evaluate(self, env):
        """评估个体的适应度"""
        env.reset()
        total_reward = 0.0

        for t in range(min(self.seq_length, max_episode_steps)):
            # 获取当前步的动作
            action = self.actions[t]
            # 执行动作
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                # 如果提前终止，惩罚未完成的步数
                penalty = (self.seq_length - t) * 0.1
                total_reward -= penalty
                break

        self.fitness = total_reward
        return total_reward


# -------------------------
# Random Search Population
# -------------------------
class RandomSearchPopulation:
    """随机搜索种群"""

    def __init__(self, population_size, seq_length, action_dim):
        self.population_size = population_size
        self.seq_length = seq_length
        self.action_dim = action_dim
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness = -np.inf

        # 初始化种群
        for _ in range(population_size):
            ind = RandomSearchIndividual(seq_length, action_dim)
            self.population.append(ind)

    def evaluate_all(self, env):
        """评估所有个体"""
        for ind in self.population:
            ind.evaluate(env)

        # 排序：按适应度降序
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # 更新最佳个体
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_individual = self.population[0]

        return self.population[0].fitness, np.mean([ind.fitness for ind in self.population])

    def next_generation(self, mutation_rate, mutation_scale, elitism_count):
        """生成下一代"""
        new_population = []

        # 精英保留
        for i in range(elitism_count):
            elite = RandomSearchIndividual(self.seq_length, self.action_dim)
            elite.actions = self.population[i].actions.copy()
            elite.fitness = self.population[i].fitness
            new_population.append(elite)

        # 交叉和变异生成剩余个体
        while len(new_population) < self.population_size:
            # 锦标赛选择父代
            parent1 = self.tournament_selection(3)
            parent2 = self.tournament_selection(3)

            # 交叉
            child = parent1.crossover(parent2)

            # 变异
            child.mutate(mutation_rate, mutation_scale)

            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def tournament_selection(self, tournament_size):
        """锦标赛选择"""
        participants = np.random.choice(self.population, tournament_size, replace=False)
        return max(participants, key=lambda x: x.fitness)


# -------------------------
# Helper functions
# -------------------------
def replay_best_individual(best_individual, target_cycles=2):
    """Replay best individual"""
    print(f"\nCollecting {target_cycles} complete gait cycles...")

    # Create new environment
    env_view = MujocoWalkEnv(XML_PATH, render=False, start_delay=0.0)
    from mujoco.viewer import launch_passive
    viewer = launch_passive(env_view.model, env_view.data)

    obs = env_view.reset()
    recorded_deltaq = []
    recorded_info = []

    cycle_length = len(animation_sequence) * transition_steps
    target_steps = target_cycles * cycle_length

    print(f"Target steps: {target_steps} ({target_cycles} cycles × {cycle_length} steps/cycle)")
    print("Starting collection...")

    step_count = 0
    completed_cycles = 0
    prev_frame_idx = 0

    while step_count < target_steps:
        # 如果步数超过最佳个体的序列长度，则循环使用
        action_idx = step_count % best_individual.seq_length
        act = best_individual.actions[action_idx]
        act = np.clip(act, -DELTA_Q_LIMIT, DELTA_Q_LIMIT)

        # Execute action
        obs, reward, done, info = env_view.step(act)

        # Record data
        recorded_deltaq.append(act.copy())
        recorded_info.append(info.copy())

        # Track cycle completion
        if env_view.kf.current_frame_index < prev_frame_idx:
            completed_cycles += 1
            print(f"  Completed cycle {completed_cycles}/{target_cycles}")
        prev_frame_idx = env_view.kf.current_frame_index

        # Render
        viewer.sync()
        time.sleep(0.001)

        step_count += 1

        # Check termination
        if done:
            print(f"WARNING: Episode terminated early at step {step_count}")
            print(f"  Reason: {info.get('termination', 'unknown')}")
            print(f"  Collected {completed_cycles} complete cycles")
            break

        # Progress update
        if step_count % 1000 == 0:
            print(f"  Step {step_count}/{target_steps}, cycle {completed_cycles}/{target_cycles}")

    print(f"\nCollection complete!")
    print(f"  Total steps: {len(recorded_deltaq)}")
    print(f"  Complete cycles: {completed_cycles}")

    # Save data
    np.save("best_random_search_deltaq.npy", recorded_deltaq)
    df = pd.DataFrame(recorded_deltaq, columns=joint_order)
    df.to_csv("best_random_search_deltaq.csv", index=False)

    print(f"\nSaved data to:")
    print(f"  - best_random_search_deltaq.npy ({len(recorded_deltaq)} steps)")
    print(f"  - best_random_search_deltaq.csv")

    # Also save info
    info_df = pd.DataFrame(recorded_info)
    info_df.to_csv("random_search_info.csv", index=False)
    print(f"  - random_search_info.csv (diagnostics)")

    # Keep viewer open for inspection
    print("\nPress Ctrl+C to exit viewer...")
    try:
        while True:
            viewer.sync()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Closing viewer...")
        env_view.close()


def test_best_individual(best_individual, steps=500, render=True):
    """Test best individual"""
    print(f"Testing best individual, steps={steps}")
    env = MujocoWalkEnv(XML_PATH, render=render, start_delay=0.0)
    obs = env.reset()
    total_reward = 0.0
    rewards = []

    for t in range(steps):
        # 循环使用动作序列
        action_idx = t % best_individual.seq_length
        act = best_individual.actions[action_idx]
        act = np.clip(act, -DELTA_Q_LIMIT, DELTA_Q_LIMIT)

        obs, reward, done, info = env.step(act)
        total_reward += reward
        rewards.append(reward)

        if render:
            env.render_frame()
            time.sleep(0.001)

        if done:
            print(f"Test terminated early: {info.get('termination', 'unknown')}")
            break

    print(f"Test complete: total_reward={total_reward:.3f}, avg_reward={np.mean(rewards):.3f}, steps={len(rewards)}")
    env.close()
    return rewards


# -------------------------
# Main training loop
# -------------------------
def train_random_search():
    """Main random search training function"""
    print("=" * 50)
    print("Starting Random Search Training")
    print(f"Population size: {POPULATION_SIZE}, Generations: {GENERATIONS}")
    print(f"Mutation rate: {MUTATION_RATE}, Mutation scale: {MUTATION_SCALE}")
    print(f"Elitism count: {ELITISM_COUNT}")
    print(f"IMU limits: penalty at >10° tilt, termination at >40° tilt")
    print(f"Startup delay: 3 seconds")
    print("=" * 50)

    # Initialize environment
    env = MujocoWalkEnv(XML_PATH, render=False, start_delay=0.0)
    obs0 = env.reset()
    obs_dim = obs0.size
    act_dim = len(joint_order)

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")
    print(f"Max episode steps: {max_episode_steps}")

    # Initialize random search population
    # 序列长度设为关键帧周期的倍数，确保完整周期
    seq_length = len(animation_sequence) * transition_steps  # 一个完整周期
    print(
        f"Individual sequence length: {seq_length} steps ({len(animation_sequence)} keyframes × {transition_steps} steps)")

    population = RandomSearchPopulation(POPULATION_SIZE, seq_length, act_dim)

    # Training statistics
    best_fitness_history = []
    avg_fitness_history = []
    generation_counts = []

    # Save best individual
    best_individual = None

    # Training log
    log_file = open("random_search_log.txt", "w")
    log_file.write("Generation,Best_Fitness,Avg_Fitness\n")

    try:
        for gen in range(GENERATIONS):
            # 评估当前种群
            best_fitness, avg_fitness = population.evaluate_all(env)

            # 记录历史
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            generation_counts.append(gen + 1)

            # 保存最佳个体
            if best_individual is None or best_fitness > best_individual.fitness:
                best_individual = RandomSearchIndividual(seq_length, act_dim)
                best_individual.actions = population.best_individual.actions.copy()
                best_individual.fitness = best_fitness

            # 记录日志
            log_file.write(f"{gen + 1},{best_fitness:.3f},{avg_fitness:.3f}\n")
            log_file.flush()

            # 打印进度
            print(f"Generation {gen + 1}/{GENERATIONS}: "
                  f"Best={best_fitness:.3f}, Avg={avg_fitness:.3f}, "
                  f"Best Individual Steps={population.best_individual.seq_length}")

            # 生成下一代
            if gen < GENERATIONS - 1:  # 最后一代不需要再生成下一代
                population.next_generation(MUTATION_RATE, MUTATION_SCALE, ELITISM_COUNT)

            # 定期测试和保存
            if (gen + 1) % test_interval == 0 or gen == GENERATIONS - 1:
                print(f"\nTesting best individual at generation {gen + 1}")
                test_rewards = test_best_individual(best_individual, steps=test_duration_steps, render=render_on_test)

                # 保存最佳个体
                np.save("best_random_search_individual.npy", best_individual.actions)
                print(f"Best individual saved to best_random_search_individual.npy")

                # 绘制训练进度
                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                plt.plot(generation_counts, best_fitness_history, 'b-', linewidth=2, label='Best Fitness')
                plt.plot(generation_counts, avg_fitness_history, 'g-', linewidth=2, alpha=0.7, label='Avg Fitness')
                plt.xlabel("Generation")
                plt.ylabel("Fitness (Reward)")
                plt.title("Random Search Progress")
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 2, 2)
                # 绘制最近50代的移动平均
                window = min(20, len(best_fitness_history))
                if window > 0:
                    moving_avg = np.convolve(best_fitness_history, np.ones(window) / window, mode='valid')
                    plt.plot(generation_counts[window - 1:], moving_avg, 'r-', linewidth=2,
                             label=f'Moving Avg (window={window})')
                    plt.xlabel("Generation")
                    plt.ylabel("Moving Average")
                    plt.title("Trend Analysis")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig("random_search_progress.png", dpi=150)
                plt.close()
                print("Progress plot saved to random_search_progress.png")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        # Close log file
        log_file.close()

        # Final save
        np.save("final_best_individual.npy", best_individual.actions)
        print("Final best individual saved as final_best_individual.npy")

        # Plot final training curve
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(generation_counts, best_fitness_history, 'b-', linewidth=2, label='Best Fitness')
        plt.plot(generation_counts, avg_fitness_history, 'g-', linewidth=2, alpha=0.7, label='Average Fitness')
        plt.xlabel("Generation")
        plt.ylabel("Fitness (Reward)")
        plt.title("Random Search Training History")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        # 计算每10代的平均
        if len(best_fitness_history) >= 10:
            group_size = 10
            grouped_best = []
            grouped_avg = []
            grouped_gen = []
            for i in range(0, len(best_fitness_history), group_size):
                group_best = best_fitness_history[i:i + group_size]
                group_avg = avg_fitness_history[i:i + group_size]
                if len(group_best) > 0:
                    grouped_best.append(np.mean(group_best))
                    grouped_avg.append(np.mean(group_avg))
                    grouped_gen.append(generation_counts[min(i + group_size - 1, len(generation_counts) - 1)])
            plt.plot(grouped_gen, grouped_best, 'b-o', linewidth=2, label='Best (per 10 gens)')
            plt.plot(grouped_gen, grouped_avg, 'g-o', linewidth=2, alpha=0.7, label='Avg (per 10 gens)')
            plt.xlabel("Generation")
            plt.ylabel("Fitness (Reward)")
            plt.title("Grouped Averages (every 10 generations)")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("final_random_search_progress.png", dpi=150)
        plt.show()

        # Replay best individual
        if best_individual is not None:
            replay_best_individual(best_individual, target_cycles=2)

        print(f"Random Search training complete!")
        print(f"Best fitness achieved: {best_individual.fitness:.3f}")
        print(f"Total generations: {len(generation_counts)}")


# -------------------------
# Main program
# -------------------------
if __name__ == "__main__":
    # Start random search training
    train_random_search()