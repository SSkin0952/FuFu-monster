# train_hill_climber.py
"""
Hill Climber for fine-tuning keyframe gait (output Δq)
- 3-second startup stabilization period
- Transition_steps=1000
- IMU stability limits (penalty starts at 10°, termination at 40°)
- Hill climbing algorithm with adaptive step size
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

# Hill Climber parameters
MAX_ITERATIONS = 500  # Maximum number of iterations
INITIAL_STEP_SIZE = 0.1  # Initial step size
STEP_SIZE_DECAY = 0.99  # Step size decay rate
MIN_STEP_SIZE = 0.01  # Minimum step size
NOISE_SCALE = 0.05  # Random noise magnitude
RESTART_PROB = 0.05  # Restart probability (when trapped in local optimum)
PATIENCE = 20  # Number of iterations tolerated without improvement

# --------------------------
# Training Process
# --------------------------
max_episode_steps = 12000  # 1200 transition_steps

# --------------------------
# Action Constraints
# --------------------------
DELTA_Q_LIMIT = 0.25  # rad, kept unchanged

# --------------------------
# IMU Limits (based on real data: max acceleration 27.6, max angular velocity 0.36)
# --------------------------
ACC_CLIP = 30.0  # Restored to 30 (from 25)
GYRO_CLIP = 5.0  # Significantly relaxed (from 0.5) ⚠️ key modification
IMU_ACCEL_TERMINATE = 60.0  # Relaxed termination threshold (from 45)
IMU_GYRO_TERMINATE = 10.0  # Significantly relaxed (from 1.0) ⚠️ key modification

# --------------------------
# Posture Limits (relaxed to facilitate learning)
# --------------------------
MAX_ROLL_PITCH = 0.175  # ~10°, termination threshold (relaxed from 5°)
MAX_TILT = 0.0873  # ~5°, penalty threshold (relaxed from 1°) ⚠️ key modification
TILT_PENALTY_SCALE = 0.1  # Reduced penalty weight (from 0.2)

# --------------------------
# Reward Weights
# --------------------------
WEIGHT_FORWARD = 1.5  # Emphasize forward speed
WEIGHT_IMU_PENALTY = 0.3  # IMU penalty weight
WEIGHT_ENERGY = 5e-4  # Energy penalty
WEIGHT_JERK = 1e-4  # Jerk penalty

# Testing / Saving
model_save_path = "hill_climber_best.pt"
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

# Actuator mapping
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
    """Improved keyframe interpolator with a 3-second startup delay"""

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

        # Check whether the startup delay has elapsed
        if not self.started and time.time() - self.start_time >= self.start_delay:
            self.started = True

        # Before startup, maintain the o1 posture
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

        # Update internal state
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
        # Sensor IDs
        self.accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")

        # Joint and actuator IDs
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_order]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_mapping[j]) for j in
                             joint_order]

        # Keyframe baseline (with 3-second delay)
        self.kf = KeyframeBaseline(keyframes_radians, joint_order, animation_sequence,
                                   transition_steps, start_delay=start_delay)

        # Internal state
        self.prev_accel = np.zeros(3)
        self.prev_gyro = np.zeros(3)
        # Reset simulation
        self.reset()

    def reset(self):
        # Full reset to the initial XML state
        mujoco.mj_resetData(self.model, self.data)
        self.kf.reset()
        self.prev_accel[:] = 0.0
        self.prev_gyro[:] = 0.0

        # Step a few times to let the system settle
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
            # Fallback if indexing fails due to version differences
            accel = np.zeros(3)
            gyro = np.zeros(3)
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        # Force quaternion normalization and unify sign
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
        Return the current observation
        """
        # 1) Current keyframe target
        kf_targets_dict = self.kf.step()
        kf_vec = np.array([kf_targets_dict[j] for j in joint_order], dtype=np.float32)

        # 2) Full qpos/qvel
        qpos = np.array(self.data.qpos, dtype=np.float32)
        qvel = np.array(self.data.qvel, dtype=np.float32)

        # 3) IMU
        accel, gyro, quat = self._read_imu()

        # --- Low-pass filter ---
        tau = 0.09
        dt = self.model.opt.timestep
        alpha = math.exp(-dt / tau)
        accel_f = alpha * self.prev_accel + (1 - alpha) * accel
        gyro_f = alpha * self.prev_gyro + (1 - alpha) * gyro
        self.prev_accel = accel_f
        self.prev_gyro = gyro_f

        # --- Clipping and normalization ---
        accel_c = np.clip(accel_f, -ACC_CLIP, ACC_CLIP) / ACC_CLIP
        gyro_c = np.clip(gyro_f, -GYRO_CLIP, GYRO_CLIP) / GYRO_CLIP

        # 4) Concatenate into observation
        obs = np.concatenate([
            kf_vec,  # Keyframe target
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

        # 1) Get the current keyframe target
        kf_targets_dict = self.kf.step()
        for i, jname in enumerate(joint_order):
            actuator_id = self.actuator_ids[i]
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = kf_targets_dict[jname] + float(delta[i])

        # 2) Step the simulation
        mujoco.mj_step(self.model, self.data)

        # 3) Read the new observation
        obs = self._get_obs()

        # 4) Reward calculation
        pos, vel = self._get_base_pos_vel()

        # Assume the robot moves in the -X direction
        forward_vel = -vel[0]  # Move in the -X direction

        reward_forward = WEIGHT_FORWARD * forward_vel

        # Energy consumption penalty
        energy = np.sum(np.abs(delta))

        # IMU readings
        accel, gyro, quat = self._read_imu()
        accel_norm = np.linalg.norm(accel)
        gyro_norm = np.linalg.norm(gyro)

        # Calculate torso tilt angle
        w, x, y, z = quat
        # Calculate tilt angle relative to the initial upright posture
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

        # Termination condition (terminate only for severe tilt)
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
# Hill Climber Solution
# -------------------------
class HillClimberSolution:
    """Solution representation for the hill climbing algorithm"""

    def __init__(self, seq_length, action_dim):
        self.seq_length = seq_length  # Sequence length
        self.action_dim = action_dim  # Action dimension

        # Initialize action sequence
        self.actions = np.zeros((seq_length, action_dim), dtype=np.float32)
        self.fitness = -np.inf  # Fitness

    def initialize_random(self):
        """Random initialization"""
        self.actions = np.random.uniform(-0.05, 0.05, (self.seq_length, self.action_dim))
        self.actions = np.clip(self.actions, -DELTA_Q_LIMIT, DELTA_Q_LIMIT)

    def evaluate(self, env):
        """Evaluate solution quality"""
        env.reset()
        total_reward = 0.0

        for t in range(min(self.seq_length, max_episode_steps)):
            action = self.actions[t]
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                # If terminated early, penalize the unfinished steps
                penalty = (self.seq_length - t) * 0.1
                total_reward -= penalty
                break

        self.fitness = total_reward
        return total_reward

    def generate_neighbor(self, step_size, noise_scale):
        """Generate a neighboring solution by perturbing the current one"""
        neighbor = HillClimberSolution(self.seq_length, self.action_dim)

        # Copy the current solution
        neighbor.actions = self.actions.copy()

        # Add perturbation
        noise = np.random.randn(*self.actions.shape) * noise_scale
        neighbor.actions += noise * step_size

        # Ensure the solution remains within limits
        neighbor.actions = np.clip(neighbor.actions, -DELTA_Q_LIMIT, DELTA_Q_LIMIT)

        return neighbor

    def restart(self):
        """Restart by randomly generating a new solution"""
        self.initialize_random()
        self.fitness = -np.inf


# -------------------------
# Hill Climber Algorithm
# -------------------------
class HillClimber:
    """Main hill climbing algorithm class"""

    def __init__(self, seq_length, action_dim):
        self.seq_length = seq_length
        self.action_dim = action_dim

        # Current solution and best solution
        self.current_solution = HillClimberSolution(seq_length, action_dim)
        self.best_solution = HillClimberSolution(seq_length, action_dim)

        # Algorithm state
        self.step_size = INITIAL_STEP_SIZE
        self.no_improve_count = 0
        self.iteration = 0

        # History records
        self.best_fitness_history = []
        self.current_fitness_history = []
        self.step_size_history = []

    def initialize(self, env):
        """Initialize the algorithm"""
        self.current_solution.initialize_random()
        self.current_solution.evaluate(env)

        self.best_solution.actions = self.current_solution.actions.copy()
        self.best_solution.fitness = self.current_solution.fitness

        print(f"Initial fitness: {self.current_solution.fitness:.3f}")

    def iterate(self, env):
        """Run one iteration"""
        self.iteration += 1

        # Generate neighboring solution
        neighbor = self.current_solution.generate_neighbor(self.step_size, NOISE_SCALE)
        neighbor_fitness = neighbor.evaluate(env)

        # Accept the neighbor if it is better
        if neighbor_fitness > self.current_solution.fitness:
            self.current_solution.actions = neighbor.actions.copy()
            self.current_solution.fitness = neighbor_fitness
            self.no_improve_count = 0  # Reset no-improvement counter

            # Update the best solution
            if neighbor_fitness > self.best_solution.fitness:
                self.best_solution.actions = neighbor.actions.copy()
                self.best_solution.fitness = neighbor_fitness
                print(f"Iteration {self.iteration}: New best fitness: {neighbor_fitness:.3f}")
        else:
            self.no_improve_count += 1

        # Adaptively adjust step size
        self.step_size = max(MIN_STEP_SIZE, self.step_size * STEP_SIZE_DECAY)

        # Random restart when trapped in a local optimum
        if np.random.random() < RESTART_PROB or self.no_improve_count > PATIENCE:
            self.current_solution.restart()
            self.current_solution.evaluate(env)
            self.no_improve_count = 0
            print(f"Iteration {self.iteration}: Random restart, fitness: {self.current_solution.fitness:.3f}")

        # Record history
        self.best_fitness_history.append(self.best_solution.fitness)
        self.current_fitness_history.append(self.current_solution.fitness)
        self.step_size_history.append(self.step_size)

        return self.best_solution.fitness, self.current_solution.fitness


# -------------------------
# Helper functions
# -------------------------
def replay_best_solution(best_solution, target_cycles=2):
    """Replay the best solution"""
    print(f"\nCollecting {target_cycles} complete gait cycles...")

    # Create a new environment
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
        # If the step count exceeds the solution sequence length, reuse it cyclically
        action_idx = step_count % best_solution.seq_length
        act = best_solution.actions[action_idx]
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
    np.save("best_hill_climber_deltaq.npy", recorded_deltaq)
    df = pd.DataFrame(recorded_deltaq, columns=joint_order)
    df.to_csv("best_hill_climber_deltaq.csv", index=False)

    print(f"\nSaved data to:")
    print(f"  - best_hill_climber_deltaq.npy ({len(recorded_deltaq)} steps)")
    print(f"  - best_hill_climber_deltaq.csv")

    # Also save diagnostic information
    info_df = pd.DataFrame(recorded_info)
    info_df.to_csv("hill_climber_info.csv", index=False)
    print(f"  - hill_climber_info.csv (diagnostics)")

    # Keep the viewer open for inspection
    print("\nPress Ctrl+C to exit viewer...")
    try:
        while True:
            viewer.sync()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Closing viewer...")
        env_view.close()


def test_best_solution(best_solution, steps=500, render=True):
    """Test the best solution"""
    print(f"Testing best solution, steps={steps}")
    env = MujocoWalkEnv(XML_PATH, render=render, start_delay=0.0)
    obs = env.reset()
    total_reward = 0.0
    rewards = []

    for t in range(steps):
        # Reuse the action sequence cyclically
        action_idx = t % best_solution.seq_length
        act = best_solution.actions[action_idx]
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
def train_hill_climber():
    """Main hill climber training function"""
    print("=" * 50)
    print("Starting Hill Climber Training")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Initial step size: {INITIAL_STEP_SIZE}, Decay: {STEP_SIZE_DECAY}")
    print(f"Noise scale: {NOISE_SCALE}, Restart probability: {RESTART_PROB}")
    print(f"Patience (no improvement): {PATIENCE} iterations")
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

    # Initialize hill climber
    seq_length = len(animation_sequence) * transition_steps  # One complete cycle
    print(
        f"Solution sequence length: {seq_length} steps ({len(animation_sequence)} keyframes × {transition_steps} steps)")

    hill_climber = HillClimber(seq_length, act_dim)
    hill_climber.initialize(env)

    # Training log
    log_file = open("hill_climber_log.txt", "w")
    log_file.write("Iteration,Best_Fitness,Current_Fitness,Step_Size\n")

    try:
        for iteration in range(MAX_ITERATIONS):
            # Run one iteration
            best_fitness, current_fitness = hill_climber.iterate(env)

            # Record log
            log_file.write(f"{iteration + 1},{best_fitness:.3f},{current_fitness:.3f},{hill_climber.step_size:.6f}\n")
            log_file.flush()

            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{MAX_ITERATIONS}: "
                      f"Best={best_fitness:.3f}, Current={current_fitness:.3f}, "
                      f"Step size={hill_climber.step_size:.4f}, "
                      f"No improve={hill_climber.no_improve_count}")

            # Periodically test and save
            if (iteration + 1) % test_interval == 0 or iteration == MAX_ITERATIONS - 1:
                print(f"\nTesting best solution at iteration {iteration + 1}")
                test_rewards = test_best_solution(hill_climber.best_solution, steps=test_duration_steps,
                                                  render=render_on_test)

                # Save the best solution
                np.save("best_hill_climber_solution.npy", hill_climber.best_solution.actions)
                print(f"Best solution saved to best_hill_climber_solution.npy")

                # Plot training progress
                plt.figure(figsize=(12, 8))

                # Fitness curve
                plt.subplot(2, 2, 1)
                iterations_list = list(range(1, len(hill_climber.best_fitness_history) + 1))
                plt.plot(iterations_list, hill_climber.best_fitness_history, 'b-', linewidth=2, label='Best Fitness')
                plt.plot(iterations_list, hill_climber.current_fitness_history, 'g-', linewidth=1, alpha=0.5,
                         label='Current Fitness')
                plt.xlabel("Iteration")
                plt.ylabel("Fitness (Reward)")
                plt.title("Hill Climber Progress")
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Step size change
                plt.subplot(2, 2, 2)
                plt.plot(iterations_list, hill_climber.step_size_history, 'r-', linewidth=2)
                plt.xlabel("Iteration")
                plt.ylabel("Step Size")
                plt.title("Adaptive Step Size")
                plt.grid(True, alpha=0.3)
                plt.yscale('log')

                # Moving average
                plt.subplot(2, 2, 3)
                window = min(20, len(hill_climber.best_fitness_history))
                if window > 0:
                    moving_avg = np.convolve(hill_climber.best_fitness_history, np.ones(window) / window, mode='valid')
                    plt.plot(iterations_list[window - 1:], moving_avg, 'r-', linewidth=2,
                             label=f'Moving Avg (window={window})')
                    plt.xlabel("Iteration")
                    plt.ylabel("Moving Average Fitness")
                    plt.title("Trend Analysis")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                # Histogram: fitness improvements
                plt.subplot(2, 2, 4)
                if len(hill_climber.best_fitness_history) > 1:
                    improvements = np.diff(hill_climber.best_fitness_history)
                    positive_improvements = improvements[improvements > 0]
                    if len(positive_improvements) > 0:
                        plt.hist(positive_improvements, bins=20, alpha=0.7, color='green')
                        plt.xlabel("Improvement Amount")
                        plt.ylabel("Frequency")
                        plt.title("Positive Improvement Distribution")
                        plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig("hill_climber_progress.png", dpi=150)
                plt.close()
                print("Progress plot saved to hill_climber_progress.png")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        # Close log file
        log_file.close()

        # Final save
        np.save("final_hill_climber_solution.npy", hill_climber.best_solution.actions)
        print("Final best solution saved as final_hill_climber_solution.npy")

        # Plot final training curve
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        iterations_list = list(range(1, len(hill_climber.best_fitness_history) + 1))
        plt.plot(iterations_list, hill_climber.best_fitness_history, 'b-', linewidth=2, label='Best Fitness')
        plt.plot(iterations_list, hill_climber.current_fitness_history, 'g-', linewidth=1, alpha=0.3,
                 label='Current Fitness')
        plt.xlabel("Iteration")
        plt.ylabel("Fitness (Reward)")
        plt.title("Hill Climber Training History")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(iterations_list, hill_climber.step_size_history, 'r-', linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Step Size")
        plt.title("Step Size Adaptation")
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        plt.subplot(2, 2, 3)
        # Average over every 10 iterations
        group_size = 10
        if len(hill_climber.best_fitness_history) >= group_size:
            grouped_best = []
            grouped_avg = []
            grouped_iter = []
            for i in range(0, len(hill_climber.best_fitness_history), group_size):
                group_best = hill_climber.best_fitness_history[i:i + group_size]
                if len(group_best) > 0:
                    grouped_best.append(np.mean(group_best))
                    grouped_iter.append(i + group_size)
            plt.plot(grouped_iter, grouped_best, 'b-o', linewidth=2, label=f'Best (per {group_size} iters)')
            plt.xlabel("Iteration")
            plt.ylabel("Average Fitness")
            plt.title(f"Grouped Averages (every {group_size} iterations)")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        # Distribution statistics of the best solution
        plt.hist(hill_climber.best_solution.actions.flatten(), bins=30, alpha=0.7, color='purple')
        plt.xlabel("Δq Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Best Solution Actions")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("final_hill_climber_progress.png", dpi=150)
        plt.show()

        # Replay the best solution
        replay_best_solution(hill_climber.best_solution, target_cycles=2)

        print(f"\nHill Climber training complete!")
        print(f"Best fitness achieved: {hill_climber.best_solution.fitness:.3f}")
        print(f"Total iterations: {len(hill_climber.best_fitness_history)}")
        print(f"Final step size: {hill_climber.step_size:.6f}")


# -------------------------
# Main program
# -------------------------
if __name__ == "__main__":
    # Start hill climber training
    train_hill_climber()