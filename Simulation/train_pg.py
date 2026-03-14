# train_pg_improved_english.py
"""
Improved Policy Gradient for fine-tuning keyframe gait (output Δq)
- Added 3-second startup stabilization period
- Kept transition_steps=1000
- Improved IMU stability limits (penalty starts at 10°, termination at 40°)
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
import gc
import psutil

matplotlib.use('Agg')  # Use non-interactive backend to avoid font issues
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning)  # Suppress font warnings

# --------------------------
# Hyperparameters
# --------------------------
XML_PATH = "fufu.xml"

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Policy
HIDDEN = 64
LEARNING_RATE = 1e-3

# Training process
batch_size = 8  # Number of episodes per update
max_episode_steps = 100000  # Maximum steps per episode
update_every = batch_size  # Update frequency
gamma = 0.99  # Discount factor
grad_clip = 1.0

# Action constraints (Δq)
DELTA_Q_LIMIT = 0.25  # rad, limits

# IMU limits (improved)
ACC_CLIP = 20.0  # m/s², clip acceleration
GYRO_CLIP = 50.0  # rad/s, clip angular velocity
IMU_ACCEL_TERMINATE = 25.0  # ||accel|| > this value triggers termination
IMU_GYRO_TERMINATE = 40.0  # ||gyro|| > this value triggers termination

# Posture limits (improved)
MAX_ROLL_PITCH = 0.08725      # rad, ~5°, termination threshold (5°)
MAX_TILT = 0.0349            # rad, ~2°, penalty threshold (2°)
TILT_PENALTY_SCALE = 0.2     # Posture penalty coefficient

# Reward weights
WEIGHT_FORWARD = 1.0  # Forward velocity reward
WEIGHT_IMU_PENALTY = 0.5  # Penalty for IMU exceeding limits
WEIGHT_ENERGY = 1e-3  # Energy consumption penalty (τ * ω)
WEIGHT_JERK = 1e-4  # Jerk penalty

# Testing / Saving
model_save_path = "policy_checkpoint.pt"
test_interval = 10  # Test with visualization every N updates
test_duration_steps = 30000  # Steps during testing
render_on_test = True

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
# Policy network (Gaussian policy)
# -------------------------
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )
        # learnable log std per action
        self.logstd = nn.Parameter(torch.zeros(act_dim) - 1.0)

    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.logstd)
        return mu, std

    def get_action(self, obs):
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.forward(obs_t)
        mu = mu.squeeze(0).numpy()
        std = std.numpy()
        act = mu + std * np.random.randn(*mu.shape)
        return act, mu, std


# -------------------------
# Helper functions
# -------------------------
def check_coordinate_system():
    """Check robot's movement direction"""
    print("Checking coordinate system...")
    env = MujocoWalkEnv(XML_PATH, render=True)
    env.reset()

    # Let robot walk in place for a few seconds
    for i in range(500):
        obs, rew, done, info = env.step(np.zeros(len(joint_order)))
        if i % 100 == 0:
            pos, vel = env._get_base_pos_vel()
            print(
                f"Step {i}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
        env.render_frame()
        time.sleep(0.01)

    env.close()
    print("Coordinate system check complete")


def replay_best_episode(best_episode, best_forward):
    """Replay the best episode"""
    print(f"\nReplaying best episode, average reward={best_forward:.3f}")

    # Create new viewer environment
    env_view = MujocoWalkEnv(XML_PATH, render=False)
    from mujoco.viewer import launch_passive
    viewer = launch_passive(env_view.model, env_view.data)

    obs_seq, act_seq, rew_seq = best_episode
    env_view.reset()

    recorded_keyframes = []

    print("Starting replay...")
    # Replay actions
    for step_idx, act in enumerate(act_seq):
        # Get current keyframe target
        current_key = env_view.kf.anim_seq[env_view.kf.current_frame_index]
        kf_targets = env_view.kf.keyframes[current_key]

        for i, jname in enumerate(joint_order):
            actuator_id = env_view.actuator_ids[i]
            env_view.data.ctrl[actuator_id] = kf_targets[i] + float(act[i])

        # Step simulation
        mujoco.mj_step(env_view.model, env_view.data)
        viewer.sync()
        time.sleep(0.001)

        recorded_keyframes.append(act.copy())

    print("Replay complete, saving data...")
    # Save data
    np.save("best_episode_deltaq.npy", recorded_keyframes)
    df = pd.DataFrame(recorded_keyframes, columns=joint_order)
    df.to_csv("best_episode_deltaq.csv", index=False)
    print("Saved best episode data to best_episode_deltaq.npy and best_episode_deltaq.csv")

    # Infinite loop display (press Ctrl+C to exit)
    print("Press Ctrl+C to exit...")
    try:
        while True:
            viewer.sync()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Closing viewer...")
        env_view.close()


def test_policy(policy, steps=500, render=True):
    """Test policy"""
    print(f"Testing policy, steps={steps}")
    env = MujocoWalkEnv(XML_PATH, render=render)
    obs = env.reset()
    total_reward = 0.0
    rewards = []

    for t in range(steps):
        act, mu, std = policy.get_action(obs)
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

def check_memory(threshold_mb=2000):
    """Check system memory, and warn / clean if exceeded"""
    mem = psutil.virtual_memory()
    used_mb = mem.used / (1024 ** 2)
    if used_mb > threshold_mb:
        print(f"Memory usage high: {used_mb:.1f} MB, running garbage collection...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def replay_best_episode(best_episode, joint_order, keyframes, actuator_ids, model_xml, transition_steps):
    """Replay best episode and save Δq + full control qpos"""
    import mujoco
    from mujoco.viewer import launch_passive

    print("Launching viewer for best episode replay...")
    env_view = MujocoWalkEnv(model_xml, render=False)
    env_view.reset()

    obs_seq, act_seq, rew_seq = best_episode
    recorded_deltaq = []
    recorded_qpos = []

    viewer = launch_passive(env_view.model, env_view.data)

    for step_idx, deltaq in enumerate(act_seq):
        # 当前 keyframe
        current_key = env_view.kf.anim_seq[env_view.kf.current_frame_index]
        kf_targets = env_view.kf.keyframes[current_key]

        full_q = []
        for i, jname in enumerate(joint_order):
            actuator_id = actuator_ids[i]
            ctrl_value = kf_targets[i] + float(deltaq[i])
            env_view.data.ctrl[actuator_id] = ctrl_value
            full_q.append(ctrl_value)

        # Step simulation
        mujoco.mj_step(env_view.model, env_view.data)
        viewer.sync()
        recorded_deltaq.append(deltaq.copy())
        recorded_qpos.append(full_q.copy())
        time.sleep(0.001)

    # Save Δq
    np.save("best_episode_deltaq.npy", np.array(recorded_deltaq))
    pd.DataFrame(recorded_deltaq, columns=joint_order).to_csv("best_episode_deltaq.csv", index=False)
    # Save full qpos
    np.save("best_episode_qpos.npy", np.array(recorded_qpos))
    pd.DataFrame(recorded_qpos, columns=joint_order).to_csv("best_episode_qpos.csv", index=False)

    print("Best episode saved: Δq + full qpos.")
    print("Press Ctrl+C to exit viewer.")
    try:
        while True:
            viewer.sync()
            time.sleep(0.01)
    except KeyboardInterrupt:
        env_view.close()
        print("Viewer closed.")
# -------------------------
# Main training loop
# -------------------------
def train():
    """Main training function with memory protection and final replay"""
    print("=" * 50)
    print("Starting Policy Gradient Training")
    print(f"Batch size={batch_size}, lr={LEARNING_RATE}, max_steps={max_episode_steps}")
    print(f"IMU limits: penalty at >2° tilt, termination at >5° tilt")
    print(f"Startup delay: 3 seconds")
    print("=" * 50)

    # Initialize environment
    env = MujocoWalkEnv(XML_PATH, render=False, start_delay=3.0)
    obs0 = env.reset()
    obs_dim = obs0.size
    act_dim = len(joint_order)

    policy = MLPPolicy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    # Training stats
    forward_speed_history = []
    episode_counts = []
    episode_count = 0
    update_count = 0

    # Save best episode
    best_episode = None
    best_forward = -1e9

    log_file = open("training_log.txt", "w")
    log_file.write("Episode,Reward,Steps,Tilt\n")

    try:
        while True:
            batch_obs, batch_acts, batch_log_probs, batch_rews = [], [], [], []

            for ep in range(batch_size):
                obs = env.reset()
                ep_obs, ep_acts, ep_log_probs, ep_rews, ep_info = [], [], [], [], []

                for t in range(max_episode_steps):
                    act, mu, std = policy.get_action(obs)
                    act = np.clip(act, -DELTA_Q_LIMIT, DELTA_Q_LIMIT)

                    # log_prob
                    obs_t = torch.from_numpy(obs.astype(np.float32))
                    mu_t, std_t = policy.forward(obs_t.unsqueeze(0))
                    mu_t, std_t = mu_t.squeeze(0), std_t.squeeze(0)
                    dist = torch.distributions.Normal(mu_t, std_t)
                    act_t = torch.from_numpy(act.astype(np.float32))
                    log_prob = dist.log_prob(act_t).sum()

                    next_obs, reward, done, info = env.step(act)

                    ep_obs.append(obs.copy())
                    ep_acts.append(act.copy())
                    ep_log_probs.append(log_prob.item())
                    ep_rews.append(reward)
                    ep_info.append(info)

                    obs = next_obs
                    if done:
                        break

                # Discounted rewards
                discounted_rewards = []
                R = 0
                for r in reversed(ep_rews):
                    R = r + gamma * R
                    discounted_rewards.insert(0, R)
                discounted_rewards = np.array(discounted_rewards)
                if len(discounted_rewards) > 1:
                    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

                # Batch accumulate
                batch_obs.extend(ep_obs)
                batch_acts.extend(ep_acts)
                batch_log_probs.extend(ep_log_probs)
                batch_rews.extend(discounted_rewards)

                # Stats
                episode_count += 1
                mean_reward = abs(np.mean(ep_rews))
                mean_tilt = np.mean([info.get('tilt_deg', 0) for info in ep_info]) if ep_info else 0

                log_file.write(f"{episode_count},{mean_reward:.3f},{len(ep_rews)},{mean_tilt:.1f}\n")
                log_file.flush()

                if mean_reward > best_forward:
                    best_forward = mean_reward
                    best_episode = (ep_obs, ep_acts, ep_rews)
                    print(f"New best episode: {episode_count}, reward={best_forward:.3f}")

                episode_counts.append(episode_count)
                forward_speed_history.append(mean_reward)

                if episode_count % 10 == 0:
                    check_memory(threshold_mb=2000)

            # Policy update
            update_count += 1
            batch_obs_t = torch.from_numpy(np.array(batch_obs)).float()
            batch_acts_t = torch.from_numpy(np.array(batch_acts)).float()
            batch_rews_t = torch.tensor(batch_rews).float()

            optimizer.zero_grad()
            mu, std = policy(batch_obs_t)
            dist = torch.distributions.Normal(mu, std)
            log_probs = dist.log_prob(batch_acts_t).sum(dim=-1)
            loss = -(log_probs * batch_rews_t).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

            print(f"Update {update_count}: loss={loss.item():.3f}, avg_reward={np.mean(forward_speed_history[-batch_size:]):.3f}")

            # Periodic model save
            if update_count % test_interval == 0:
                torch.save({
                    'policy_state': policy.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'update_count': update_count,
                    'best_reward': best_forward,
                    'forward_history': forward_speed_history
                }, model_save_path)
                print(f"Model saved to {model_save_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        log_file.close()
        torch.save(policy.state_dict(), "final_policy.pt")
        print("Final model saved as final_policy.pt")

        # Replay best episode (full animation + save Δq + qpos)
        if best_episode is not None:
            replay_best_episode(best_episode, joint_order, keyframes_radians,
                                env.actuator_ids, XML_PATH, transition_steps)
        print("Training complete!")


# -------------------------
# Main program
# -------------------------
if __name__ == "__main__":
    # Optional: run coordinate system check first
    # check_coordinate_system()

    # Start training
    train()