import mujoco
import mujoco.viewer
import time
import numpy as np
import math
import pandas as pd

model = mujoco.MjModel.from_xml_path("fufu.xml")
data = mujoco.MjData(model)

print("=== Keyframe Animation Control + IMU Monitoring ===")

# Keyframe data (converted to radians)
keyframes_degrees = {
    'o1': [0, 0, 0, 0, 0, 0, 0, 0],  # Initial standing pose
    'o2': [-27.5, 0, 32.5, 0, -5, 0, 0, 0],
    'o3': [-22.5, -5, 20, 27, 2.5, -22, 28, -28],
    'o4': [-18, -19.5, 31.5, 60, -13.5, -40.5, 22, -22],
    'o5': [0, -39, 0, 70, 0, -31, 0, 0],
    'o6': [-14, -36, 45, 30, -31, 6, 28, -28],
    'o7': [-19.5, -18, 60, 31.5, -40.5, -13.5, 22, -22]
}

# Convert degrees to radians
keyframes_radians = {}
for key, degrees in keyframes_degrees.items():
    keyframes_radians[key] = [math.radians(angle) for angle in degrees]

# Joint order (matching Excel column order)
joint_order = [
    'right_hip_joint', 'left_hip_joint',
    'right_knee_joint', 'left_knee_joint',
    'right_ankle_joint', 'left_ankle_joint',
    'right_hair_joint', 'left_hair_joint'
]

# Animation sequence
animation_sequence = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7']

# Animation parameters
start_delay = 0.0
frame_duration = 0.05
transition_steps = 1000


class KeyframeAnimator:
    def __init__(self, keyframes, joint_order):
        self.keyframes = keyframes
        self.joint_order = joint_order
        self.current_frame_index = 0
        self.step_count = 0
        self.transition_progress = 0
        self.start_time = None
        self.started = False
        self.total_steps = 0
        self.complete_cycles = 0

    def update_start(self):
        if self.start_time is None:
            self.start_time = time.time()

        if time.time() - self.start_time >= start_delay:
            self.started = True

    def get_current_targets(self):
        if not self.started:
            return dict(zip(self.joint_order, self.keyframes['o1']))

        current_key = animation_sequence[self.current_frame_index]

        if self.current_frame_index == len(animation_sequence) - 1:
            next_key = 'o2'
        else:
            next_key = animation_sequence[self.current_frame_index + 1]

        current_frame = self.keyframes[current_key]
        next_frame = self.keyframes[next_key]

        if self.transition_progress < transition_steps:
            alpha = self.transition_progress / transition_steps
            alpha_ease = self.ease_in_out(alpha)
            targets = []
            for i in range(len(current_frame)):
                interpolated = current_frame[i] + (next_frame[i] - current_frame[i]) * alpha_ease
                targets.append(interpolated)
        else:
            targets = next_frame

        self.step_count += 1
        self.total_steps += 1
        self.transition_progress += 1

        if self.transition_progress >= transition_steps:
            self.transition_progress = 0
            self.current_frame_index += 1
            if self.current_frame_index >= len(animation_sequence):
                self.current_frame_index = 1
                self.complete_cycles += 1

        return dict(zip(self.joint_order, targets))

    def ease_in_out(self, t):
        return t * t * (3 - 2 * t)

    def get_status(self):
        if not self.started:
            return "Waiting to start"

        current_key = animation_sequence[self.current_frame_index]
        progress_percent = self.transition_progress / transition_steps * 100
        return f"Current frame: {current_key} ({self.current_frame_index + 1}/{len(animation_sequence)}), " \
               f"Transition progress: {progress_percent:.1f}%"


# ============================================================
# IMU monitoring class
# ============================================================
class IMUMonitor:
    def __init__(self):
        self.accel_history = []
        self.gyro_history = []
        self.quat_history = []
        self.tilt_history = []

        # Statistics
        self.accel_norm_history = []
        self.gyro_norm_history = []

        # Get sensor IDs
        self.accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        self.gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")

    def read_imu(self, data):
        """Read IMU data"""
        accel = data.sensordata[model.sensor_adr[self.accel_id]:
                                model.sensor_adr[self.accel_id] + 3].copy()

        gyro = data.sensordata[model.sensor_adr[self.gyro_id]:
                               model.sensor_adr[self.gyro_id] + 3].copy()

        quat = data.sensordata[model.sensor_adr[self.quat_id]:
                               model.sensor_adr[self.quat_id] + 4].copy()

        return accel, gyro, quat

    def calculate_tilt(self, quat):
        """Calculate body tilt angle relative to vertical"""
        w, x, y, z = quat
        # 计算 roll 和 pitch
        tilt_rad = np.arccos(min(1.0, max(-1.0, 1 - 2 * (x * x + y * y))))
        tilt_deg = math.degrees(tilt_rad)
        return tilt_deg, tilt_rad

    def update(self, data):
        accel, gyro, quat = self.read_imu(data)
        tilt_deg, tilt_rad = self.calculate_tilt(quat)

        accel_norm = np.linalg.norm(accel)
        gyro_norm = np.linalg.norm(gyro)

        self.accel_history.append(accel)
        self.gyro_history.append(gyro)
        self.quat_history.append(quat)
        self.tilt_history.append(tilt_deg)
        self.accel_norm_history.append(accel_norm)
        self.gyro_norm_history.append(gyro_norm)

        return {
            'accel': accel,
            'gyro': gyro,
            'quat': quat,
            'accel_norm': accel_norm,
            'gyro_norm': gyro_norm,
            'tilt_deg': tilt_deg,
            'tilt_rad': tilt_rad
        }

    def get_statistics(self):
        if len(self.accel_norm_history) == 0:
            return None

        accel_norms = np.array(self.accel_norm_history)
        gyro_norms = np.array(self.gyro_norm_history)
        tilts = np.array(self.tilt_history)

        stats = {
            'accel_norm': {
                'mean': np.mean(accel_norms),
                'std': np.std(accel_norms),
                'min': np.min(accel_norms),
                'max': np.max(accel_norms),
                'p95': np.percentile(accel_norms, 95),
                'p99': np.percentile(accel_norms, 99)
            },
            'gyro_norm': {
                'mean': np.mean(gyro_norms),
                'std': np.std(gyro_norms),
                'min': np.min(gyro_norms),
                'max': np.max(gyro_norms),
                'p95': np.percentile(gyro_norms, 95),
                'p99': np.percentile(gyro_norms, 99)
            },
            'tilt': {
                'mean': np.mean(tilts),
                'std': np.std(tilts),
                'min': np.min(tilts),
                'max': np.max(tilts),
                'p95': np.percentile(tilts, 95),
                'p99': np.percentile(tilts, 99)
            }
        }

        return stats

    def print_statistics(self):
        stats = self.get_statistics()
        if stats is None:
            print("No data collected yet")
            return

        print("\n" + "=" * 70)
        print("=== IMU Statistics (based on {} samples) ===".format(len(self.accel_norm_history)))
        print("=" * 70)

        print("\n[Acceleration Norm (m/s²)]")
        print(f"Mean: {stats['accel_norm']['mean']:.3f}")
        print(f"Std: {stats['accel_norm']['std']:.3f}")
        print(f"Min: {stats['accel_norm']['min']:.3f}")
        print(f"Max: {stats['accel_norm']['max']:.3f}")
        print(f"95th percentile: {stats['accel_norm']['p95']:.3f}")
        print(f"99th percentile: {stats['accel_norm']['p99']:.3f}")

        print("\n[Gyro Norm (rad/s)]")
        print(f"Mean: {stats['gyro_norm']['mean']:.3f}")
        print(f"Std: {stats['gyro_norm']['std']:.3f}")
        print(f"Min: {stats['gyro_norm']['min']:.3f}")
        print(f"Max: {stats['gyro_norm']['max']:.3f}")
        print(f"95th percentile: {stats['gyro_norm']['p95']:.3f}")
        print(f"99th percentile: {stats['gyro_norm']['p99']:.3f}")

        print("\n[Tilt Angle (deg)]")
        print(f"Mean: {stats['tilt']['mean']:.2f}°")
        print(f"Std: {stats['tilt']['std']:.2f}°")
        print(f"Min: {stats['tilt']['min']:.2f}°")
        print(f"Max: {stats['tilt']['max']:.2f}°")
        print(f"95th percentile: {stats['tilt']['p95']:.2f}°")
        print(f"99th percentile: {stats['tilt']['p99']:.2f}°")

        print("\n" + "=" * 70)
        print("Recommended RL Limits")
        print("=" * 70)
        print(f"ACC_CLIP = {stats['accel_norm']['p99'] * 1.2:.1f}")
        print(f"GYRO_CLIP = {stats['gyro_norm']['p99'] * 1.2:.1f}")
        print(f"IMU_ACCEL_TERMINATE = {stats['accel_norm']['max'] * 1.5:.1f}")
        print(f"IMU_GYRO_TERMINATE = {stats['gyro_norm']['max'] * 1.5:.1f}")
        print(f"MAX_TILT = {math.radians(stats['tilt']['p95']):.4f}")
        print(f"MAX_ROLL_PITCH = {math.radians(stats['tilt']['max'] * 1.5):.4f}")
        print("=" * 70 + "\n")

    def save_to_csv(self, filename='imu_data.csv'):
        df = pd.DataFrame({
            'accel_x': [a[0] for a in self.accel_history],
            'accel_y': [a[1] for a in self.accel_history],
            'accel_z': [a[2] for a in self.accel_history],
            'gyro_x': [g[0] for g in self.gyro_history],
            'gyro_y': [g[1] for g in self.gyro_history],
            'gyro_z': [g[2] for g in self.gyro_history],
            'quat_w': [q[0] for q in self.quat_history],
            'quat_x': [q[1] for q in self.quat_history],
            'quat_y': [q[2] for q in self.quat_history],
            'quat_z': [q[3] for q in self.quat_history],
            'accel_norm': self.accel_norm_history,
            'gyro_norm': self.gyro_norm_history,
            'tilt_deg': self.tilt_history
        })
        df.to_csv(filename, index=False)
        print(f"IMU data save to {filename}")

class KeyframeRecorder:
    def __init__(self, joint_order, record_interval=200):
        self.joint_order = joint_order
        self.record_interval = record_interval
        self.records = {}
        self.current_cycle_step = 0
        self.last_cycle_index = None
        self.cycle_count = 0

        self.joint_qpos_addr = {}
        for jname in joint_order:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self.joint_qpos_addr[jname] = model.jnt_qposadr[jid]

    def update(self, animator, data):

        if self.last_cycle_index is None:
            self.last_cycle_index = animator.current_frame_index

        if animator.current_frame_index == 1 and self.last_cycle_index != 1:
            print("\n=== Start new cycle ===")
            self.cycle_count += 1
            self.current_cycle_step = 0

        self.last_cycle_index = animator.current_frame_index

        self.current_cycle_step += 1

        if self.current_cycle_step % self.record_interval == 0:
            key = f"t{self.current_cycle_step}"
            self.records[key] = self.read_joint_angles(data)
            print(f"[key frame] {key}: {self.records[key]}")

    def read_joint_angles(self, data):
        angles_deg = []
        for jname in self.joint_order:
            addr = self.joint_qpos_addr[jname]
            rad = data.qpos[addr]
            angles_deg.append(math.degrees(rad))
        return angles_deg

    def save(self, filename="recorded_keyframes.csv"):
        """Save as CSV"""
        df = pd.DataFrame(self.records).T
        df.columns = self.joint_order
        df.to_csv(filename)
        print(f"\nKey record to {filename}")
animator = KeyframeAnimator(keyframes_radians, joint_order)
imu_monitor = IMUMonitor()
keyframe_recorder = KeyframeRecorder(joint_order, record_interval=200)
filtered_targets = {}
tau = 0.09
dt = model.opt.timestep
alpha = math.exp(-dt / tau)

for joint_name in joint_order:
    filtered_targets[joint_name] = keyframes_radians['o1'][joint_order.index(joint_name)]

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

try:
    viewer = mujoco.viewer.launch_passive(model, data)
    print("Viewer launched successfully")
    print(f"Startup delay: {start_delay} seconds...")
    print("Press Ctrl+C to stop")

    initial_targets = dict(zip(joint_order, keyframes_radians['o1']))
    for joint_name, target in initial_targets.items():
        actuator_name = actuator_mapping[joint_name]
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id != -1:
            data.ctrl[actuator_id] = target

    print("Initializing standing pose...")
    for i in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    # interval for printing runtime status
    print_interval = 5.0
    last_print_time = time.time()
    start_countdown = time.time()
    program_start_time = time.time()

    print("\n=== Starting animation + IMU monitoring ===")
    print(f"Runtime status will be displayed every {print_interval} seconds\n")

    while True:
        current_time = time.time()

        # update animator state
        animator.update_start()

        # startup countdown
        if not animator.started:
            remaining_time = start_delay - (current_time - start_countdown)
            if remaining_time > 0 and int(remaining_time) != int(last_print_time):
                print(f"Countdown to start: {int(remaining_time) + 1} seconds")
                last_print_time = remaining_time

        # get current target pose
        targets = animator.get_current_targets()

        # apply control
        for joint_name, target_cmd in targets.items():
            prev = filtered_targets[joint_name]
            target_filtered = alpha * prev + (1 - alpha) * target_cmd
            filtered_targets[joint_name] = target_filtered

            actuator_name = actuator_mapping[joint_name]
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id != -1:
                data.ctrl[actuator_id] = target_filtered

        # simulation step
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)

        # ===== Update IMU data =====
        imu_data = imu_monitor.update(data)

        if animator.started:
            keyframe_recorder.update(animator, data)

        # print real-time IMU status every 1000 steps
        if animator.step_count % 1000 == 0 and animator.started:
            print("\n=== Real-time IMU Data ===")
            print(
                f"Acceleration: [{imu_data['accel'][0]:+.3f}, {imu_data['accel'][1]:+.3f}, {imu_data['accel'][2]:+.3f}] m/s²")
            print(f"  Norm: {imu_data['accel_norm']:.3f} m/s²")
            print(f"Angular velocity: [{imu_data['gyro'][0]:+.3f}, {imu_data['gyro'][1]:+.3f}, {imu_data['gyro'][2]:+.3f}] rad/s")
            print(f"  Norm: {imu_data['gyro_norm']:.3f} rad/s")
            print(f"Tilt angle: {imu_data['tilt_deg']:.2f}°")

        # ===== periodic runtime status report =====
        if current_time - last_print_time >= print_interval:

            elapsed_time = current_time - program_start_time
            steps_per_second = animator.total_steps / elapsed_time if elapsed_time > 0 else 0

            print("\n" + "=" * 50)
            print("=== Runtime Status Report ===")
            print(f"Runtime: {elapsed_time:.1f} seconds")
            print(f"Total steps: {animator.total_steps}")
            print(f"Simulation frequency: {steps_per_second:.1f} steps/s")
            print(f"Completed cycles: {animator.complete_cycles}")
            print(f"Animation status: {animator.get_status()}")
            print("=" * 50)

            # print IMU statistics
            imu_monitor.print_statistics()

            last_print_time = current_time

except KeyboardInterrupt:
    print("\n" + "=" * 50)
    print("=== Animation stopped ===")

    elapsed_time = time.time() - program_start_time

    print(f"Total runtime: {elapsed_time:.1f} seconds")
    print(f"Total steps: {animator.total_steps}")
    print(f"Completed cycles: {animator.complete_cycles}")
    print("=" * 50)

    # print final IMU statistics
    imu_monitor.print_statistics()

    # save collected data
    imu_monitor.save_to_csv('imu_walking_data.csv')
    keyframe_recorder.save("walking_cycle_keyframes.csv")

finally:
    if 'viewer' in locals():
        viewer.close()

print(f"\nAnimation finished, {animator.complete_cycles} full cycles executed")