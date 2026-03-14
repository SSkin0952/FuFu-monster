# -*- coding: utf-8 -*-
from math import sin, cos, pi
from pylx16a.lx16a import *
import time
import threading
import os


# Initialize serial connection first
try:
    print("Initializing /dev/ttyUSB0...")
    LX16A.initialize("/dev/ttyUSB0", 0.1)
    print("/dev/ttyUSB0 initialized successfully")
except Exception as e:
    print(f"Failed to initialize COM3: {e}")
    print("Please check:")
    print("1. /dev/ttyUSB0 port connection")
    print("2. USB cable")
    print("3. Driver installation")
    quit()

# Servo ID assignment - now 8 servos
servo_ids = [1, 2, 3, 4, 5, 6, 7, 8]
servos = []

print("Testing servo connections...")

# Test each servo connection individually
for servo_id in servo_ids:
    try:
        print(f"Testing servo {servo_id}...", end=" ")
        servo = LX16A(servo_id)

        # Test reading current angle
        angle = servo.get_physical_angle()
        print(f"OK - Current angle: {angle} deg")

        # Set angle limits, specific limits for hair servos
        if servo_id in [7, 8]:
            if servo_id == 7:
                servo.set_angle_limits(57, 109.2)
            else:  # servo_id == 8
                servo.set_angle_limits(55, 106)
        else:
            servo.set_angle_limits(0, 240)

        servos.append(servo)

    except ServoTimeoutError as e:
        print(f"Timeout - Servo {servo_id} not responding")
        servos.append(None)
    except Exception as e:
        print(f"Error with servo {servo_id}: {e}")
        servos.append(None)

successful_servos = sum(1 for s in servos if s is not None)
print(f"\nSuccessfully connected to {successful_servos} out of {len(servo_ids)} servos")

if successful_servos == 0:
    print("No servos responding. Exiting...")
    quit()

# Updated gait positions p1 to p7 (walking cycle)
GAIT_POSITIONS = [
    # p1 - Standing (initial position)
    {1: 117.36, 2: 118.56, 3: 143.04, 4: 124.56, 5: 130.32, 6: 138.24, 7: 109.2, 8: 106},
    # p2 - Right leg steps out (safe stop point)
    {1: 89.86, 2: 118.56, 3: 110.54, 4: 124.56, 5: 125.32, 6: 138.24, 7: 109.2, 8: 106},
    # p3 - Right leg continues, left leg back
    {1: 94.86, 2: 123.56, 3: 123.04, 4: 151.56, 5: 127.82, 6: 160.24, 7: 81.2, 8: 78},
    # p4 - Transition phase
    {1: 99.36, 2: 138.06, 3: 111.54, 4: 184.56, 5: 116.82, 6: 178.74, 7: 87.2, 8: 84},
    # p5 - Left leg forward
    {1: 117.36, 2: 157.56, 3: 143.04, 4: 194.56, 5: 130.32, 6: 169.24, 7: 109.2, 8: 106},
    # p6 - Transition phase
    {1: 103.36, 2: 154.56, 3: 98.04, 4: 154.56, 5: 99.32, 6: 139.24, 7: 81.2, 8: 78},
    # p7 - Return phase
    {1: 97.86, 2: 136.56, 3: 83.04, 4: 156.06, 5: 89.82, 6: 151.74, 7: 87.2, 8: 84}
]

STANDING_POSITION = GAIT_POSITIONS[0]
SAFE_STOP_POSITIONS = [2, 5]  # p2 is safe positions to stop immediately

# Hair servo parameters
HAIR_LEFT_RANGE = (50, 115)
HAIR_RIGHT_RANGE = (48, 112)


def clamp_angle(angle, min_angle=0, max_angle=240):
    """Clamp angle to valid servo range"""
    return max(min(angle, max_angle), min_angle)


def smooth_interpolate(start, end, progress):
    """Smooth interpolation function"""
    # Smooth step function for more natural movement
    smooth_progress = progress * progress * (3 - 2 * progress)
    return start + (end - start) * smooth_progress


def interpolate_gait(current_gait, next_gait, progress):
    """Interpolate between two gait positions"""
    result = {}
    for servo_id in range(1, 9):
        if servo_id in current_gait and servo_id in next_gait:
            start_angle = current_gait[servo_id]
            end_angle = next_gait[servo_id]
            interpolated_angle = smooth_interpolate(start_angle, end_angle, progress)
            result[servo_id] = clamp_angle(interpolated_angle)
    return result


def move_to_position(position_dict, duration=2.0):
    """Move all servos to specified position smoothly"""
    steps = int(duration * 20)  # 20 steps per second
    if steps < 1:
        steps = 1

    step_delay = duration / steps

    # Get current positions
    current_positions = {}
    for i, servo in enumerate(servos):
        if servo is not None:
            servo_id = servo_ids[i]
            try:
                current_positions[servo_id] = servo.get_physical_angle()
            except:
                current_positions[servo_id] = position_dict.get(servo_id, 90)

    # Smooth movement
    for step in range(steps + 1):
        progress = step / steps
        intermediate_pos = interpolate_gait(current_positions, position_dict, progress)

        for i, servo in enumerate(servos):
            if servo is not None:
                servo_id = servo_ids[i]
                try:
                    angle = intermediate_pos.get(servo_id, 90)
                    servo.move(angle)
                except Exception as e:
                    print(f"Movement error servo {servo_id}: {e}")

        time.sleep(step_delay)


def calculate_walking_gait(t, walking_speed=0.2):
    """
    Calculate gait position based on time for continuous walking
    Uses p1 to p7 as keyframes and loops from p2 to p7 (p1 is only for start/stop)
    """
    # Normalize time to gait cycle (0 to 1), starting from p2
    cycle_progress = (t * walking_speed) % 1.0

    # Map cycle progress to gait positions (p2 to p7 and back to p2)
    num_walking_positions = len(GAIT_POSITIONS) - 1  # p2 to p7
    position_index = cycle_progress * num_walking_positions

    current_index = int(position_index) % num_walking_positions + 1  # +1 to start from p2
    next_index = (current_index + 1) % len(GAIT_POSITIONS)
    if next_index == 0:  # If next is p1, skip to p2
        next_index = 1

    interpolation_progress = position_index - int(position_index)

    # Interpolate between current and next position
    current_gait = GAIT_POSITIONS[current_index]
    next_gait = GAIT_POSITIONS[next_index]

    return interpolate_gait(current_gait, next_gait, interpolation_progress), cycle_progress, current_index


def get_gait_phase_description(position_index):
    """Get description of current gait phase"""
    phases = {
        1: "Standing (p1)",
        2: "Right leg steps out (p2) - SAFE STOP",
        3: "Right forward, left back (p3)",
        4: "Transition phase (p4)",
        5: "Left leg forward (p5) - SAFE STOP",
        6: "Transition phase (p6)",
        7: "Return phase (p7)"
    }
    return phases.get(position_index, f"Position {position_index}")


def print_gait_info(gait_position, cycle_progress, position_index):
    """Print information about current gait position"""
    phase_desc = get_gait_phase_description(position_index)
    safe_stop = " [SAFE STOP]" if position_index in SAFE_STOP_POSITIONS else ""
    print(f"Cycle: {cycle_progress * 100:.1f}% - {phase_desc}{safe_stop}")
    print(f"  Left:  Hip={gait_position[1]:.1f} deg, Knee={gait_position[3]:.1f} deg, Ankle={gait_position[5]:.1f} deg")
    print(f"  Right: Hip={gait_position[2]:.1f} deg, Knee={gait_position[4]:.1f} deg, Ankle={gait_position[6]:.1f} deg")
    print(f"  Hair:  L={gait_position[7]:.1f} deg, R={gait_position[8]:.1f} deg")


def complete_walking_cycle_to_safe_stop(current_position_index):
    """
    Continue walking cycle until reaching p2, then return to p1.
    (Keeps the gait smooth instead of jumping directly.)
    """
    print(f"Completing walking cycle from p{current_position_index} to p2, then to p1...")

    total_positions = len(GAIT_POSITIONS)
    positions_to_move = []

    # ?????????,???? p2(? index == 1)
    next_index = current_position_index
    while True:
        next_index = (next_index % total_positions) + 1  # ???? p1?p2?...?p7?p1
        positions_to_move.append(next_index)
        if next_index == 1:  # ?? p2(??p2? index=1)
            break

    # ??????????
    for pos_index in positions_to_move:
        print(f"Moving to p{pos_index}...")
        move_to_position(GAIT_POSITIONS[pos_index], 0.5)
        time.sleep(0.1)

    # ??? p2 ? p1(??)
    print("Returning to standing position p1...")
    move_to_position(STANDING_POSITION, 1.0)


# Move to standing position first
print("\nMoving to standing position (p1)...")
move_to_position(STANDING_POSITION, 3.0)
time.sleep(1)

print("\nStarting continuous walking gait (p2 to p7 loop)...")
print("Press Ctrl+C to stop and return to standing position")
print("Safe stop points: p2 and p5 (immediate stop)")
print("Other positions will complete cycle to next safe stop")

t = 0
frame_count = 0
walking_speed = 0.15  # Adjust this to change walking speed
current_position_index = 1  # Start from p1

def get_rpi_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_str = f.readline().strip()
        return float(temp_str) / 1000.0
    except:
        return None


def health_check_routine(interval=5.0):

    while True:
        try:
            print("\n[Health Check] -----------------------------")
            for i, servo in enumerate(servos):
                if servo is None:
                    continue
                servo_id = servo_ids[i]
                try:
                    temp = servo.get_temp()
                    vin = servo.get_vin() / 1000.0  # mV → V
                    warn = []
                    if temp > 60:
                        warn.append("⚠️ HIGH TEMP")
                    if vin < 6.0:
                        warn.append("⚠️ LOW VOLTAGE")
                    warn_str = " ".join(warn) if warn else "OK"
                    print(f"Servo {servo_id}: Temp={temp:.1f}°C | Vin={vin:.2f}V {warn_str}")
                except Exception as e:
                    print(f"Servo {servo_id}: ❌ Read failed ({e})")

            rpi_temp = get_rpi_temperature()
            if rpi_temp:
                rpi_warn = "⚠️ HIGH CPU TEMP" if rpi_temp > 70 else "OK"
                print(f"Raspberry Pi CPU Temp: {rpi_temp:.1f}°C {rpi_warn}")
            else:
                print("Raspberry Pi temperature read failed")

            print("--------------------------------------------\n")

        except Exception as e:
            print(f"[Health Monitor Error] {e}")

        time.sleep(interval)

monitor_thread = threading.Thread(target=health_check_routine, args=(5.0,), daemon=True)
monitor_thread.start()
print("\n✅ Health monitoring thread started (every 5s)")

# =====================  Walking Routine =====================

t = 0
frame_count = 0
walking_speed = 0.15
current_position_index = 1

try:
    print("Moving to p2 to start walking cycle...")
    move_to_position(GAIT_POSITIONS[1], 1.0)
    current_position_index = 2

    while True:
        current_gait, cycle_progress, position_index = calculate_walking_gait(t, walking_speed)
        current_position_index = position_index

        for i, servo in enumerate(servos):
            if servo is not None:
                servo_id = servo_ids[i]
                try:
                    servo.move(current_gait.get(servo_id, 90))
                except Exception as e:
                    print(f"Movement error servo {servo_id}: {e}")

        frame_count += 1
        if frame_count % 25 == 0:
            print(f"\nFrame {frame_count}:")
            print_gait_info(current_gait, cycle_progress, position_index)

        time.sleep(0.05)
        t += 0.1

except KeyboardInterrupt:
    print(f"\n\nWalking interrupted by user at p{current_position_index}")
    if current_position_index in SAFE_STOP_POSITIONS:
        move_to_position(STANDING_POSITION, 1.5)
    else:
        complete_walking_cycle_to_safe_stop(current_position_index)
        move_to_position(STANDING_POSITION, 1.5)

finally:
    print("\nEnsuring standing position (p1)...")
    move_to_position(STANDING_POSITION, 1.0)
    print("✅ Successfully returned to standing position (p1)")
    print("Program exited.")