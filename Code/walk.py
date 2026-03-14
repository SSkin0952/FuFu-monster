# -*- coding: utf-8 -*-
from math import sin, cos, pi
from pylx16a.lx16a import *
import time

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

# Complete gait positions p1 to p10
GAIT_POSITIONS = [
    # p1 - Standing (initial position)
    {1: 117.36, 2: 118.56, 3: 143.04, 4: 124.56, 5: 130.32, 6: 138.24, 7: 109.2, 8: 106},
    # p2 - Right leg steps out (safe stop point)
    {1: 89.86, 2: 118.56, 3: 110.54, 4: 124.56, 5: 125.32, 6: 138.24, 7: 109.2, 8: 106},
    # p3 - Right leg continues, left leg back
    {1: 94.86, 2: 123.56, 3: 123.04, 4: 151.56, 5: 127.82, 6: 160.24, 7: 81.2, 8: 78},
    # p4 - Transition phase
    {1: 99.36, 2: 138.06, 3: 111.54, 4: 184.56, 5: 116.82, 6: 178.74, 7: 87.2, 8: 84},
    # p5 - Left leg forward (safe stop point)
    {1: 117.36, 2: 157.56, 3: 143.04, 4: 194.56, 5: 130.32, 6: 169.24, 7: 109.2, 8: 106},
    # p6 - Transition phase
    {1: 103.36, 2: 154.56, 3: 98.04, 4: 154.56, 5: 99.32, 6: 139.24, 7: 81.2, 8: 78},
    # p7 - Return phase
    {1: 97.86, 2: 136.56, 3: 83.04, 4: 156.06, 5: 89.82, 6: 151.74, 7: 87.2, 8: 84}
]

STANDING_POSITION = GAIT_POSITIONS[0]


def clamp_angle(angle, min_angle=0, max_angle=240):
    """Clamp angle to valid servo range"""
    return max(min(angle, max_angle), min_angle)


def smooth_interpolate(start, end, progress):
    """Smooth interpolation function"""
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


def move_to_position(position_dict, position_name, duration=2.0):
    """Move all servos to specified position smoothly"""
    print(f"\nMoving to {position_name}...")

    steps = int(duration * 20)
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

    print(f"✅ Reached {position_name}")


def print_position_info(position_dict, position_name):
    """Print detailed information about the current position"""
    print(f"\n=== {position_name} ===")
    print("Servo Angles:")
    print(
        f"  Left Leg:  Hip={position_dict[1]:.1f} deg, Knee={position_dict[3]:.1f} deg, Ankle={position_dict[5]:.1f} deg")
    print(
        f"  Right Leg: Hip={position_dict[2]:.1f} deg, Knee={position_dict[4]:.1f} deg, Ankle={position_dict[6]:.1f} deg")
    print(f"  Hair:      Left={position_dict[7]:.1f} deg, Right={position_dict[8]:.1f} deg")


def wait_for_enter(prompt="Press Enter to continue to next position..."):
    """Wait for user to press Enter"""
    input(prompt)


def get_position_description(position_index):
    """Get description of each position"""
    descriptions = {
        1: "STANDING POSITION - Both feet on ground",
        2: "RIGHT LEG STEPPING OUT - Right hip forward, left knee bent",
        3: "RIGHT LEG CONTINUES - Right forward, left back, hair lowered",
        4: "RIGHT LEG MAX FORWARD - Right leg fully extended forward",
        5: "TRANSITION PHASE - Preparing for left leg movement",
        6: "LEFT LEG STARTS FORWARD - Left hip begins forward motion",
        7: "LEFT LEG CONTINUES - Left leg moving forward",
        8: "LEFT LEG FORWARD - Left forward, right back, hair lowered",
        9: "LEFT LEG MAX FORWARD - Left leg fully extended forward",
        10: "TRANSITION TO STANDING - Returning to initial position"
    }
    return descriptions.get(position_index, f"Position {position_index}")


# Main program
try:
    print("\n" + "=" * 60)
    print("STEP-BY-STEP GAIT POSITION TEST")
    print("=" * 60)
    print("This program will move through each gait position one by one.")
    print("After each movement, press Enter to continue to the next position.")
    print("=" * 60)

    # Start from standing position
    move_to_position(GAIT_POSITIONS[0], "Position 1 - Standing")
    print_position_info(GAIT_POSITIONS[0], "POSITION 1 - STANDING")
    print(get_position_description(1))
    wait_for_enter()

    # Move through positions 2 to 10
    for i in range(1, 10):
        position_name = f"Position {i + 1}"
        move_to_position(GAIT_POSITIONS[i], position_name)
        print_position_info(GAIT_POSITIONS[i], f"POSITION {i + 1}")
        print(get_position_description(i + 1))

        if i < 9:  # Don't wait after the last position
            wait_for_enter()

    # Return to standing position
    print("\n" + "=" * 60)
    print("RETURNING TO STANDING POSITION")
    print("=" * 60)
    move_to_position(STANDING_POSITION, "Final Standing Position")
    print_position_info(STANDING_POSITION, "FINAL POSITION - STANDING")

    print("\n" + "=" * 60)
    print("GAIT TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("All positions have been tested.")
    print("Robot is now back in standing position.")

except KeyboardInterrupt:
    print("\n\nProgram interrupted by user")

except Exception as e:
    print(f"\n\nUnexpected error: {e}")
    import traceback

    traceback.print_exc()

finally:
    # Ensure we return to standing position
    print("\nEnsuring final position is standing...")
    try:
        for i, servo in enumerate(servos):
            if servo is not None:
                servo_id = servo_ids[i]
                target_angle = STANDING_POSITION.get(servo_id, 90)
                servo.move(target_angle)
        print("Robot returned to standing position.")
    except Exception as e:
        print(f"Error returning to standing: {e}")

    print("Program exited.")