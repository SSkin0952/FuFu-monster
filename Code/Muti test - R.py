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
        print(f"✅ OK - Current angle: {angle}°")

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
        print(f"❌ Timeout - Servo {servo_id} not responding")
        servos.append(None)
    except Exception as e:
        print(f"❌ Error with servo {servo_id}: {e}")
        servos.append(None)

successful_servos = sum(1 for s in servos if s is not None)
print(f"\nSuccessfully connected to {successful_servos} out of {len(servo_ids)} servos")

if successful_servos == 0:
    print("No servos responding. Exiting...")
    quit()

STANDING_ANGLES = {
    1: 117.36,  # Left hip
    2: 118.56,  # Right hip
    3: 143.04,  # Left knee
    4: 124.56,  # Right knee
    5: 130.32,  # Left ankle
    6: 138.24,  # Right ankle
    7: 109.2,  # Left hair - lowest position
    8: 106  # Right hair - lowest position
}

# Hair servo parameters
HAIR_LEFT_RANGE = (57, 109.2)  # Left hair angle range
HAIR_RIGHT_RANGE = (55, 106)  # Right hair angle range


def calculate_hair_angles(phase):
    """
    Calculate hair angles based on gait phase
    Hair moves up when center of gravity is low (legs bent), 
    and down when standing upright
    """
    # Use leg movement phase to synchronize hair movement
    # When sin(phase) is near 0, legs are in middle position, hair should be in middle
    # When sin(phase) is negative, center of gravity is low, hair should move up
    # When sin(phase) is positive, center of gravity is high, hair should move down

    # Map leg movement phase to hair movement
    # Use cos to create 90-degree phase difference with leg movement
    hair_phase_factor = cos(phase)

    # Map from -1 to 1 range to 0 to 1 range
    normalized_phase = (hair_phase_factor + 1) / 2

    # Calculate hair angles
    left_hair_angle = HAIR_LEFT_RANGE[0] + (1 - normalized_phase) * (HAIR_LEFT_RANGE[1] - HAIR_LEFT_RANGE[0])
    right_hair_angle = HAIR_RIGHT_RANGE[0] + (1 - normalized_phase) * (HAIR_RIGHT_RANGE[1] - HAIR_RIGHT_RANGE[0])

    return left_hair_angle, right_hair_angle


# Move to standing position first
print("\nMoving to standing position...")
for i, servo in enumerate(servos):
    if servo is not None:
        servo_id = servo_ids[i]
        try:
            target_angle = STANDING_ANGLES.get(servo_id, 90)  # Default 90 degrees if not found
            servo.move(target_angle)
            print(f"Servo {servo_id} moved to {target_angle}°")
            time.sleep(0.2)
        except Exception as e:
            print(f"Failed to move servo {servo_id}: {e}")

time.sleep(1)
print("\nStarting leg movement simulation...")

t = 0
frame_count = 0

try:
    while True:
        # Basic gait parameters
        hip_amplitude = 15  # Hip swing amplitude
        knee_amplitude = 20  # Knee bend amplitude
        ankle_amplitude = 10  # Ankle compensation amplitude

        base_phase = t

        # Calculate leg joint angles
        leg_angles = [
            sin(base_phase) * hip_amplitude + STANDING_ANGLES[1],  # Servo 1 - Left hip
            sin(base_phase + pi) * hip_amplitude + STANDING_ANGLES[2],  # Servo 2 - Right hip
            sin(base_phase + pi / 3) * knee_amplitude + STANDING_ANGLES[3],  # Servo 3 - Left knee
            sin(base_phase + pi + pi / 3) * knee_amplitude + STANDING_ANGLES[4],  # Servo 4 - Right knee
            sin(base_phase + 2 * pi / 3) * ankle_amplitude + STANDING_ANGLES[5],  # Servo 5 - Left ankle
            sin(base_phase + pi + 2 * pi / 3) * ankle_amplitude + STANDING_ANGLES[6]  # Servo 6 - Right ankle
        ]

        # Calculate hair angles
        left_hair_angle, right_hair_angle = calculate_hair_angles(base_phase)

        # Combine all angles (6 legs + 2 hair)
        all_angles = leg_angles + [left_hair_angle, right_hair_angle]

        # Set angles for all servos (only move successfully connected ones)
        for i, servo in enumerate(servos):
            if servo is not None:
                try:
                    servo.move(all_angles[i])
                except Exception as e:
                    print(f"Movement error servo {servo_ids[i]}: {e}")

        frame_count += 1
        if frame_count % 20 == 0:  # Print status every 20 frames
            print(f"Frame {frame_count}: t={t:.2f}")
            for i, angle in enumerate(all_angles):
                if servos[i] is not None:
                    print(f"  Servo {servo_ids[i]}: {angle:.1f}°")

            # Special display for hair status
            print(f"  Hair status: L={left_hair_angle:.1f}° ({'UP' if left_hair_angle < 83 else 'DOWN'}), "
                  f"R={right_hair_angle:.1f}° ({'UP' if right_hair_angle < 80.5 else 'DOWN'})")

        time.sleep(0.05)
        t += 0.1

except KeyboardInterrupt:
    print("\nProgram interrupted by user")

except Exception as e:
    print(f"\nUnexpected error: {e}")

finally:
    # Return to standing position when program ends
    print("\nMoving all servos to standing position...")
    for i, servo in enumerate(servos):
        if servo is not None:
            try:
                servo_id = servo_ids[i]
                target_angle = STANDING_ANGLES.get(servo_id, 90)
                servo.move(target_angle)
                print(f"Servo {servo_id} moved to standing position ({target_angle}°)")
                time.sleep(0.1)
            except Exception as e:
                print(f"Failed to move servo {servo_id} to standing: {e}")

    print("Program exited")