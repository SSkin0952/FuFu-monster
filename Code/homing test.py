from pylx16a.lx16a import *
import time


def reset_all_servos_to_standing():
    """
    Reset all servos to standing position
    This program moves all servos to their predefined standing angles
    """

    # Standing angles for all servos
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

    # Initialize serial connection
    try:
        print("Initializing /dev/ttyUSB0...")
        LX16A.initialize("/dev/ttyUSB0", 0.1)
        print("/dev/ttyUSB0 initialized successfully")
    except Exception as e:
        print(f"Failed to initialize serial connection: {e}")
        print("Please check:")
        print("1. /dev/ttyUSB0 port connection")
        print("2. USB cable")
        print("3. Driver installation")
        return

    # Servo ID assignment
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
        return

    # Move all servos to standing position
    print("\n" + "=" * 50)
    print("RESETTING ALL SERVOS TO STANDING POSITION")
    print("=" * 50)

    for i, servo in enumerate(servos):
        if servo is not None:
            servo_id = servo_ids[i]
            try:
                target_angle = STANDING_ANGLES.get(servo_id, 90)
                print(f"Moving servo {servo_id} to {target_angle}°...", end=" ")
                servo.move(target_angle)
                print("✅ DONE")
                time.sleep(0.3)  # Slightly longer delay for safety
            except Exception as e:
                print(f"❌ FAILED: {e}")

    # Verify final positions
    print("\n" + "=" * 50)
    print("VERIFYING FINAL POSITIONS")
    print("=" * 50)

    for i, servo in enumerate(servos):
        if servo is not None:
            servo_id = servo_ids[i]
            try:
                current_angle = servo.get_physical_angle()
                target_angle = STANDING_ANGLES.get(servo_id, 90)
                status = "✅ MATCH" if abs(current_angle - target_angle) < 2 else "⚠️ CLOSE"
                print(f"Servo {servo_id}: Current={current_angle:.1f}°, Target={target_angle}° - {status}")
            except Exception as e:
                print(f"Servo {servo_id}: ❌ Failed to read position - {e}")

    print("\n" + "=" * 50)
    print("RESET COMPLETE")
    print("=" * 50)


def main():
    """
    Main function to run the reset program
    """
    print("Servo Reset Program")
    print("This program will reset all servos to their standing positions.")

    # Confirm with user
    user_input = input("Do you want to continue? (y/n): ").strip().lower()
    if user_input not in ['y', 'yes']:
        print("Reset cancelled.")
        return

    # Execute reset
    reset_all_servos_to_standing()

    # Completion message
    print("\nAll servos have been reset to standing position.")
    print("You can now safely disconnect power or run other programs.")


if __name__ == "__main__":
    main()