# Bipedal Robot Motion Control

This repository contains motion control and testing programs for a custom-built bipedal robot powered by a Raspberry Pi and LX-16A serial bus servos.

The project implements servo initialization, system boot diagnostics, gait testing, and continuous walking control.

---

## Hardware

- Raspberry Pi
- 8 × LX-16A serial bus servos
- USB-to-TTL servo controller
- Custom 3D-printed bipedal robot frame

---

## Dependencies

This project uses the PyLX-16A Python library for controlling LX-16A serial bus servos.

PyLX-16A was developed by Ethan Lipson:  
https://github.com/ethanlipson/PyLX-16A

---

## Project Structure

### Boot and System Tests

**boot test.py**

Performs a comprehensive boot diagnostic of the robot system, including:

- Power supply verification
- CPU and memory checks
- Sensor communication tests
- Actuator self-tests
- Safety system validation
- Network connectivity check

---

### Servo Initialization and Testing

**servo-test.py**

Basic servo communication test that initializes the serial bus and verifies servo response.

**hello-world.py**

Simple sinusoidal motion test for verifying servo control functionality.

---

### Homing and Reset

**homing test.py**

Moves all servos to a predefined standing configuration to safely initialize the robot.

This is typically run before gait testing.

---

### Gait Testing

**Muti test - R.py**

Step-by-step gait position testing.

The robot moves through predefined gait positions to verify joint motion and stability.

---

### Greeting / Gesture Control

**Ciallo.py**

Implements a simple greeting behavior for the robot.

This script controls only the two lateral arm servos (servo 7 and servo 8), which correspond to the robot's side “hair” appendages. The servos move in a sinusoidal pattern within safe angle limits to create a waving motion, allowing the robot to perform a greeting gesture.

Features:

- Independent control of servo 7 and servo 8
- Safe angle clamping to avoid hitting mechanical limits
- Smooth sinusoidal waving motion
- Designed for quick demonstration or interaction

### Continuous Walking

**walk.py**

Implements the full walking controller using predefined gait keyframes and smooth interpolation between them.

Features include:

- multi-servo coordinated motion
- smooth gait interpolation
- safe stop positions
- servo temperature and voltage monitoring
- Raspberry Pi system health monitoring