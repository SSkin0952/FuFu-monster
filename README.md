# FuFu Monster – Bipedal Robot Platform

FuFu Monster is a **custom-designed bipedal robot platform** integrating mechanical design, real-world control, and physics-based simulation.

The project includes:

- full **SolidWorks mechanical design**
- **Python-based robot control system**
- **MuJoCo simulation environment**
- **reinforcement learning gait optimization**

The goal of this project is to explore **biped locomotion, robot control, and sim-to-real gait development** on a compact hardware platform.

---

# Project Overview

FuFu Monster is powered by:

- **Raspberry Pi**
- **LX-16A bus servos**
- **IMU sensors**
- **custom 3D-printed mechanical structure**

The robot features:

- 8 Degrees of Freedom (8-DOF)
- biped walking mechanism
- expressive side appendages ("hair")
- onboard computing and power system

---


# Repository Structure

```
FuFu-monster/
├── Code
│   Robot control software running on Raspberry Pi
│
├── Simulation
│   MuJoCo simulation environment and reinforcement learning experiments
│
└── SW-assembly
    Mechanical design files (SolidWorks)
```

# Code

Code/
Contains Python scripts used to control the physical robot.

Main features:

- LX-16A servo communication
- multi-servo coordination
- walking behavior control
- diagnostic and testing scripts
- gesture behaviors (e.g., waving)

---

# Simulation

Simulation/
Contains the MuJoCo physics simulation environment used to:

- test walking behaviors
- collect IMU data
- optimize gait parameters
- train reinforcement learning controllers

Implemented training methods include:

- Hill Climbing optimization
- Policy Gradient reinforcement learning
- Random Search

---

# Mechanical Design

SW-assembly/
Contains the full **SolidWorks design** of the robot including:

- full robot assembly
- leg mechanisms
- servo mounts
- electronics mounting structure
- 3D-printable parts

---

# Hardware Architecture

Main hardware components:

- Controller: **Raspberry Pi**
- Actuators: **LX-16A Bus Servos**
- Power System: **Rechargeable Li-ion battery pack**
- Sensors: **IMU**
- Structure: **3D-printed mechanical components**

---

# Simulation to Real Robot

The development pipeline is designed to support **sim-to-real gait transfer**:

1. Design robot structure in **SolidWorks**
2. Build physics model in **MuJoCo**
3. Optimize walking gait through **reinforcement learning**
4. Deploy optimized control to the **real robot**

---

# Dependencies

Robot control software requires: Python 3, pylx16a, numpy

Simulation requires: mujoco, numpy, torch, matplotlib, pandas

---

# Acknowledgment

This project uses the LX-16A servo communication library:

PyLX-16A  
https://github.com/ethanlipson/PyLX-16A

---

# Future Work

Planned improvements include:

- improved walking stability
- reinforcement learning gait transfer to hardware
- better perception and interaction behaviors
- advanced motion planning

---

# Author

Wei Sun(ws2782@columbia.edu)
Columbia University
