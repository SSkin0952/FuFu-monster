# Bipedal Robot Simulation (MuJoCo)

This directory contains the MuJoCo simulation environment and reinforcement learning experiments for the bipedal robot.

The simulation is based on a physics model defined in `fufu.xml`, with robot geometry provided as STL meshes.

The scripts in this folder implement different approaches for gait generation and optimization.

---

# Directory Structure

simulation/

│
├── fufu.xml  
│   MuJoCo robot model describing the robot body, joints, sensors, and actuators.
│
├── mesh/  
│   Contains STL geometry files for each robot component used in the MuJoCo model.
│
├── walk simulation.py  
│   Keyframe-based walking animation and IMU monitoring.
│
├── train_hill_climber.py  
│   Hill-climbing optimization for gait fine-tuning.
│
├── train_pg.py  
│   Policy Gradient reinforcement learning for gait learning.
│
├── train_random_search.py  
│   Random search optimization for gait parameters.
│
└── README.md