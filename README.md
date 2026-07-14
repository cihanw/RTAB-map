# LLM-Controlled Autonomous Drone 🚁🧠

Welcome to the **LLM-Controlled Autonomous Drone** project! This repository contains a fully autonomous drone simulation built on **ROS 2 Jazzy** and **Gazebo Harmonic**, leveraging advanced SLAM (Simultaneous Localization and Mapping) and multi-layered path planning algorithms.

This project was developed iteratively in collaboration with an AI agent to achieve robust, collision-free autonomous exploration in unknown environments.

---

## 🌟 Key Features

1. **Autonomous Exploration (NBV)**
   - Implements a **Next-Best-View (NBV)** planner using frontier-based exploration to systematically map unknown areas.
   - Evaluates frontiers based on proximity and expected information gain.

2. **Advanced SLAM & Odometry**
   - Uses **RTAB-Map** for dense 3D mapping and global loop closure detection.
   - Fuses Visual Odometry (from a simulated Intel RealSense D455) and IMU data using an **Extended Kalman Filter (EKF)** via `robot_localization` for robust state estimation.

3. **Multi-Layered Planning Architecture**
   - **Global Planner**: Uses the **Theta*** algorithm to find shortest, line-of-sight paths through the known octomap grid.
   - **Local Planner**: Utilizes an **Artificial Potential Field (APF)** to actively repel the drone from sudden obstacles, featuring a "Turn-Then-Go" motion model for enhanced safety and visual tracking.

4. **Gazebo Harmonic Simulation**
   - Simulates a PX4-style x500 quadcopter with realistic rigid body dynamics.
   - Multiple simulated environments (Husarion Office, Depot warehouse) for testing perceptual aliasing and obstacle avoidance.

---

## 🏗️ System Architecture

The project is structured as a standard ROS 2 workspace (`drone_ws`). The core logic resides in `drone_sim`:

- `slam.launch.py`: Initializes the RTAB-Map SLAM backend, visual odometry, and EKF fusion.
- `sim.launch.py`: Spawns the Gazebo Harmonic world, the x500 drone, and sets up `ros_gz_bridge`.
- `autonomous.launch.py`: The master launch file that coordinates the bringup of the simulation, SLAM, and all three planners sequentially to avoid time-sync issues.
- `nbv_planner.py`: Node responsible for frontier extraction and goal generation.
- `theta_star_planner.py`: Node responsible for global path generation.
- `local_planner.py`: Node responsible for velocity command execution and collision avoidance.

---

## 🚀 Getting Started

### Prerequisites
- **Ubuntu 24.04** (Recommended)
- **ROS 2 Jazzy**
- **Gazebo Harmonic**
- Python 3 virtual environment (`venv`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/cihanw/LLM-Controlled-Autonomous-Drone.git
   cd LLM-Controlled-Autonomous-Drone/drone_ws
   ```
2. Install dependencies:
   ```bash
   sudo apt install ros-jazzy-rtabmap-ros ros-jazzy-robot-localization
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt # (Make sure scipy, numpy, opencv-python are installed)
   ```
3. Build the workspace:
   ```bash
   source /opt/ros/jazzy/setup.bash
   colcon build --symlink-install
   ```

### Running the Simulation
To launch the full autonomous stack in the default Depot world:
```bash
cd drone_ws
source install/setup.bash
source venv/bin/activate
ros2 launch drone_sim autonomous.launch.py
```
The drone will automatically take off, identify frontiers, and begin mapping the environment.

---

## 🛠️ Lessons Learned & Bug Fixes
Throughout the development of this project, several significant robotics challenges were overcome. See `tasks/lessons.md` for deep-dives into:
- Visual Odometry scale drift and featureless-wall blindness.
- Perceptual Aliasing (faulty loop closures in symmetrical warehouse aisles).
- TF tree buffering and strictly enforcing `use_sim_time`.
- Hysteresis thresholds to prevent Local Planner oscillation.

## 📜 License
This project is open-source and available under the MIT License.
