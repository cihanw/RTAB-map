# Task Plan: Test Run & Bug Evaluation

## Objectives
1. Perform a live test run to monitor the drone's behavior without making any modifications to the core project logic.
2. Save detailed logs of the test run.
3. Save 1 camera frame every 60 frames (approx. every 2 seconds at 30 fps).
4. Evaluate **Problem 1**: Drone getting too close to obstacles, facing them, and losing visual odometry (VO).
5. Evaluate **Problem 2**: Map distortion likely due to faulty loop closures.

## Checklist
- [x] Create a `record_test.py` script in the workspace to listen to the camera topic (`/world/husarion_office/model/drone/model/d455/link/link/sensor/realsense_d455/image`) and save every 60th frame to a `log/frames/` directory.
- [x] Write a bash script `start_test.sh` to launch `autonomous.launch.py` and pipe all console output to `log/test_run.log`.
- [x] Start the test run and let the drone explore.
- [x] Monitor the logs for VO loss events (Problem 1) and rtabmap loop closure rejections/acceptances (Problem 2).
- [x] Stop the test run after sufficient exploration.
- [x] Analyze the collected logs and frames.
- [x] Propose solutions for Problem 1 (e.g. tuning `min_safe_distance`, `influence_radius`, or local planner states).
- [x] Propose solutions for Problem 2 (e.g. tuning `Vis/MinInliers`, `Rtabmap/LoopThr`, or `RGBD/OptimizeMaxError`).

## Review
- The test run successfully explored the environment for several minutes and saved frames. While the rare events did not trigger in this short window, detailed code analysis of the `local_planner.py` and `slam.launch.py` combined with historical `lessons.md` provides a clear evaluation of the root causes for both problems.
