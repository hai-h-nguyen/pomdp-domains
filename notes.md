1. RUN docker (in every terminal later if want to access ros)
```
cd ~/./ur_environment/
./RUN-DOCKER-CONTAINER.sh
```

2. Load the gripper using the pendant:
- Close gradually until the peg is correctly held and then close 100%

3. Launch the ros script
```
roslaunch tactile_feedback_control system_bringup.launch
```

4. Loading and start running ROS_driver on the pendant

5. Run the filtering for F/T signals (the peg is in the air and no vibration)
```python pdomains/robot_utils/ft_filter.py -t "wrench" -z```

6. Publish the hole coordinate:
```
rosrun tf static_transform_publisher 0.01 -0.522 0.117 0 0 0 base hole_coordinate 100
```

7. Run rviz with the saved configuration
```
rviz
```

8. Run the script (must copied into ur_environment/catkin_ws/src)
