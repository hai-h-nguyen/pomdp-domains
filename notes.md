1. RUN docker (in every terminal later if want to access ros)
```
cd ~/./ur_environment/
./RUN-DOCKER-CONTAINER.sh
```

2. Load the gripper using the pendant:
- Close gradually until the peg is correctly held and then close 100%

3. Launch the ros script
```roslaunch tactile_feedback_control system_bringup.launch```

4. Loading and start running ROS_driver on the pendant

5. Run the filtering for F/T signals (the peg is in the air and no vibration)
```python pdomains/robot_utils/ft_filter.py -t "wrench" -z```

6. Start the red detector node
```python pdomains/robot_utils/red_detector_publisher.py```

7. Get the hole pose (reset position before randomizing the initial position) by moving the peg to the desire location
and get ```self.ur5e.get_cartesian_state()```

7. Determine the hole coodinate: move the peg to the end of the hole and look up for the transform:
```rosrun tf tf_echo /base /robotiq_85_flexwrist_tip_link_with_peg```

7. Move to the peg to the end of the hole and
```rosrun static_transform_publisher x y z yaw pitch roll frame_id child_frame_id period_in_ms```
```rosrun tf static_transform_publisher 0.0 0.0 0.0 -1.57079632679 -1.57079632679 1.57079632679 robotiq_85_flexwrist_tip_link_with_peg desired_hole_coordinate 100``
```rosrun tf tf_echo /base /desired_hole_coordinate```

7. Get the result to publish
```rosrun tf static_transform_publisher 0.012 -0.515 0.077 0.02 0.001 0.002 base hole_coordinate 100``` (round)
```rosrun tf static_transform_publisher 0.010 -0.517 0.087 0.02 0.001 0.001 base hole_coordinate 100``` (square)

7. Get the offset z by print arm_tip_pos_in_hole before minus the offset when the peg is at the hole coordinate's origin: 
need comment to not go to a random location, need to set the current tip2hole_offset_z in the yaml file to 0

8. Run rviz with the saved configuration
```rviz```

9. Run the script (must copied into ur_environment/catkin_ws/src)
