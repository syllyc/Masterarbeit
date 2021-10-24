#####################
### Operating System ###
#####################
Ubuntu: 16.04
ROS version: kinetic

################################
### DMP ROS Package dependences ###
################################
numpy
scipy
matplotlib
PyYAML

################################
### Required 3rd-part ROS Packages ###
################################
1. moveit: http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/getting_started/getting_started.html
2. universal_robot: https://github.com/ros-industrial/universal_robot

#######################
### Package Description ###
######################
Package Name: learn_from_demo
Node: ur10_control.py
Description: Control UR10 online (Sending joint Poition to UR10 Controller directly)
	    Inverse Kinematics is refer to: https://github.com/mc-capolei/python-Universal-robot-kinematics/blob/master/universal_robot_kinematics.py
				    https://github.com/mc-capolei/python-Universal-robot-kinematics/issues/1
Usage:
roslaunch learn_from_demo control_ur10.launch  # start the simulation environment Gazebo and needed ros node
rosrun learn_from_demo ur10_control.py  # move the robot to desired pose (Move straight, Draw circle)

Note: 
Use command "rosnode list" to check if the node "joint_states_listener.py" is running.   
Node joint_states_listener.py is used to get the current state of the joints (position, velocity and torque)






