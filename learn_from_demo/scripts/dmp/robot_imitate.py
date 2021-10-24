#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Test DMP: Test learned Trajectory im Simulation Gazebo #

import rospy, sys
import moveit_commander
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import Pose, PoseStamped
from copy import deepcopy
from read_write import read_traj_gen

import math
import numpy as np

class DMPtest:
    def __init__(self):
        """
        Initializing and Setting
        """
        # Initialize move_group API
        moveit_commander.roscpp_initialize(sys.argv)

        # Initialize ROS node
        rospy.init_node('dmp_test', anonymous=True) 

        # Initialize the move group which will be planned by moveit
        group_arm = moveit_commander.MoveGroupCommander('manipulator')   

        # get the name of End-Effector
        end_effector_link = group_arm.get_end_effector_link()
                        
        # Set the reference frame for calculating IK
        reference_frame = 'base_link'
        group_arm.set_pose_reference_frame(reference_frame)
                
        # When motion planning failed, allow replaning
        group_arm.allow_replanning(True)
       
        # Set the allowable joint error value (unit:rad)
        group_arm.set_goal_joint_tolerance(0.001)

        # Set Position (unit:meter) and Oreintation (unit:rad）allowed error
        group_arm.set_goal_position_tolerance(0.001)
        group_arm.set_goal_orientation_tolerance(0.01)

        # Set the maximum speed and acceleration allowed
        group_arm.set_max_acceleration_scaling_factor(0.5)
        group_arm.set_max_velocity_scaling_factor(0.5)
        

        """
        Control the Robot to the 'Ready' Pose
        """
        # Control the manipulator(arm) to the initial position
        group_arm.set_named_target('home')
        group_arm.go()
        rospy.sleep(1)
         
        # Set the Ready Pose（init：rad）
        ready_positions = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]  # ready_position: Pose which is ready to perform task
        group_arm.set_joint_value_target(ready_positions)
                 
        # Control robot to the Ready Pose
        group_arm.go()
        rospy.sleep(1)


        """
        From the ready pose move to the start position of the Movement (Trajectory generated by DMP)
        """
        # Load generated Trajectory 
        name_file = '/home/syl/ROS_WorkSpaces/rosTuT_ws/src/learn_from_demo/trajectory_gen/traj_gen_21.txt'
        traj_gen = read_traj_gen(name_file)

        # Set current pose as the start pose of the motion
        # group_arm.set_start_state_to_current_state()

        # Define target Pose:
        target_pose = PoseStamped()
        target_pose.header.frame_id = reference_frame
        target_pose.header.stamp = rospy.Time.now()    

        target_pose.pose.orientation.x = -0.5
        target_pose.pose.orientation.y = 0.5
        target_pose.pose.orientation.z = 0.5
        target_pose.pose.orientation.w = 0.5
                
        # Initilize the waypoints list
        waypoints = []        

        # Load the Trajectory that are generated by DMP
        data_size = traj_gen.shape[0] 
        rospy.loginfo("The number of the waypoints of the trajectory: " + str(data_size))
        # data_size = 258
        for i in range(data_size):
            target_pose.pose.position.x = traj_gen[i][0]/2+0.2
            target_pose.pose.position.y = traj_gen[i][1]
            target_pose.pose.position.z = traj_gen[i][2]
            wpose = deepcopy(target_pose.pose)
            waypoints.append(deepcopy(wpose))

        fraction = 0.0   # fraction of success planed waypoints 
        maxtries = 100   # maximum attempt times
        attempts = 0     # already attempt times
        
        # set the current pose as the initial pose
        group_arm.set_start_state_to_current_state()
 
        # attempt to plan a trajectory in cartesian space 
        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = group_arm.compute_cartesian_path (
                                    waypoints,   # waypoint poses
                                    0.01,        # eef_step
                                    0.0,         # jump_threshold
                                    True)        # avoid_collisions
            
            attempts += 1
            
            # print the process of motion planning 
            if attempts % 10 == 0:
                rospy.loginfo("Still trying after " + str(attempts) + " attempts...")
                     
        # if motion path plan sucessful (fraction = 100 %), robot start to execute the motion trajectory 
        if fraction == 1.0:
            rospy.loginfo("Path computed successfully. Moving the arm.")
            group_arm.execute(plan)
            rospy.loginfo("Path execution complete.")
        # if motion path plan failed
        else:
            rospy.loginfo("Path planning failed with only " + str(fraction) + " success after " + str(maxtries) + " attempts.")  

        rospy.sleep(1)

        """
        Back to Home Pose and Exit 
        """
        # Control the manipulator(arm) to return to the initial position
        group_arm.set_named_target('home')
        group_arm.go()
        rospy.sleep(1)
        
        # close and exit moveit
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

if __name__ == "__main__":
    try:
        DMPtest()
    except rospy.ROSInterruptException:
        pass    