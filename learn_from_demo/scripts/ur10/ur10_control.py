#!/usr/bin/env python

import rospy
import roslib
import math, time, sys
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt  # plot 
from learn_from_demo.srv import ReturnJointStates  # Service to get current joint states
# from kinematics import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from universal_robot_kinematics import HTrans, invKine
import tf.transformations as tf  # library for converting between rotaion matrics, Euler angles and quaternoins


def plot_trajectory(trajectory):
    """
    plot the circle trajectory in 2D (x,y plane)
    """
    print("Trajectory plotting...... ")
    plt.figure()
    plt.axis('equal')  # maintain aspect ratio
    size_traj = len(trajectory)

    for i in range(size_traj):
        if i == 0:
            plt.scatter(trajectory[i][0], trajectory[i][1], marker='o', color='blue', label='trajectory')
        else:
            plt.scatter(trajectory[i][0], trajectory[i][1], marker='o', color='blue')

    plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.xlabel('$X/m$')
    plt.ylabel('$Y/m$')
    plt.legend(loc='best')

    plt.show()


def call_return_joint_states(joint_names):
    """
    Get the current joints states
    """
    rospy.wait_for_service("return_joint_states")
    try:
        s = rospy.ServiceProxy("return_joint_states", ReturnJointStates)
        resp = s(joint_names)
    except rospy.ServiceException, e:
        print "error when calling return_joint_states: %s"%e
        sys.exit(1)
    for (ind, joint_name) in enumerate(joint_names):
        if(not resp.found[ind]):
            print "joint %s not found!"%joint_name
    return (resp.position, resp.velocity, resp.effort)


def joint_publisher(pose_desired):
    """
    Publish Joint position to the joint controller
    """
    pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)
    rospy.init_node('ur10_joints_publisher', anonymous=True)

    # Create an instantiation of a JointTrajectory message
    JointTrajectory_object = JointTrajectory()
    # create the list of joints that will be controlled
    JointTrajectory_object.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    # add the time stamp and frame_id to the header
    # JointTrajectory_object.header.stamp = rospy.Time.now()
    JointTrajectory_object.header.frame_id = 'base_link'

    period = 0.1
    rate = rospy.Rate(1/period)
      
    point_object = JointTrajectoryPoint()
    # point_object.positions = [math.sin(angle), -1.57+math.sin(angle), math.sin(angle), math.sin(angle)/2, math.sin(angle)/2, 0]  
    point_object.positions = pose_desired
    point_object.velocities = [0, 0, 0, 0, 0, 0]
    point_object.accelerations = [0, 0, 0, 0, 0, 0]
    point_object.effort = [0] 

    # point_object.time_from_start.secs[1]

    JointTrajectory_object.points.append(point_object)
    JointTrajectory_object.points[0].time_from_start = rospy.Duration.from_sec(period)
    # JointTrajectory_object.points[counter].time_from_start = rospy.Duration.from_sec(1)
    
    pub.publish(JointTrajectory_object)
    rate.sleep()

    JointTrajectory_object.points.pop()


def pose_interploation(start_pose, end_pose, num_interpolate):
    """
    Interpolation between two given Poses
    Args:
    start_pose: A list of  start pose:[x,y,z,rx,ry,rz] unit:(m, radian)
    end_pose: A list of  end pose:[x,y,z,rx,ry,rz] unit:(m, radian)
    num_interpolate: number of interpolation

    Returns:
    pose_list_out: A list of set of Poses after inerpolation
    """
    position_list_in = []  # A list of set of Poses 

    for i in range(3):
        position_list_in.append(np.linspace(start_pose[i, 3], end_pose[i, 3], num_interpolate))
    
    # print "position_list_in"
    # print position_list_in[0]

    # interpolate the x, y, z
    pose_list = []
    for i in range(num_interpolate):
        start_pose[0, 3] = position_list_in[0][i]
        start_pose[1, 3] = position_list_in[1][i]
        start_pose[2, 3] = position_list_in[2][i]
        pose_list.append(deepcopy(start_pose))

    # print "pose_list:"
    # print pose_list[9]
   
    return pose_list


def get_curr_joint_state(joint_names, euler_type):
    """
    Get the current joint states, calculate the pose of the last joint by Forward Kinematics
    """
    # get current joint positions 
    (position_curr, velocity_curr, effort_curr) = call_return_joint_states(joint_names)
    joint_position_curr = [position_curr[0], position_curr[1], position_curr[2], position_curr[3], position_curr[4], position_curr[5]]
    # print "joint position start: ", joint_position_curr

    # Use Forwards Kinematics to calculate the current TCP Pose 
    c = [0]
    th = np.matrix([[joint_position_curr[0]],[joint_position_curr[1]],[joint_position_curr[2]],\
                    [joint_position_curr[3]],[joint_position_curr[4]],[joint_position_curr[5]]])
    pose_curr_matrix = HTrans(th,c)

    # print "pose homogeneous matrix is:"
    # print pose_curr_matrix

    # from homogeneous matrix to get the rotation matrix
    Rotation_matrix_curr = pose_curr_matrix[0:3, 0:3]  

    # convert pose matrix to euler angles
    euler_angle_curr = tf.euler_from_matrix(Rotation_matrix_curr, euler_type)  
    # print "Euler Angles are:"
    # print euler_angle_curr

    return joint_position_curr, pose_curr_matrix, euler_angle_curr


def ik_seclector(q_candidates, q_refer, w=[1]*6):
    """
    Select the optimal solutions among a set of feasible joint value solutions.
    Args:
        q_candidates: A set of feasible joint value solutions (unit: radian)
        q_refer: A list of q_referenced joint value solution (unit: radian)
        w: A list of weight corresponding to robot joints

    Returns:
        A list of optimal joint value solution.
    """
    q_candidates = q_candidates.T
    q_list = []
    for k in range(8):
        q_list.append([q_candidates[k,0], q_candidates[k,1], q_candidates[k,2], q_candidates[k,3], q_candidates[k,4], q_candidates[k,5]])

    error = []
    for q in q_list:
        error.append(sum([w[i] * (q[i] - q_refer[i]) ** 2 for i in range(6)]))
    
    # print "q_solutions: "
    # print q_candidates
    
    return q_list[error.index(min(error))]


def move2home(sleep_sec):
    """
    move robot to home pose, where each joint angle is 0
    """
    pose_home = [0, 0, 0, 0, 0, 0]
    joint_publisher(pose_home)  # the first pose will lost
    joint_publisher(pose_home)  # start from second pose, strange here
    time.sleep(sleep_sec)
    print "move to home pose"


def ur10_ik_solver(pose_desired, joint_position_curr):
    """
    Modify the Ik from: https://github.com/mc-capolei/python-Universal-robot-kinematics/blob/master/universal_robot_kinematics.py
    Set the range of joint1 between (-180, 180)
    Args:
        pose_desired: desired pose
        joint_position_curr: referenced pose for select multi-solution of Invser Kinematics
    Return:
        ik_solution_mod: selected Invser Kinematic solution
    """
     # Calculate Inverse Kinematics
    th_ik = invKine(pose_desired)  # the candidates of the ik solutions
    # print "all solutions"
    # print th_ik
    # print ""

    # judge the size of theta1 (the first joint angle)
    if th_ik[0,0] <= np.pi and th_ik[0,4] <= np.pi:
        # judge if the current theta smaller than 0
        if joint_position_curr[0] < 0:
            joint_position_curr[0] = 2*(np.pi) + joint_position_curr[0]

        # select the modified ik solutions
        ik_solution_mod = ik_seclector(th_ik, joint_position_curr)
        # print "modifeid ik solution: ", ik_solution_mod

    else: # th_ik[0,0] > np.pi or th_ik[0,4] > np.pi:
        # judge if the current theta smaller than 0
        if joint_position_curr[0] < 0:
            joint_position_curr[0] = 2*(np.pi) + joint_position_curr[0]

        # select the modified ik solutions
        ik_solution_mod = ik_seclector(th_ik, joint_position_curr)
        # calculate the angle which equal to -(360 - theta)
        ik_solution_mod[0] = -(2*np.pi - ik_solution_mod[0])
        # print "modifeid ik solution: ", ik_solution_mod
    
    return ik_solution_mod


"""
For testing:
"""

def test_kinematics():
    """
    1. Input a set of joint postion
    2. Move robot to this pose and record this pose as input of inverse kinematics
    3. Replay this pose by using the solution of inverse kinematics
    """
    
    # Set a set of joint positions
    joint_deg = [60, -90, 90, -90, -90, 0]
    pose_desired = np.deg2rad(joint_deg)
    # print "radian of 240 degree: ", np.deg2rad(240)
    print "Input Joint Angles in rad: ", pose_desired

    # Move to this pose and record
    joint_publisher(pose_desired)
    time.sleep(4)  # wait robot reach to the target pose

    # get current joint positions 
    joint_position_curr, pose_curr_matrix, euler_angle_curr = get_curr_joint_state(joint_names, euler_type)
    print "current joint position: ", joint_position_curr
    print "record pose homogeneous matrix is:"
    print pose_curr_matrix
    print ""
    # print "Euler Angles are:", euler_angle_curr

    '''
    # Check 8 different Ik solutions
    i=6
    joint_ik_solution = [th_ik[0,i], th_ik[1,i], th_ik[2,i], th_ik[3,i], th_ik[4,i], th_ik[5,i]]
    joint_publisher(joint_ik_solution)

    time.sleep(3)
    '''

    # move back to home
    sleep_sec = 3
    move2home(sleep_sec)

    # calculate the IK 
    ik_solution_mod = ur10_ik_solver(pose_curr_matrix, joint_position_curr)

    # replay the pose
    joint_publisher(ik_solution_mod)
    time.sleep(4)  # wait robot reach to the goal pose


    # get current joint states:
    joint_position_curr, pose_curr_matrix, euler_angle_curr = get_curr_joint_state(joint_names, euler_type)
    print "current joint position: ", joint_position_curr
    print "current pose homogeneous matrix is:"
    print pose_curr_matrix
    print ""


def move_straight():
    """
    move to a point in a straight line and keep the same oriantation with the start pose
    """
    # get current joint positions 
    joint_position_curr, pose_curr_matrix, euler_angle_curr = get_curr_joint_state(joint_names, euler_type)
    print "current joint position: ", joint_position_curr
    print "start pose homogeneous matrix is:"
    print pose_curr_matrix
    print ""

    # Set a goal position (the oreintation keep same)
    pose_goal = deepcopy(pose_curr_matrix)
    pose_goal[0, 3] = 0.7  # goal position in x-axis 
    pose_goal[1, 3] = 0.8  # goal position in y-axis
    pose_goal[2, 3] = 0.6  # goal position in z-axis

    # Interplation between current pose and goal pose
    num_interpolate = 20  # number of interpolation
    pose_list = pose_interploation(pose_curr_matrix, pose_goal, num_interpolate)  # pose list after interpolation
    print "The goal pose Homogeneous Matrix:"
    print pose_list[num_interpolate-1]

    # move to the goal pose
    for i in range(num_interpolate):
        # get current joint state
        joint_position_curr, pose_curr_matrix, euler_angle_curr = get_curr_joint_state(joint_names, euler_type)

        # calculate the IK solution
        ik_solution_mod = ur10_ik_solver(pose_list[i], joint_position_curr)

        # publish the calculated joint positions to UR10
        joint_publisher(ik_solution_mod)
    
    # get current joint states:
    time.sleep(2)  # wait for reach to the goal pose
    joint_position_curr, pose_curr_matrix, euler_angle_curr = get_curr_joint_state(joint_names, euler_type)
    print "current joint position: ", joint_position_curr
    print "current pose homogeneous matrix is:"
    print pose_curr_matrix
    print ""


def move_circle():
    """
    Move to draw a Circle
    """
    # get current joint positions 
    joint_position_curr, pose_curr_matrix, euler_angle_curr = get_curr_joint_state(joint_names, euler_type)
    print "current joint position: ", joint_position_curr

    # Set the Circle 
    centerX = -0.6
    centerY = 0.0
    centerZ = 0.4  
    radius = 0.2

    # Set the first Waypoint
    th = 0
    waypoint1_circle = deepcopy(pose_curr_matrix)
    waypoint1_circle[0, 3] = centerX + radius
    waypoint1_circle[1, 3] = centerY 
    waypoint1_circle[2, 3] = centerZ

    # Move to the first waypoint of the circle
    num_interpolate = 20  # number of interpolation
    pose_list = pose_interploation(pose_curr_matrix, waypoint1_circle, num_interpolate)  # pose list after interpolation
    print "The goal pose Homogeneous Matrix:"
    print pose_list[num_interpolate-1]

    # move to the goal pose
    for i in range(num_interpolate): 
        # get current joint state
        joint_position_curr, pose_curr_matrix, euler_angle_curr = get_curr_joint_state(joint_names, euler_type)

        # calculate the IK solution
        ik_solution_mod = ur10_ik_solver(pose_list[i], joint_position_curr)

        # publish the calculated joint positions to UR10
        joint_publisher(ik_solution_mod)

    # get current joint positions 
    time.sleep(1)
    joint_position_curr, pose_curr_matrix, euler_angle_curr = get_curr_joint_state(joint_names, euler_type)
    print "current joint position: ", joint_position_curr
    print "start pose homogeneous matrix is:"
    print pose_curr_matrix
    print ""

    # set the orientation of the TCP
    pose_circle = deepcopy(pose_curr_matrix)
    
    while(1):
        # get current joint position as the reference joint position
        joint_position_curr, pose_curr_matrix, euler_angle_curr = get_curr_joint_state(joint_names, euler_type)
        
        # Draw Circle
        pose_circle[0, 3] = centerX + radius * math.cos(th)
        pose_circle[1, 3] = centerY + radius * math.sin(th)
        pose_circle[2, 3] = centerZ 

        # calculate the IK solution
        ik_solution_mod = ur10_ik_solver(pose_circle, joint_position_curr)

        # pubish the joint position to ur10 controller
        joint_publisher(ik_solution_mod)
        th = th - 0.1



if __name__ == '__main__':
    try:
        """
        Set parameters:
        """
        # Set Joint Name of UR10 for getting the current joint states
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

        # pose in euler angles format; arg:'rxyz': r-rotation, 'szyz': s-static
        euler_type = 'rzyx'
        print "type of euler angles: ", euler_type

        """
        Move robot to the home pose (0, 0, 0, 0, 0, 0)
        """
        sleep_sec = 3  #move to home than wait 3 seconds 
        move2home(sleep_sec)

        """
        Move to a start pose0
        """
        # Set joint positions for Pose 0
        joint_deg_0 = [0, -90, 90, -90, -90, 0]
        pose_0 = np.deg2rad(joint_deg_0)  # convert degree to radian
        print "Joint Angles of Start pose: ", pose_0
        # Move to Pose0
        joint_publisher(pose_0)
        time.sleep(1.5)  # wait robot reach to the target pose

        """
        Test1: Kinematics of UR10
        """
        # test_kinematics()

        """
        Test2: Move to a desired point in a straight line and keep the same oriantation with the start pose
        """
        # move_straight()

        """
        Test3: Dras a Circle
        """
        move_circle()

    except rospy.ROSInterruptException:
        pass


