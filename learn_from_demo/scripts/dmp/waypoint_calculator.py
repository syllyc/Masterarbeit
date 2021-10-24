#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
** Dynamical Movement Primitive (DMP) with Locally Weighted Regression(LWR) **
"""
# import 3rd module:
import copy
import numpy as np
import numpy.matlib
import scipy.io  # for reading mat files
from scipy.interpolate import interp1d
from scipy import stats  # for calculating gaussian distribution
from matplotlib import pyplot as plt
import yaml

# import user module:
from init_GMM_timeBased import init_GMM_timeBased
# from plot_result import plot_result
from plot_result3D import plot_result3d
from Load_Save_Data import load_handwriting_data, load_LinearMotion, write_traj_gen, read_traj_gen
import DMP
import Obstacle_avoid


def load_data(nb_Samples):
    """
    @brief: Load the training data (Trajectories of Hand/Wrist)
    :param: nb_Samples: Number of collected demonstrations
    :return: data_train: Training data
    Structure of training data: 2D:(x,y) or 3D:(x,y,z)
    """
    data_train = []  # list variable to save the training data (each element is a demonstration)
    # set data path
    # f_folder = 'data/3D_data/'
    # f_name = 'Wrist_position3d'
    # f_type = '.txt'
    f_folder = 'data/3D_data/movement2/'
    f_name = 'data'
    f_type = '.txt'
    for i in range(nb_Samples):
        # construct the data path
        data_path = '%s%s%s%s' % (f_folder, f_name, i+1, f_type)
        # load training data
        data_load = np.loadtxt(data_path, dtype=float, skiprows=1)
        data_load = data_load.T
        data_train.append(data_load)
    return data_train


'''
main loop
'''
#if __name__ == '__main__':
def waypoint_calculator():
    '''
    Set Parameters 
    '''
    with open('/home/syl/ROS_WorkSpaces/rosTuT_ws/src/learn_from_demo/scripts/dmp/config_param1.yaml', 'r') as file:
        param_config = yaml.load(file, Loader=yaml.FullLoader)  # read yaml config file
    # print(Parameter['Param2']['KP'])

    # Parameters:
    class Param:
        nb_Samples = param_config['nb_Samples']  # Number of demonstrations
        nb_Data = param_config['nb_Data']  # Length of training data after resampling (interpolated training data)
        nbStates = param_config['nbStates']  # Number of activation function (i.e. number of States in the GMM)
        KP = param_config['KP']       # Stiffness gain
        KV = (2*KP)**0.5  # Damping gain (with ideal underdamped damping ratio)
        alpha = param_config['alpha']     # Decay factor
        dt = param_config['dt']     # Duration of time step
        nbVar = 1     # Number of variables for the radial basis function [s] (decay term)
        nbDim = 3     # Dimension of movement [x, y, z] # data_train[0].shape[0]

    param = Param()  # instantiate an object

    '''
    load training data (Trajectories of Hand/Wrist)
    '''
    # load real training data
    # data_train = load_data(param.nb_Samples)
    # print(data_train[0].shape[1])
    # print(data_train[1][1][2])

    # load handwriting data for test
    data_path_handwrite = param_config['data_handwrite_path']
    data_train = load_handwriting_data(param.nb_Samples, data_path_handwrite)

    # load linear motion trajectory for testing obstacle avoidance
    # data_train = load_LinearMotion()

    '''
    Use training data to calculate the external force term 
    '''
    Force_external, sIn, Force_NL, s, position_end = DMP.learnFromDemo(param, data_train)
    # print(Force_external[2][0])
    # print(sIn)

    '''
    Use calculated external force to generate the new Force
    '''
    Weight, currF = DMP.generateForce(param, sIn, Force_external)

    '''
    Set Scaling Factor to adjust calculated external force according to new start and goal position
    '''
    # get start & end position of training data
    position_start = np.atleast_2d([data_train[0][0][0], data_train[0][1][0], data_train[0][2][0]]).T
    print('Start position of training data: ', position_start)
    print('End position of training data: ', position_end[0])

    # set new start & end position
    # position_start_new = np.atleast_2d([-4, -9, -1]).T  # here set the start position
    # position_goal_new = np.atleast_2d([-6, -7, 1]).T

    position_start_new = np.atleast_2d([0.9, -0.35, 0.2]).T  # here set the start position
    position_goal_new = np.atleast_2d([0.55, 0.2, 0.5]).T

    # position_start_new = np.atleast_2d([-3+2, -7, -16]).T  # for linear motion
    # position_goal_new = np.atleast_2d([7+2, 13, 24]).T

    # scaling generate external force
    Force_adjust = DMP.adjustForce(param, currF, position_start, position_end, position_start_new, position_goal_new)

    # scaling generate external force by rotation
    # Force_adjust = DMP.adjustForce3D(currF, position_start, position_end, position_start_new, position_goal_new)

    '''
    Use the current achieved DMP to create a new Trajectory (starting from x_0 toward x_goal)
    '''
    # set position of obstacle:
    # position_Obstacle = np.atleast_2d([-17.15, 2.56, 6.37]).T
    position_Obstacle = np.atleast_2d([-17.44, 1.85, 2.194]).T

    # position_Obstacle = np.atleast_2d([-25.68, 7.49, 1.063]).T
    # position_Obstacle = np.atleast_2d([-3.849, 7.27, 1.983]).T
    # position_Obstacle = np.atleast_2d([-4.61, -4.76, 0.611]).T

    # position_Obstacle = np.atleast_2d([2+2, 3, 4]).T  # linear motion for testing obstacle avoidance

    # set parameter for obstacle avoidance
    Gamma = 1000
    Beta = 20/np.pi
    # plot the obstacle avoidance model
    dot_phi_max = Obstacle_avoid.model2(Gamma, Beta)  # the maximum of the model of obstacle avoidance

    # generate trajectory
    Obstacle_detect = False  # Determine if obstacles are detected
    trajectory_gen = DMP.generateTrajectory(param, sIn, Force_adjust,
                                            position_start_new, position_goal_new,
                                            position_Obstacle, Gamma, Beta, dot_phi_max, Obstacle_detect)

    '''
    Plot Results
    '''
    # plot_obstacle = False  # plot obstacle or not
    # plot_result3d(param, s, trajectory_gen, sIn, Force_NL, Force_adjust, Weight, position_Obstacle, plot_obstacle)

    '''
    Write generated trajectory into a text file
    '''
    # name_file = 'trajectory_gen/traj_gen_S.txt'
    # write_traj_gen(trajectory_gen.T, name_file)

    '''
    Read generated trajectory into a text file
    '''
    # data_read = read_traj_gen(name_file)
    # print(data_read.shape)

    return trajectory_gen.T

# if __name__ == '__main__':
#     waypoint_calculator()