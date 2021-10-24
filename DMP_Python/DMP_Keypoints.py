"""
** Dynamical Movement Primitive (DMP) with Keypoints **
"""
# import 3rd module:
import copy
import numpy as np
import numpy.matlib
import scipy.io  # for reading mat files
import yaml  # for read yaml config file

# import user module:
from plot_result3D import plot_result3d
from Load_Save_Data import load_handwriting_data, load_LinearMotion, write_traj_gen, read_traj_gen, load_data
import Obstacle_avoid
from plot_function import plot_main

from Training_Traj_Preprocessor import plot_keypoints, keypoint_finder, velocity_calculator, data_train_segment, keypoints_selector
from Model_Training import external_force_calculator, skill_model_generator
from Skill_Model_Optimizer import skill_model_optimizer
from Trajectory_Generator import trajectory_generator, trajectory_connector, trajectory_generator2
from kalman_filter import kalman_filter
from variance_calculation import deviation_calculator  # for calculating the deviation


def training_data_loader(param):
    """
    load training data (Trajectories of Hand/Wrist)
    """
    # load real training data
    # data_train = load_data(param.nb_Samples)
    # print(data_train[0].shape[1])
    # print(data_train[1][1][2])

    # load handwriting data for test
    data_path_handwrite = param_config['data_handwrite_path']
    data_train = load_handwriting_data(param.nb_Samples, data_path_handwrite)

    # load linear motion trajectory for testing obstacle avoidance
    # data_train = load_LinearMotion()

    return data_train


def preprocess_training_traj(param, data_train, threshold_keypoint):
    """
    Preprocess the training trajectory: Find Keypoints and Segment the Training Trajectory
    :param param: initial parameters
    :param data_train: training trajectory
    :param threshold_keypoint: the threshold use to remove keypoints which are located too close
    :return: data_train_set: A list to save segmented training trajectory parts
             number_segments: Number of training trajectory segments
    """

    '''
    calculate the velocity of the training trajectory (use function: velocity_calculator)
    '''
    # get the velocity from training data
    v_train = velocity_calculator(param, data_train)  # list of velocity, each element is the velocity of one demo
    v_train = v_train[0]  # here we only use one demo
    # print("velocity of training data is: ", v_train)

    # get the position from training data
    p_train = data_train[0][0:3, :]

    '''
    Get the keypoints of the training trajectory
    '''
    keypoints_id, keypoints_list, threshold_eucl, keypoints_list_x, keypoints_list_y, keypoints_list_z \
    = keypoint_finder(data_train, p_train, v_train)

    '''
    Use Keypoints Selector to remove the keypoints which are too close
    '''
    keypoints_list, keypoints_id = keypoints_selector(keypoints_list, threshold_keypoint, keypoints_id, v_train)
    print('ID of Keypoints are: ', keypoints_id)

    """
    Segment Training trajectory according to keypoints
    """
    # get the segmented training trajectory
    data_train_set = data_train_segment(data_train, keypoints_id)
    number_segments = len(data_train_set)

    # plot the trajectory with keypoints
    # plot_keypoints(data_train, keypoints_id)

    return data_train_set, keypoints_list, number_segments, keypoints_list_x, keypoints_list_y, keypoints_list_z, keypoints_id


def model_training(param, data_train_set, number_segments, modified_DMP):
    """
    Use segmented Training Trajectory to get the skill model
    :param param:
    :param data_train_set:
    :param number_segments:
    :param modified_DMP:
    :return:
    """
    skill_model = []  # A List to save the learned skill(forces), each element is the learned force of the segment
    Weight = []  # A List to save the Wights of GMM, each element is the weight of the segment
    phase_variable = []  # A List to save phase variable, each element is the phase variable of the segment
    for i in range(number_segments):
        """
        Use training data to calculate the external force term 
        """
        # Force_NL: A List to save calculated external force, each element is the for one demonstration
        # sIn: phase variable for each training trajectory segment
        Force_NL, sIn = external_force_calculator(param, data_train_set[i], modified_DMP)

        '''
        Use calculated external force to get the skill model (Force)
        '''
        # W: the weight for each segment of training trajectory
        # currF: learned model(force) of each segment of training trajectory by using GMM
        W, currF = skill_model_generator(param, sIn, Force_NL)

        '''
        Save Leaned Skill model parts and Weight into List
        '''
        Weight.append(copy.deepcopy(W))  # Weight
        skill_model.append(copy.deepcopy(currF))  # Force
        phase_variable.append(copy.deepcopy(sIn))  # Phase variable

    return skill_model, Weight, phase_variable


def optimize_skill_model(skill_model, number_segments, keypoints_list):
    """
    Optimize the learned skill model according to the the new start, new end position and keypoints
    :param skill_model: A List to save the learned skill model (force), each element is the force of the segment
    :param number_segments: the number of segments
    :param keypoints_list: A List of found keypoints, each element is the keypoint(x, y, z)
    :return:
    """
    # Parameters:
    skill_model_opt = []  # A List of optimized skill model, each element is the optimized force of segment
    # get start and end position of training trajectory parts from keypoint list
    start_points = keypoints_list[0:number_segments]  # a list to save all start position of training trajectory parts
    end_points = keypoints_list[1:number_segments+1]  # a list to save all end position of training trajectory parts

    for i in range(number_segments):
        # new start and position of training trajectory parts (keypoints of total training trajectory)
        # note here set same as the training trajectory, but can adjust manually
        start_points_new = start_points[i]
        end_points_new = end_points[i]

        # Optimization: Adjust the Magnitude and Orientation of the Force (skill model)
        Force_adjust = skill_model_optimizer(skill_model[i], start_points[i], end_points[i], start_points_new, end_points_new)
        skill_model_opt.append(copy.deepcopy(Force_adjust))

    return skill_model_opt, start_points, end_points

def obstacle_avoidance():
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

def generate_trajectory(param, phase_variable, skill_model_opt, position_start_new, position_goal_new, number_segments,
                        goal_threshold):
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

    # Whether detect obstacle or not
    Obstacle_detect = False

    # Generate Trajectory
    trajectory_gen_list = []  # A List to save generate trajectory segments
    # goal_threshold = 0.13  # threshold for judging if the end position is close enough to the goal position (for hand writing)
    # goal_threshold = 0.005  # threshold for judging if the end position is close enough to the goal position (for real data)
    goal_threshold_last = 0.01  # for the last segment
    p_start_new = position_goal_new[0]  # initial a value for the new start position
    for i in range(number_segments):
        # trajectory_generator1:
        # trajectory_gen = trajectory_generator(param, phase_variable[i], skill_model_opt[i],
        #                                       position_start_new[i].T, position_goal_new[i].T,
        #                                       position_Obstacle, Gamma, Beta, dot_phi_max, Obstacle_detect)

        # trajectory_generator2:
        if i == 0:  # the first segment
            trajectory_gen, p_start_new = trajectory_generator2(param, phase_variable[i], skill_model_opt[i],
                                                               position_start_new[i].T, position_goal_new[i].T, goal_threshold,
                                                               position_Obstacle, Gamma, Beta, dot_phi_max, Obstacle_detect)

        elif i == number_segments:  # the last segment
            trajectory_gen, p_start_new = trajectory_generator2(param, phase_variable[i], skill_model_opt[i],
                                                               p_start_new, position_goal_new[i].T, goal_threshold_last,
                                                               position_Obstacle, Gamma, Beta, dot_phi_max, Obstacle_detect)
        else:
            trajectory_gen, p_start_new = trajectory_generator2(param, phase_variable[i], skill_model_opt[i],
                                                               p_start_new, position_goal_new[i].T, goal_threshold,
                                                               position_Obstacle, Gamma, Beta, dot_phi_max, Obstacle_detect)

        trajectory_gen_list.append(copy.deepcopy(trajectory_gen))

    return trajectory_gen_list, position_Obstacle


def add_gaussian_noise(data_train):
    mean_value = 0
    variance = 0.0005
    standard_deviation = np.sqrt(variance)
    for i in range(3):
        for j in range(param.nb_Data):
            data_train[0][i, j] += np.random.normal(mean_value, standard_deviation)

    return data_train


def m_TO_mm(data_m):
    """
    Convert meter to millimeter
    :param data_train: the training data in meter
    :return: the training data in millimeter
    """
    data_mm_list = []
    data_mm = data_m[0]*1000
    data_mm_list.append(data_mm)

    return data_mm_list


"""
main function:
"""
if __name__ == '__main__':
    """
    Load and Set Parameters 
    """
    # Load YAML config file
    with open('config_param1.yaml', 'r') as file:
        param_config = yaml.load(file, Loader=yaml.FullLoader)  # yaml config file
    # print(Parameter['Param2']['KP'])

    # Set Parameters:
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
        threshold_keypoint = param_config['threshold_keypoint']  # threshold used for select keypoints
        threshold_goal = param_config['threshold_goal']  # threshold for judging if the end position is close enough to the goal position
        data_handover_path = param_config['data_handover_path']  # handover data
        traj_gen_path = param_config['traj_gen_path']  # for saving generated motion trajectory

    param = Param()  # instantiate an object

    """
    load training data (Trajectories of Hand/Wrist)
    """
    # load the handwrite training data:
    data_train = training_data_loader(param)
    # data_train = add_gaussian_noise(data_train)

    # load the real training data:
    # data_train = load_data(param.data_handover_path)  # the raw training data

    """
    Preprocess the raw training trajectory with Kalman Filter
    """
    # data_train = kalman_filter(data_train)
    # convert meter to millimeter
    # data_train = m_TO_mm(data_train)

    """
    Preprocess the training Trajectory (Include: Find Keypoints, Training Trajectory Segmentation)
    """
    # threshold_keypoint for hand writing
    # threshold_keypoint = 5
    # threshold_keypoint for real data
    # threshold_keypoint = 200  # threshold used for select keypoints: remove the keypoints locate too close
    # threshold_keypoint = 0.2

    data_train_set, keypoints_list, number_segments, keypoints_list_x, keypoints_list_y, keypoints_list_z, keypoints_ID_list \
        = preprocess_training_traj(param, data_train, param.threshold_keypoint)

    """
    Model Training:
    """
    modified_DMP = True  # Set True for using modified
    skill_model, Weight, phase_variable = model_training(param, data_train_set, number_segments, modified_DMP)

    """
    Optimize learned skill Model:
    """
    skill_model_opt, position_start_list, position_end_list = optimize_skill_model(skill_model, number_segments, keypoints_list)

    """
    Generate Trajectory:
    """
    # note: here set same start and end position
    position_start_new = position_start_list
    position_goal_new = position_end_list
    trajectory_gen_list, position_Obstacle = generate_trajectory(param, phase_variable, skill_model_opt, position_start_new
                                                                 , position_goal_new, number_segments, param.threshold_goal)

    '''
    Connect generated trajectory parts
    '''
    trajectory_gen = trajectory_connector(trajectory_gen_list)

    '''
    Plot Results
    '''
    plot_obstacle = False  # plot obstacle or not
    # plot_result3d(param, s, trajectory_gen, sIn, Force_NL, Force_adjust, Weight, position_Obstacle, plot_obstacle)

    # plot generated trajectory with keypoint
    plot_main(param, data_train, trajectory_gen, phase_variable[0], skill_model[0], skill_model_opt[0], Weight[0],
              position_Obstacle, plot_obstacle, keypoints_list, keypoints_list_x, keypoints_list_y, keypoints_list_z,
              keypoints_ID_list)

    # For Check
    # plot_main(param, data_train, data_train_set[4], phase_variable[0], skill_model[0], skill_model_opt[0], Weight[0],
    #           position_Obstacle, plot_obstacle, keypoints_list, keypoints_list_x, keypoints_list_y, keypoints_list_z)


    '''
    Write generated trajectory into a text file
    '''
    # name_file = 'trajectory_gen/traj_gen_S.txt'
    # convert the unit from millimeter to meter
    # trajectory_gen = trajectory_gen/1000  # adjust according to the unit of the record data (mm or m)
    # write_traj_gen(trajectory_gen.T, param.traj_gen_path)

    '''
    Read generated trajectory into a text file
    '''
    # data_read = read_traj_gen(param.traj_gen_path)
    # print(data_read[0])
    # print(data_read.shape)


    '''
    Calculate the Deviation
    '''
    deviation = deviation_calculator(data_train, trajectory_gen)
    print(deviation)


