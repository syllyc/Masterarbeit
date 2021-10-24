"""
** Dynamical Movement Primitive (DMP) with Locally Weighted Regression(LWR) **
"""
# import 3rd module:
import copy
import numpy as np
from scipy.interpolate import interp1d
import yaml

# import user module:
from plot_result3D import plot_result3d
from Load_Save_Data import load_handwriting_data, load_LinearMotion, write_traj_gen, read_traj_gen, load_data
import DMP
import Obstacle_avoid
# from KeyPoints import plot_keypoints, keypoint_finder, velocity_calculator
from kalman_filter import kalman_filter
from variance_calculation import deviation_calculator  # for calculating the deviation

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
        # KV = 7  # Set Damping gain manuel
        alpha = param_config['alpha']     # Decay factor
        dt = param_config['dt']     # Duration of time step
        nbVar = 1     # Number of variables for the radial basis function [s] (decay term)
        nbDim = 3     # Dimension of movement [x, y, z] # data_train[0].shape[0]

    param = Param()  # instantiate an object

    """
    load training data (Trajectories of Hand/Wrist)
    """
    # load real training data
    # data_train = load_data(param.nb_Samples)
    # print(data_train[0].shape[1])
    # print(data_train[1][1][2])
    # convert meter to millimeter
    # data_train = m_TO_mm(data_train)

    # load handwriting data for test
    data_path_handwrite = param_config['data_handwrite_path']
    data_train = load_handwriting_data(param.nb_Samples, data_path_handwrite)

    """
    Resample
    """
    # set interpolation function (use interpolation function returned by `interp1d`)
    data_size = data_train[0].shape[1]
    x_train = np.arange(data_size)  # independent variable of interpolation function
    f_interpolate = interp1d(x_train, data_train[0], kind='cubic')  # f(x) is data_train

    # independent variable of the
    x_interpolate = np.linspace(x_train.min(), x_train.max(), param.nb_Data)
    # x_interpolate = np.linspace(x_train.min(), x_train.max(), num_interpolate)

    # resampling
    data_resample = f_interpolate(x_interpolate)  # position(x,y,z): use interpolation function
    data_resample_list = []
    data_resample_list.append(data_resample)
    data_train = copy.deepcopy(data_resample_list)

    """
    add gaussian white noise manually
    """
    mean_value = 0
    variance = 0.0005
    standard_deviation = np.sqrt(variance)
    # for i in range(3):
    #     for j in range(param.nb_Data):
    #         data_train[0][i, j] += np.random.normal(mean_value, standard_deviation)

    # load linear motion trajectory for testing obstacle avoidance
    # data_train = load_LinearMotion()

    """
    Preprocess the raw training trajectory with Kalman Filter
    """
    data_train_raw = copy.deepcopy(data_train)  # the raw data do not filter through kalman filter
    # data_train = kalman_filter(data_train)  # the raw data filtered through kalman filter

    """
    Use training data to calculate the external force term 
    """
    Force_external, sIn, Force_NL, s, position_end = DMP.learnFromDemo(param, data_train)

    '''
    Use calculated external force to generate the new Force
    '''
    Weight, currF, force_mean = DMP.generateForce(param, sIn, Force_external)

    '''
    Set Scaling Factor to adjust calculated external force according to new start and goal position
    '''
    # get start & end position of training data
    position_start = np.atleast_2d([data_train[0][0][0], data_train[0][1][0], data_train[0][2][0]]).T
    print('Start position of training data: ', position_start)
    print('End position of training data: ', position_end[0])

    # set new start & end position
    # S: two times
    # position_start_new = np.atleast_2d([11, 17, 2]).T  # here set the start position
    # position_goal_new = np.atleast_2d([-12, -14, 2]).T
    # S: half
    # position_start_new = np.atleast_2d([2.9, 4.25, 2]).T  # here set the start position
    # position_goal_new = np.atleast_2d([-3, -3.5, 2]).T

    # S: Change direction
    # position_start_new = np.atleast_2d([-16, -16, 2]).T  # here set the start position
    # position_goal_new = position_end[0]

    # A
    # position_start_new = np.atleast_2d([-12, -4, 2]).T  # here set the start position
    # position_goal_new = np.atleast_2d([-10, -3, 2]).T

    # B
    # position_start_new = np.atleast_2d([-15, -1, 2]).T  # here set the start position
    # position_goal_new = np.atleast_2d([-14.5, -0.5, 2]).T

    # C
    # position_start_new = np.atleast_2d([-23, -20, 2]).T  # here set the start position
    # position_goal_new = np.atleast_2d([-13, -10, 2]).T

    # D
    # position_start_new = np.atleast_2d([11, -18, 2]).T  # here set the start position
    # position_goal_new = np.atleast_2d([12, -21, 2]).T

    # F
    # position_start_new = np.atleast_2d([-7, -9, 2]).T  # here set the start position
    # position_goal_new = np.atleast_2d([20, -20, 2]).T

    # G
    # position_start_new = position_start
    # position_goal_new = np.atleast_2d([5, -5, 2]).T

    # here set the start position
    position_start_new = position_start
    position_goal_new = position_end[0]

    # For test real training trajectory
    # position_start_new = np.atleast_2d([0.7, 0.7, 0.3]).T
    # position_goal_new = np.atleast_2d([0.7, -0.7, 0.3]).T
    # position_start_new = np.atleast_2d([0.7, -0.7, 0.3]).T
    # position_goal_new = np.atleast_2d([0.7, 0.7, 0.25]).T

    # position_start_new = np.atleast_2d([700, 700, 300]).T
    # position_goal_new = np.atleast_2d([700, -700, 300]).T
    # convert start point and end point
    # position_start_new = np.atleast_2d([700, -700, 300]).T
    # position_goal_new = np.atleast_2d([700, 700, 250]).T

    # position_start_new = np.atleast_2d([-7, -9, -1]).T  # for linear motion
    # position_goal_new = np.atleast_2d([20, -20, 1]).T

    # position_start_new = np.atleast_2d([0.9, -0.35, 0.2]).T  # for linear motion
    # position_goal_new = np.atleast_2d([0.55, 0.2, 0.5]).T

    # scaling generate external force
    # Force_adjust = currF
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
    plot_obstacle = False  # plot obstacle or not
    plot_result3d(param, s, trajectory_gen, sIn, Force_NL, Force_adjust, Weight, position_Obstacle,
                  plot_obstacle, force_mean, data_train_raw)

    # plot_result3d(param, s, trajectory_gen, sIn, Force_NL, currF, Weight, position_Obstacle)

    '''
    Write generated trajectory into a text file
    '''
    name_file = 'trajectory_gen/traj_gen_21_inverse.txt'
    trajectory_gen = trajectory_gen / 1000
    # write_traj_gen(trajectory_gen.T, name_file)

    '''
    Read generated trajectory into a text file
    '''
    # data_read = read_traj_gen(name_file)
    # print(data_read[0])
    # print(data_read.shape)

    '''
    Calculate the Deviation
    '''
    deviation = deviation_calculator(data_train, trajectory_gen)
    print(deviation)



