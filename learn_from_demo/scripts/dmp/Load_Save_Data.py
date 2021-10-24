"""
Functions that used to Load Training Data
"""
import numpy as np
import scipy.io  # for reading mat files
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy.matlib
from scipy import stats  # for calculating gaussian distribution
import copy
from init_GMM_timeBased import init_GMM_timeBased
# from plot_result import plot_result

def load_handwriting_data(nb_Samples, data_path_hdwrite):
    """
    load training data set (here is a handwriting task: uppercase letter G)
    Structure of training data:
        13 set of Demonstrations
        each Demonstration has three element: position, velocity, acceleration
        Dimension: two-->(x,y)
    """

    '''
    # load handwriting training data set 
    '''
    data_raw = scipy.io.loadmat(data_path_hdwrite)  # the type of raw data is 'Dictionary'
    # print(data_raw.keys())  # show the keywords of the 'Dictionary'
    data_load = data_raw['demos']
    # print('Shape of data_load: ', data_load.shape)
    data_handwrite = []  # List to save the load handwriting training data, each element is one Demonstration

    for i in range(nb_Samples):
        demos = data_load[0][i]['pos']  # get the first element of training data set (total: 13 sets)
        demos = demos[0][0]  # restructure the data, otherwise the structure is 1x1
        data_handwrite.append(demos)
    # print(demos[0].shape[0])

    '''
    set z-axis trajectory manually 
    '''
    nb_data = demos[0].shape[0]
    # z_train = np.ones(nb_data) * 3

    # set motion in z-axis as sin function
    # x_input = np.arange(nb_data)
    # z_train = np.sin(x_input)/2

    # set motion in z-axis as curve of the first degree
    z_train = np.linspace(-1, 2, nb_data)

    # set motion in z-axis as Parabola
    # x_input = np.linspace(-2, 2, nb_data)
    # z_train = -0.5*x_input**2 + 4
    # z_train = x_input ** 2

    '''
    Add the 3rd dimension to the Handwriting data
    '''
    data_train = []
    data_demo = np.zeros((3, nb_data))

    # print(data_handwrite[1][0])
    for i in range(nb_Samples):
        data_demo[0, :] = data_handwrite[i][0]
        data_demo[1, :] = data_handwrite[i][1]
        data_demo[2, :] = z_train
        data_train.append(copy.deepcopy(data_demo))  # deepcopy:otherwise variables share same memory, error occurs

    return data_train

def load_LinearMotion():
    """
    :@brief: Use a Linear Motion trajectory to test Obstacle Avoidance
    :return:
    """
    # Parameters:
    nb_data = 50
    data_train = []
    data_demo = np.zeros((3, nb_data))
    # direction vector
    direction_vector = np.array([1, 2, 4])
    # point of obstacle
    point = np.array([2, 3, 4])
    # Parametric equation:
    t = np.linspace(-5, 5, nb_data)

    data_demo[0, :] = point[0] + direction_vector[0]*t  # X
    data_demo[1, :] = point[1] + direction_vector[1]*t  # Y
    data_demo[2, :] = point[2] + direction_vector[2]*t  # Z

    data_train.append(copy.deepcopy(data_demo))

    return data_train

def write_traj_gen(trajectory_gen, name_file):
    """
    :@brief: Save generated trajectory in to Text File (.txt)
    :param trajectory_gen: learned trajectory by dmp
    :param name_file: name of saved text file
    :return:
    """
    # print('shape of generated trajectory: ', trajectory_gen.shape[0])
    # name_file = 'trajectory_gen/traj_gen_S.txt'
    np.savetxt(name_file, trajectory_gen, fmt='%.6f ; %.6f ; %.6f')
    print('Info: generated trajectory is saved')

def read_traj_gen(name_file):
    """
    :@brief: Load generated trajectory from Text File (.txt)
    :param name_file: name of loaded text file
    :return: loaded trajectory
    """
    traj_gen = np.loadtxt(name_file, dtype=float, delimiter=';')
    print('Info: generated trajectory is loaded')
    return traj_gen




