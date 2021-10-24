import numpy as np
import scipy.io  # for reading mat files
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy.matlib
from scipy import stats  # for calculating gaussian distribution
import copy
from init_GMM_timeBased import init_GMM_timeBased
# from plot_result import plot_result
from plot_result3D import plot_result3d
import cv2  # opencv

def obstacleAvoidance(position_Obstacle, position_tcp, velocity, Gamma, Beta, dot_phi_max):
    # Parameters:
    # Gamma = 1000
    # Beta = 20/np.pi
    # position_Obstacle = np.atleast_2d([6, 6.2, 6.3]).T
    # position_tcp = np.atleast_2d([5.5, 5.5, 5.5]).T
    # velocity = np.atleast_2d([2, 2, 2]).T

    '''
    Calculate the Rotation Matrix:
    '''
    # Calculate the Rotation Axis
    vector_position = position_Obstacle - position_tcp
    Rot_axis = np.cross(vector_position.T, velocity.T)
    # print(Rot_axis[0])

    # Calculate the Rotation Vector for Rodrigues' rotation formula
    angle_rot = np.pi/2
    Rot_vector = Rot_axis/angle_rot

    # Calculate the Rotation Matrix
    Rot = cv2.Rodrigues(Rot_vector)

    '''
    Calculate the steering angle phi 
    '''
    len1 = np.linalg.norm(vector_position)
    len2 = np.linalg.norm(velocity)
    # phi = np.arccos(vector_position.T.dot(velocity) / (len1 * len2))
    phi = np.arccos(np.clip(vector_position.T.dot(velocity) / (len1 * len2), -1, 1))  # solve problem arccos at 0 or 180
    # print(phi[0][0])

    '''
    Calculate the Term for Obstacle Avoidance
    '''
    # if phi[0][0] == 0.0:
    #     # term_obstacle = Rot[0].dot(velocity)*dot_phi_max
    #     term_obstacle = np.ones([3, 1]) * dot_phi_max
    # else:
    #     term_obstacle = Gamma*Rot[0].dot(velocity)*phi*np.exp(-Beta*phi)

    if len1 <= 10:  # avoid obstacle when move near to the obstacle
        # original model:
        # term_obstacle = Gamma * Rot[0].dot(velocity) * phi * np.exp(-Beta * phi)

        # use modified model2:
        if phi[0][0] == 0.0:
            term_obstacle = Rot[0].dot(velocity) * 46 * stats.norm.pdf(phi, 0.06, 0.32)*0.1  # try model2: norm distribution
        else:
            term_obstacle = Rot[0].dot(velocity) * 46 * stats.norm.pdf(phi, 0.06, 0.32)
    else:
        term_obstacle = np.zeros([3, 1])
    # print('Term of Obstacle Avoidance', term_obstacle)

    return term_obstacle

def model(Gamma, Beta):
    """
    :@brief: show the model of obstacle avoidance
    :param Gamma: constant
    :param Beta: constant
    :return:
    """
    # define steering angle phi
    phi = np.linspace(-np.pi/2, np.pi/2, 200)  # rad
    phi_degree = phi*180/np.pi

    # model of obstacle avoidance
    dot_phi = Gamma*phi*np.exp(-Beta*abs(phi))

    # plot the model of obstacle avoidance
    # plt.figure()
    # plt.plot(phi_degree, dot_phi, 'red')
    # plt.title('Model of Obstacle Avoidance', fontsize='medium', fontweight='bold')
    # plt.xlabel('$\phi$: steering angle (degree)', fontsize=12)
    # plt.ylabel('$\dot{\phi}$', fontsize=12)
    # plt.grid()
    # plt.show()

    # find the maximum and minimum value from dot_phi
    dot_phi_max = np.amax(dot_phi)
    # dot_phi_min = np.amin(dot_phi)
    # print('maximum: ', dot_phi_max)
    # print('minimum: ', dot_phi_min)

    return dot_phi_max


def model2(Gamma, Beta):
    """
    :@brief: show the model of obstacle avoidance
    :param Gamma: constant
    :param Beta: constant
    :return:
    """
    # define steering angle phi
    phi = np.linspace(-np.pi/2, np.pi/2, 200)  # rad
    phi_degree = phi*180/np.pi

    # model of obstacle avoidance
    dot_phi = Gamma*phi*np.exp(-Beta*abs(phi))
    # dot_phi2 = 60 * np.exp(-Beta * abs(phi))
    dot_phi3 = 46 * stats.norm.pdf(phi, 0.06, 0.32)

    # # plot the model of obstacle avoidance
    # plt.figure()
    # # the original differential equation (model) of obstacle avoidance:
    # plt.plot(phi_degree, dot_phi, 'red', label='model1')
    # # plt.plot(phi_degree, dot_phi2, 'blue')
    # # the modified differential equation (model) of obstacle avoidance:
    # plt.plot(phi_degree, dot_phi3, 'blue', label='model2')

    # plt.title('Model of Obstacle Avoidance', fontsize='medium', fontweight='bold')
    # plt.xlabel('$\phi$: steering angle (degree)', fontsize=12)
    # # plt.ylabel('$\dot{\phi}$', fontsize=12)
    # plt.ylabel('$f(\phi)$', fontsize=12)
    # plt.legend(loc='best')
    # plt.grid()
    # plt.show()

    # find the maximum and minimum value from dot_phi
    dot_phi_max = np.amax(dot_phi)
    # dot_phi_min = np.amin(dot_phi)
    # print('maximum: ', dot_phi_max)
    # print('minimum: ', dot_phi_min)

    return dot_phi_max





