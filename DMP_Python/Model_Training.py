import numpy as np
import scipy.io  # for reading mat files
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy.matlib  # Matrix library
from scipy import stats  # for calculating gaussian distribution
import copy
from plot_result3D import plot_result3d
import cv2  # opencv
from Obstacle_avoid import obstacleAvoidance


def init_GMM_timeBased(sIn, param):
    """
    Calculate a Gaussian Mixture Model
    :param sIn: phase variable
    :param param: initial parameters
    :return: Gaussian Mixture Model
    """
    # Parameters
    params_diagRegFact = 1E-4
    # divide the phase variable into several pieces
    TimingSep = np.linspace(np.amin(sIn), np.amax(sIn), param.nbStates+1)

    Priors = np.zeros(param.nbStates)
    Mu = np.zeros(param.nbStates)
    Sigma = np.zeros(param.nbStates)

    for i in range(param.nbStates):
        # find the time points of each phase variable section
        idtmp = np.argwhere((sIn >= TimingSep[i]) & (sIn < TimingSep[i+1]))
        Priors[i] = np.size(idtmp)  # the number in each section
        Mu[i] = np.mean(sIn[idtmp])  # calculate the mean value of each section
        Sigma[i] = np.var(sIn[idtmp])  # calculate the variance of each section
        # optional regularization term to avoid numerical instability
        Sigma[i] = Sigma[i] + np.eye(1)*params_diagRegFact
    Priors = Priors / np.sum(Priors)

    class GMM_model:
        priors = Priors
        mu = Mu
        sigma = Sigma
        std = np.sqrt(Sigma)
        params_diagregfact = params_diagRegFact

    gmm_model = GMM_model()

    return gmm_model
    # print(param.Priors)


def learnFromDemo(param, data_train):
    """
    :@brief: Use training data to calculate the external force term
    :param[in] param: Initial parameters for calculate the external force term
    :param[in] data_train: Training data
    :return: ForceNL_concatenate: The external force term
    """

    '''
    Parameters for the calculation of DMP
    '''
    # set the structure of training data
    posID = np.arange(param.nbDim)  # ID of Position: here first three rows(0,1,2) are x, y and z separately
    # print('ID of position is: ', posID)
    velID = np.arange(param.nbDim, 2 * param.nbDim, 1)
    # print('ID of velocity is: ', velID)
    accID = np.arange(2 * param.nbDim, 3 * param.nbDim, 1)
    # print('ID of acceleration is: ', accID)

    # set number of interpolate function
    # num_interpolate = 100

    # set phase variable
    time_axis = np.arange(param.nb_Data)
    sIn = np.exp(-param.alpha * time_axis * param.dt)  # sIn: s --> phase variable

    '''
    Calculate the external force from demonstrations
    '''
    # init Parameters:
    Force_NL = []  # list variable to save the
    s = []  # list to construct the Demonstration data as [x,dx,ddx](9 x DataSize)
    position_end = []  # list to save the last point of each Demonstration
    position_start = np.atleast_2d([data_train[0][0, 0], data_train[0][1, 0], data_train[0][2, 0]]).T

    # data_train[0] = data_train[0][:, 0:50]  # for test
    for i in range(param.nb_Samples):  # number of the demonstration
        # number of each Demonstration(size of real data)
        data_size = data_train[i].shape[1]
        # get the target position:
        point_last = np.atleast_2d([data_train[i][0, data_size-1],     # the last observed point in x-axis
                                    data_train[i][1, data_size-1],     # the last observed point in y-axis
                                    data_train[i][2, data_size-1]]).T  # the last observed point in z-axis
        position_end.append(copy.deepcopy(point_last))  # save the last point of each Demonstration

        # set interpolation function (use interpolation function returned by `interp1d`)
        x_train = np.arange(data_size)  # independent variable of interpolation function
        f_interpolate = interp1d(x_train, data_train[i], kind='cubic')  # f(x) is data_train

        # independent variable of the
        x_interpolate = np.linspace(x_train.min(), x_train.max(), param.nb_Data)
        # x_interpolate = np.linspace(x_train.min(), x_train.max(), num_interpolate)

        # resampling
        demo_resample = f_interpolate(x_interpolate)  # position(x,y,z): use interpolation function
        dx = np.gradient(demo_resample[0])/param.dt  # calculate velocity in x-axis
        dy = np.gradient(demo_resample[1])/param.dt  # calculate velocity in y-axis
        dz = np.gradient(demo_resample[2])/param.dt  # calculate velocity in z-axis

        ddx = np.gradient(dx)/param.dt  # calculate acceleration in x-axis
        ddy = np.gradient(dy)/param.dt  # calculate acceleration in y-axis
        ddz = np.gradient(dz)/param.dt  # calculate acceleration in z-axis

        # construct a data matrix
        s_element = np.vstack((demo_resample, dx, dy, dz, ddx, ddy, ddz))

        # calculate the external force term through demonstration data
        # f_element = (s_element[accID, :]-(np.matlib.repmat(point_last, 1, param.nb_Data) - s_element[posID, :])*param.KP
        #              + s_element[velID, :] * param.KV) / np.matlib.repmat(sIn, param.nbDim, 1)

        # calculate the external force term by using modified DMP
        f_element = (s_element[accID, :] + s_element[velID, :]*param.KV)/param.KP \
                    - (np.matlib.repmat(point_last, 1, param.nb_Data) - s_element[posID, :]) \
                    + (np.matlib.repmat(point_last, 1, param.nb_Data) - np.matlib.repmat(position_start, 1, param.nb_Data))*sIn

        s.append(copy.deepcopy(s_element))
        Force_NL.append(copy.deepcopy(f_element))

    # concatenate Force_NL to a new matrix
    if param.nb_Samples == 1:
        ForceNL_concatenate = Force_NL[0]
    else:
        ForceNL_concatenate = np.hstack((Force_NL[0], Force_NL[1]))
        for i in range(2, param.nb_Samples):
            ForceNL_concatenate = np.hstack((ForceNL_concatenate, Force_NL[i]))
    # print(ForceNL_concatenate.shape)

    return ForceNL_concatenate, sIn, Force_NL, s, position_end

def force_connector(param, Force_NL):
    """
    concatenate List Force_NL to matrix Form
    :param param:
    :param Force_NL: A List save the calculated external force, each element is for a demonstration
    :return: ForceNL_concatenate: The concatenated Force Matrix
    """
    # concatenate Force_NL to a new matrix
    if param.nb_Samples == 1:
        ForceNL_concatenate = Force_NL[0]
    else:
        ForceNL_concatenate = np.hstack((Force_NL[0], Force_NL[1]))
        for i in range(2, param.nb_Samples):
            ForceNL_concatenate = np.hstack((ForceNL_concatenate, Force_NL[i]))
    # print(ForceNL_concatenate.shape)

    return ForceNL_concatenate


def external_force_calculator(param, data_train, modified_DMP):
    """
    :@brief: Use training data to calculate the external force term
    :param[in] param: Initial parameters for calculate the external force term
    :param[in] data_train: Training data
    :param[in] modified_DMP: choose to use modified DMP or original DMP
    :return: ForceNL_concatenate: The external force term
    """

    """
    Parameters for the calculation of DMP
    """
    # set the structure of training data
    posID = np.arange(param.nbDim)  # ID of Position: here first three rows(0,1,2) are x, y and z separately
    # print('ID of position is: ', posID)
    velID = np.arange(param.nbDim, 2 * param.nbDim, 1)
    # print('ID of velocity is: ', velID)
    accID = np.arange(2 * param.nbDim, 3 * param.nbDim, 1)
    # print('ID of acceleration is: ', accID)

    # set number of interpolate function
    # num_interpolate = 100

    # set phase variable
    time_axis = np.arange(param.nb_Data)
    sIn = np.exp(-param.alpha * time_axis * param.dt)  # sIn: s --> phase variable

    """
    Calculate the external force from demonstrations
    """
    # init Parameters:
    Force_NL = []  # list variable to save the
    position_start = np.atleast_2d([data_train[0, 0], data_train[1, 0], data_train[2, 0]]).T
    # s = []  # list to construct the Demonstration data as [x,dx,ddx](9 x DataSize)
    # position_end = []  # list to save the last point of each Demonstration

    # data_train[0] = data_train[0][:, 0:50]  # for test
    # for loop is used for multiple demonstrations, note: here we only use one demonstration
    for i in range(param.nb_Samples):  # number of the demonstration
        # get the number of each Demonstration(size of real data)
        data_size = data_train.shape[1]
        # get the target position:
        point_last = np.atleast_2d([data_train[0, data_size - 1],  # the last observed point in x-axis
                                    data_train[1, data_size - 1],  # the last observed point in y-axis
                                    data_train[2, data_size - 1]]).T  # the last observed point in z-axis

        # position_end.append(copy.deepcopy(point_last))  # save the last point of each Demonstration

        # set interpolation function (use interpolation function returned by `interp1d`)
        x_train = np.arange(data_size)  # independent variable of interpolation function
        f_interpolate = interp1d(x_train, data_train, kind='cubic')  # f(x) is data_train

        # independent variable of the interpolation function
        x_interpolate = np.linspace(x_train.min(), x_train.max(), param.nb_Data)
        # x_interpolate = np.linspace(x_train.min(), x_train.max(), num_interpolate)

        # Resampling
        demo_resample = f_interpolate(x_interpolate)  # position(x,y,z): use interpolation function

        '''
        calculate the velocity and acceleration of the training trajectory:
        '''
        # calculate the velocity of the training trajectory:
        dx = np.gradient(demo_resample[0]) / param.dt  # velocity in x-axis
        dy = np.gradient(demo_resample[1]) / param.dt  # velocity in y-axis
        dz = np.gradient(demo_resample[2]) / param.dt  # velocity in z-axis
        # calculate the acceleration of the training trajectory:
        ddx = np.gradient(dx) / param.dt  # acceleration in x-axis
        ddy = np.gradient(dy) / param.dt  # acceleration in y-axis
        ddz = np.gradient(dz) / param.dt  # acceleration in z-axis

        # construct a data matrix
        s_element = np.vstack((demo_resample, dx, dy, dz, ddx, ddy, ddz))  # a  matrix (9 x nb_Data)

        '''
        Use Spring-Damped Dynamic Equation to calculate the external Force Term
        '''
        if modified_DMP:  # True or False
            # calculate the external force term by using modified DMP
            f_element = (s_element[accID, :] + s_element[velID, :] * param.KV) / param.KP \
                         - (np.matlib.repmat(point_last, 1, param.nb_Data) - s_element[posID, :]) \
                         + (np.matlib.repmat(point_last, 1, param.nb_Data) - np.matlib.repmat(position_start, 1,
                                                                                              param.nb_Data)) * sIn
        else:
            # calculate the external force term through demonstration data
            f_element = (s_element[accID, :]-(np.matlib.repmat(point_last, 1, param.nb_Data) - s_element[posID, :])*param.KP
                         + s_element[velID, :] * param.KV) / np.matlib.repmat(sIn, param.nbDim, 1)

        # s.append(copy.deepcopy(s_element))  # A List to save position, velocity and acceleration of the training data

        Force_NL.append(copy.deepcopy(f_element))  # A List to save calculated external force of the training data

    return Force_NL, sIn


def skill_model_generator(param, sIn, Force_NL):
    """
    :@brief: Use calculated external force to generate the new Force
    :param param:
    :param sIn: phase variable
    :param ForceNL_concatenate:
    :return:
    """

    '''
    concatenate Force_NL to a new matrix
    '''
    ForceNL_concatenate = force_connector(param, Force_NL)

    '''
    Setting of the basis functions and reproduction:
    '''
    # initial parameters of GMM
    gmm_model = init_GMM_timeBased(sIn, param)

    # Set Sigma manually
    for i in range(param.nbStates):
        gmm_model.sigma[i] = 2E-3
    gmm_model.std = np.sqrt(gmm_model.sigma)

    # Compute activation functions
    Weight = np.zeros((param.nbStates, param.nb_Data))
    for i in range(param.nbStates):
        # get gaussian distribution for each section
        Weight[i, :] = stats.norm.pdf(sIn, gmm_model.mu[i], gmm_model.std[i])
    Weight = Weight / np.matlib.repmat(np.sum(Weight, axis=0), param.nbStates, 1)  # proportion
    # Weight used when have multi Demonstrations(Samples)
    Weight_Samples = np.matlib.repmat(Weight, 1, param.nb_Samples)

    '''
    Nonlinear force profile retrieval
    '''
    X = np.ones((param.nb_Samples * param.nb_Data, 1))  # Input
    Y = copy.deepcopy(ForceNL_concatenate.T)  # Output

    # calculate the Mean Force Matrix
    MeanForce = np.zeros((param.nbDim, param.nbStates))
    for i in range(param.nbStates):
        W = np.diag(Weight_Samples[i, :])
        muf = (np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)).T  # Local Weighted Regression (LWR)
        MeanForce[0][i] = muf[0][0]
        MeanForce[1][i] = muf[1][0]
        MeanForce[2][i] = muf[2][0]

    '''
    calculate the external force at each time step
    '''
    currF = MeanForce.dot(Weight)
    return Weight, currF






