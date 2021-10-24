import numpy as np
from scipy.interpolate import interp1d
import numpy.matlib  # Matrix library
from scipy import stats  # for calculating gaussian distribution
import copy
from plot_result3D import plot_result3d
import cv2  # opencv
from Obstacle_avoid import obstacleAvoidance


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


def learnFromDemo2(param, data_train, modified_DMP):
    """
    :@brief: Use training data to calculate the external force term
    :param[in] param: Initial parameters for calculate the external force term
    :param[in] data_train: Training data
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
    s = []  # list to construct the Demonstration data as [x,dx,ddx](9 x DataSize)
    position_end = []  # list to save the last point of each Demonstration
    position_start = np.atleast_2d([data_train[0, 0], data_train[1, 0], data_train[2, 0]]).T

    # data_train[0] = data_train[0][:, 0:50]  # for test
    # for loop is used for multiple demonstrations, note: here we only use one demonstration
    for i in range(param.nb_Samples):  # number of the demonstration
        # get the number of each Demonstration(size of real data)
        data_size = data_train.shape[1]
        # get the target position:
        point_last = np.atleast_2d([data_train[0, data_size - 1],  # the last observed point in x-axis
                                    data_train[1, data_size - 1],  # the last observed point in y-axis
                                    data_train[2, data_size - 1]]).T  # the last observed point in z-axis
        position_end.append(copy.deepcopy(point_last))  # save the last point of each Demonstration

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

        s.append(copy.deepcopy(s_element))  # A List to save position, velocity and acceleration of the training data
        Force_NL.append(copy.deepcopy(f_element))  # A List to save calculated external force of the training data

    # concatenate Force_NL to a new matrix
    if param.nb_Samples == 1:
        ForceNL_concatenate = Force_NL[0]
    else:
        ForceNL_concatenate = np.hstack((Force_NL[0], Force_NL[1]))
        for i in range(2, param.nb_Samples):
            ForceNL_concatenate = np.hstack((ForceNL_concatenate, Force_NL[i]))
    # print(ForceNL_concatenate.shape)

    return ForceNL_concatenate, sIn, Force_NL, s, position_end


def init_GMM_timeBased(sIn, param):
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


def generateForce(param, sIn, ForceNL_concatenate):
    """
    :@brief: Use calculated external force to generate the new Force
    :param param:
    :param sIn: phase variable
    :param ForceNL_concatenate:
    :return:
    """
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
    # Test X_input use different linear function
    # t = np.linspace()
    # X = np.array(sIn)[:, np.newaxis]
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
    return Weight, currF, MeanForce

def adjustForce(param, currF, position_start, position_end, position_start_new, position_goal_new):
    """
    :@brief: Set Scaling Factor to adjust calculated external force according to new start and goal position
    :param param: initial parameters
    :param currF: generate external force
    :param position_start: the first point of training data
    :param position_end: the last point of training data
    :param position_start_new: the new start position
    :param position_goal_new: the new goal position
    :return: Force_adjust
    """

    # get the direction vectors (2D: in x-y plane )
    vector_new = position_goal_new[0:2, :] - position_start_new[0:2, :]  # vector of new trajectory
    vector_train = position_end[0][0:2, :] - position_start[0:2, :]  # vector of training data
    # print("position_goal_new", position_goal_new[0:2, :])

    # calculate the size of scaling factor
    scaling_factor_size = np.linalg.norm(position_goal_new - position_start_new) / \
                          np.linalg.norm(position_end[0] - position_start)

    # calculate the rotation angle
    angle_rot = np.arctan2(vector_new[1][0], vector_new[0][0]) - np.arctan2(vector_train[1][0], vector_train[0][0])
    # angle = angle_rot*180/np.pi

    # calculate the rotation matrix
    Rot = np.atleast_2d([[np.cos(angle_rot), -np.sin(angle_rot), 0],
                         [np.sin(angle_rot), np.cos(angle_rot), 0],
                         [0, 0, 1]])
    # test_rot = Rot[0:2, 0:2].dot(vector_train)

    '''
    # scaling factor set for each dimension
    scaling_factor = np.zeros((param.nbDim, 1))
    for i in range(param.nbDim):
        scaling_factor[i][0] = (position_goal_new[i][0]-position_start_new[i][0]) / \
                               (position_end[0][i][0]-position_start[i][0])

    # adjust generated external force with scaling factor
    Force_adjust = currF * scaling_factor
    '''

    # Use the rotation matrix to change the direction of learned force and zoom it with scaling factor
    # scaling_factor_size = 1
    Force_adjust = Rot.dot(currF) * scaling_factor_size

    return Force_adjust


def generateTrajectory(param, sIn, Force_adjust, position, position_target,
                       position_Obstacle, Gamma, Beta, dot_phi_max, Obstacle_detect):
    """
    :@brief: Use the current achieved DMP to create a new Trajectory (starting from x_0 toward x_goal)
    :param param:
    :param sIn:
    :param Force_adjust:
    :param position: new start position
    :param position_target:
    :return:
    """
    # P_gain and D_gain matrix
    PD_gain = np.hstack((np.eye(param.nbDim) * param.KP, np.eye(param.nbDim) * param.KV))  # Stack arrays horizontally

    # set a threshold to let the trajectory move to the goal position
    num_plan = 0
    goal_thresh = 0.001
    at_goal = False
    max_plan_length = 500
    velocity = np.zeros((param.nbDim, 1))  # init start velocity is zero
    trajectory_gen = []
    position_start = copy.deepcopy(position)
    threshold = 0.1  # threshold for prevent of overshoot

    # parameters for obstacle avoidance
    term_avoid_obstacle = np.zeros([3, 1])
    # Obstacle_detect = True  # Determine if obstacles are detected

    while num_plan < param.nb_Data or (not at_goal and num_plan < max_plan_length):
        # Calculate Obstacle Avoidance Term:
        if num_plan >= 1:  # start from second loop because the initial velocity is zero
            if Obstacle_detect:  # Determine if obstacles are detected
                term_avoid_obstacle = obstacleAvoidance(position_Obstacle, position, velocity, Gamma, Beta, dot_phi_max)

        # Computer acceleration, velocity and position
        if num_plan < param.nb_Data:
            # original DMP
            # acceleration = PD_gain.dot(np.vstack((position_target - position, -velocity))) \
            #                + (Force_adjust[:, num_plan]*sIn[num_plan])[:, np.newaxis] \
            #                + term_avoid_obstacle  # term used for obstacle avoidance

            # modified DMP
            acceleration = PD_gain.dot(np.vstack((position_target - position, -velocity))) \
                           - param.KP * (position_target - position_start) * sIn[num_plan] \
                           + (Force_adjust[:, num_plan]*param.KP)[:, np.newaxis] \
                           + term_avoid_obstacle  # term used for obstacle avoidance

            # for test the stability of the system: without the external force term, only has the PD controller
            # acceleration = PD_gain.dot(np.vstack((position_target - position, -velocity))) \
            #                - param.KP * (position_target - position_start) * sIn[num_plan] \
            #                + term_avoid_obstacle  # term used for obstacle avoidance

        else:
            # original DMP
            acceleration = PD_gain.dot(np.vstack((position_target - position, -velocity)))

            # modified DMP
            # acceleration = PD_gain.dot(np.vstack((position_target - position, -velocity))) \
            #                 - param.KP * (position_target - position_start) * sIn[num_plan]

        velocity = velocity + acceleration * param.dt
        position = position + velocity * param.dt

        '''
        Judge if the current position is close enough to the goal position
        '''
        # if np.linalg.norm(position - position_target) <= threshold:
        #     break

        # If plan is at least minimum length, check to see if the end position is close enough to goal position
        if num_plan >= param.nb_Data:
            at_goal = True
            for k in range(param.nbDim):
                if abs(position[k][0]-position_target[k][0]) > goal_thresh:
                    at_goal = False

        trajectory_gen.append(copy.deepcopy(position))
        num_plan += 1

    # convert List to numpy array
    plan_gen = np.zeros((param.nbDim, len(trajectory_gen)))
    for i in range(len(trajectory_gen)):
        for j in range(param.nbDim):
            plan_gen[j][i] = trajectory_gen[i][j][0]

    return plan_gen


'''
def adjustForce3D(currF, position_start, position_end, position_start_new, position_goal_new):
    """
    :@brief: Set Scaling Factor to adjust calculated external force according to new start and goal position
             Rotation in 3D
    :param currF: generate external force
    :param position_start: the first point of training data
    :param position_end: the last point of training data
    :param position_start_new: the new start position
    :param position_goal_new: the new goal position
    :return: Force_adjust
    """
    # get the vectors
    vector_new = position_goal_new - position_start_new  # vector of new trajectory
    vector_train = position_end[0] - position_start  # vector of training data

    # norm of vectors
    len_new = np.linalg.norm(vector_new)
    len_train = np.linalg.norm(vector_train)

    # calculate the size of scaling factor
    scaling_factor_size = np.linalg.norm(position_goal_new - position_start_new) / \
                          np.linalg.norm(position_end[0] - position_start)

    # calculate the rotation angle
    angle_rot = np.arccos(vector_new.T.dot(vector_train)/(len_new*len_train))

    # calculate the rotation vector
    axis_rot = np.cross(vector_new.T, vector_train.T)  # vertical vector (rotation axis)

    norm_axis_rot = np.linalg.norm(axis_rot.T)  # norm of vertical vector
    vector_rot = (axis_rot/norm_axis_rot)*angle_rot

    # calculate the rotation matrix by using Rodrigues' formula
    Rot = cv2.Rodrigues(vector_rot)
    # test_rot = Rot[0].T.dot(vector_train)

    # Use the rotation matrix to change the direction of learned force and zoom it with scaling factor
    Force_adjust = Rot[0].T.dot(currF) * scaling_factor_size

    return Force_adjust
'''


