import numpy as np

def skill_model_optimizer(currF, position_start, position_end, position_start_new, position_goal_new):
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
    vector_new = position_goal_new[0:2, :] - position_start_new[0:2, :]  # vector of new trajectory, row vector(x,y,z)
    vector_train = position_end[0:2, :] - position_start[0:2, :]  # vector of training data, row vector(x,y,z)
    # print("position_goal_new", position_goal_new[0:2, :])

    # calculate the size of scaling factor
    scaling_factor_size = np.linalg.norm(position_goal_new - position_start_new) / \
                          np.linalg.norm(position_end - position_start)

    # calculate the rotation angle
    angle_rot = np.arctan2(vector_new[0, 1], vector_new[0, 0]) - np.arctan2(vector_train[0, 1], vector_train[0, 0])
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


