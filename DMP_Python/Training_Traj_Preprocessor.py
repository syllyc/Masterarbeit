import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

def velocity_calculator(param, data_train):
    """
    Calculate the Velocity (and acceleration) to find the keypoints of the training trajectory
    :param param: parameters for calculation
    :param data_train: training trajectory
    :return: vel_train: calculated velocity
    """
    vel_train = []  # List to save velocity
    acc_train = []  # List to save acceleration
    for i in range(param.nb_Samples):  # number of the demonstration
        # number of each Demonstration(size of real data)
        # data_size = data_train[i].shape[1]

        # Calculate the velocity of the training trajectory
        dx = np.gradient(data_train[i][0, :]) / param.dt  # calculate velocity in x-axis
        dy = np.gradient(data_train[i][1, :]) / param.dt  # calculate velocity in y-axis
        dz = np.gradient(data_train[i][2, :]) / param.dt  # calculate velocity in z-axis

        # Calculate the acceleration of the training trajectory
        ddx = np.gradient(dx) / param.dt  # calculate acceleration in x-axis
        ddy = np.gradient(dy) / param.dt  # calculate acceleration in y-axis
        ddz = np.gradient(dz) / param.dt  # calculate acceleration in z-axis

        # construct a data matrix
        v_element = np.vstack((dx, dy, dz))  # velocity for each demonstration
        a_element = np.vstack((ddx, ddy, ddz))  # acceleration for each demonstration

        # append each velocity and acceleration matrix into the List
        vel_train.append(deepcopy(v_element))
        acc_train.append(deepcopy(a_element))

    return vel_train


def data_train_segment(data_train, keypoints_id):
    """
    Use Keypoints to segment the training trajectory
    :param data_train: training data
    :param keypoints_id: ID of keypoints
    :return: a list of segmented training trajectory
    """
    # calculate the the number of segments according to the number of keypoints
    number_segments = len(keypoints_id) - 1
    print('number of segments:', number_segments)
    data_train_set = []  # a list includes the segments of the training trajectory
    for i in range(number_segments):
        data_segment = data_train[0][:, keypoints_id[i]: keypoints_id[i + 1] + 1]
        data_train_set.append(deepcopy(data_segment))

    return data_train_set  # return the segmented training trajectory


# def keypoint_selector(v_train):

def length_calculator(p_train):
    """
    Calculate the total distance of the training trajectory
    :param p_train: the position calculated from the training data
    :return: total distance of the training trajectory
    """
    # get the number of the training data
    num_points = p_train.shape[1]

    # calculate the total distance
    traj_distance = 0
    for i in range(0, num_points - 1):
        traj_distance += abs(p_train[0, i] - p_train[0, i + 1])
    print("total distance of trajecotry is: ", traj_distance)

    # calculate the mean distance
    distance_mean = traj_distance / (num_points - 1)

    return distance_mean, traj_distance

def keypoint_extractor(data_train, keypoint_ID):
    """
    Use the Keypoints ID to Find the Keypoint from the training trajectory
    :param data_train: training data
    :param keypoint_ID: Index of the Keypoints
    :return: keypoints_list: A List of Keypoints
    """
    num_keypoints = len(keypoint_ID)
    keypoints_list = []
    for i in range(num_keypoints):
        keypoint = np.atleast_2d([data_train[0][0, keypoint_ID[i]],
                                  data_train[0][1, keypoint_ID[i]],
                                  data_train[0][2, keypoint_ID[i]]])
        keypoints_list.append(deepcopy(keypoint))

    return keypoints_list


# def kp_find_vel(num_points, p_train, threshold_eucl, v_train):
#     keypoint_ID_x = []  # list to save the keypoints select from the x direction
#     threshold_v_x = 0.002  # threshold for x-axis
#     for i in range(1, num_points - 1):
#         # set keypoint
#         waypoint = np.array([p_train[0, i], p_train[1, i]])
#         if np.linalg.norm(waypoint - keypoint) > threshold_eucl:
#             if abs(v_train[0, i]) > threshold_v_x and abs(v_train[0, i + 1]) > threshold_v_x:
#                 if v_train[0, i] * v_train[0, i + 1] <= 0:
#                     keypoint_ID_x.append(deepcopy(i))
#                     keypoint = np.array([p_train[0, i], p_train[1, i]])
#                 if np.linalg.norm(waypoint - keypoint_last) <= threshold_eucl and i > num_points - 50:
#                     break
#     return keypoint_ID_x

def keypoint_finder(data_train, p_train, v_train):
    """
    Find Keypoints from the Training Trajectory
    :param data_train: the Training Trajectory
    :param p_train: the position calculated from the training data
    :param v_train: the velocity calculated from the training data
    :return: the selected list of keypoints
    """

    # Get the number of the training data
    num_points = p_train.shape[1]

    # Set Parameters
    keypoint_ID_x = []  # list to save the ID of keypoints select from the x direction
    keypoint_ID_y = []  # list to save the ID of keypoints select from the y direction
    keypoint_ID_z = []  # list to save the ID of keypoints select from the z direction

    # keypoints_list_x = []  # list to save the keypoints select from the x direction
    # keypoints_list_y = []  # list to save the keypoints select from the y direction
    # keypoints_list_z = []  # list to save the keypoints select from the z direction

    '''
    Calculate the threshold for the Clustering
    '''
    # Get the total position of the training trajectory and calculate the mean distance for each step
    distance_mean, traj_distance = length_calculator(p_train)  # the mean distance for each step
    # print("mean distance of trajectory is:", distance_mean)

    # Set the threshold for Euclidean Clustering
    # threshold_eucl = distance_mean * 20  # set the threshold based ont the number of the time steps
    threshold_eucl = traj_distance / 10  # set the threshold 10% of the total length

    '''
    Initialize the first and last keypoints
    '''
    # set the start point as the first keypoint
    keypoint = np.array([p_train[0, 0], p_train[1, 0]])
    # set the last point as the last keypoint
    keypoint_last = np.array([p_train[0, num_points - 1], p_train[1, num_points - 1]])
    # select from the start point

    '''
    Find the Keypoints along x-axis: the direction of velocity changed in x-axis 
    '''
    threshold_v_x = 0.002  # threshold for x-axis
    for i in range(1, num_points - 1):
        # set keypoint
        waypoint = np.array([p_train[0, i], p_train[1, i]])
        if np.linalg.norm(waypoint - keypoint) > threshold_eucl:
            if abs(v_train[0, i]) > threshold_v_x and abs(v_train[0, i + 1]) > threshold_v_x:
                if v_train[0, i] * v_train[0, i + 1] <= 0:
                    keypoint_ID_x.append(deepcopy(i))  # ID of keypoint
                    keypoint = np.array([p_train[0, i], p_train[1, i]])
                if np.linalg.norm(waypoint - keypoint_last) <= threshold_eucl and i > num_points - 50:
                    break
    # Get the keypoints list from x-axis
    keypoints_list_x = keypoint_extractor(data_train, keypoint_ID_x)

    '''
    Find the Keypoints along y-axis: the direction of velocity changed in y-axis
    '''
    keypoint = np.array([p_train[0, 0], p_train[1, 0]])
    threshold_v_y = 0.015  # threshold for y-axis
    # threshold_v_y = 0.6  # threshold for y-axis
    for i in range(1, num_points - 1):
        # set keypoint
        waypoint = np.array([p_train[0, i], p_train[1, i]])
        if np.linalg.norm(waypoint - keypoint) > threshold_eucl:
            if abs(v_train[1, i]) > threshold_v_y and abs(v_train[1, i + 1]) > threshold_v_y:
                if v_train[1, i] * v_train[1, i + 1] <= 0:
                    keypoint_ID_y.append(deepcopy(i))
                    keypoint = np.array([p_train[0, i], p_train[1, i]])
                if np.linalg.norm(waypoint - keypoint_last) <= threshold_eucl and i > num_points - 50:
                    break
    # Get the keypoints list from y-axis
    keypoints_list_y = keypoint_extractor(data_train, keypoint_ID_y)

    '''
    Find the Keypoints along z-axis: the direction of velocity changed in z-axis
    '''
    keypoint = np.array([p_train[0, 0], p_train[1, 0]])
    threshold_v_z = 0.02  # threshold for y-axis
    for i in range(1, num_points - 1):
        # set keypoint
        waypoint = np.array([p_train[0, i], p_train[1, i]])
        if np.linalg.norm(waypoint - keypoint) > threshold_eucl:
            if abs(v_train[2, i]) > threshold_v_z and abs(v_train[2, i + 1]) > threshold_v_z:
                if v_train[2, i] * v_train[2, i + 1] <= 0:
                    keypoint_ID_z.append(deepcopy(i))
                    keypoint = np.array([p_train[0, i], p_train[1, i]])
                if np.linalg.norm(waypoint - keypoint_last) <= threshold_eucl and i > num_points - 50:
                    break
    # Get the keypoints list from z-axis
    keypoints_list_z = keypoint_extractor(data_train, keypoint_ID_z)
    print("keypoints in z-axis", keypoint_ID_z)

    '''
    Combine the found Keypoints ID List (keypoint_ID_x, keypoint_ID_y and keypoint_ID_z)
    '''
    # keypoints in x-axis + keypoints in y-axis
    keypoint_ID = list(set(keypoint_ID_x).union(set(keypoint_ID_y)))

    # note: the first and last keypoints are the start and end point so:
    keypoint_ID.append(0)
    keypoint_ID.append(num_points - 1)

    # add keypoint found in z-axis
    # keypoint_ID.append(keypoint_ID_z[0])

    # sort the list from small to large
    keypoint_ID = sorted(keypoint_ID)

    '''
    Use the Keypoints ID to Find the Keypoint from the training trajectory
    '''
    keypoints_list = keypoint_extractor(data_train, keypoint_ID)

    return keypoint_ID, keypoints_list, threshold_eucl, keypoints_list_x, keypoints_list_y, keypoints_list_z

def change_rate_calculator(keypoints_id, v_train):
    """
    Calculate the change rate at the keypoints for judging which keypoints should be removed
    :param keypoints_id: the ID of the keypoint which need to select
    :param v_train:
    :return:
    """
    num_length = 5  # number of the points before and after the keypoints

    # get the ID of first point
    ID_start = keypoints_id - num_length
    ID_end = keypoints_id + num_length

    # calculate the change rate around the keypoint
    change_rate = 0
    for i in range(ID_start, ID_end):
        change_rate = change_rate + v_train[i, 0]

    print('change rate is: ', change_rate)
    return change_rate


def keypoints_selector(keypoints_list, threshold_keypoint, keypoints_id, v_train):
    """
    Keypoints selector: Remove keypoints which are too close
    :param keypoints_list: List of found Keypoints
    :return: Selected Keypoints
    """
    num_keypoints = len(keypoints_list)
    duplicates_ID = []  # A list to save the ID of duplicates elements in keypoints list
    for i in range(num_keypoints-2):  # the last point must be keep stay, so -2
        # calculate the distance between each two keypoint, if too close than remove the second one
        if np.linalg.norm(keypoints_list[i] - keypoints_list[i+1]) <= threshold_keypoint:
            keypoints_list[i+1] = deepcopy(keypoints_list[i])  # find the IDs of elements which are duplicated
            duplicates_ID.append(i + 1)

    # remove duplicates from list
    keypoints_list = [keypoints_list[i] for i in range(0, len(keypoints_list), 1) if i not in duplicates_ID]
    keypoints_id = [keypoints_id[i] for i in range(0, len(keypoints_id), 1) if i not in duplicates_ID]

    return keypoints_list, keypoints_id

def circle_generator(circle_center, radius):
    """
    Generate a Circle Trajectory
    parm: circle_center
          radius
    return: the circle trajectory
    """
    # set the range of the circle: 360 degrees
    radian_range = np.arange(0, 6.3, 0.02)
    # array to save the circle trajectory
    trajectory_circle = np.zeros((2, radian_range.shape[0]))
    i = 0
    for th in radian_range:
        trajectory_circle[0, i] = circle_center[0] + radius * np.sin(th)
        trajectory_circle[1, i] = circle_center[1] + radius * np.cos(th)
        i = i + 1
    return trajectory_circle


def plot_keypoints(data_train, keypoints_list):
    # number of training data
    num_points = data_train[0].shape[1]
    # print(num_points)
    plt.figure()

    '''
    Plot the Trajectory of Training Trajectory in 2D
    '''
    # plt.plot(data_train[0][0], data_train[0][1], 'gray', label='training data')
    for i in range(1, num_points - 1):
        plt.scatter(data_train[0][0, i], data_train[0][1, i], marker='o', color='gray')

    '''
    Plot the Start Position and End Position of the Trajectory in 2D
    '''
    # # plot the start Position
    # plt.scatter(data_train[0][0, 0], data_train[0][1, 0], marker='X', color='red', s=60)
    # # Plot the end Position
    # plt.scatter(data_train[0][0, num_points-1], data_train[0][1, num_points-1], marker='X', color='red', s=60)

    '''
    Plot the Euclidean Distance Circle
    '''
    # generate the circile trajectory
    circle_center = np.array([data_train[0][0, 0], data_train[0][1, 0]])
    radius = 1.5
    traj_circle = circle_generator(circle_center, radius)
    # plt.plot(traj_circle[0, :], traj_circle[1, :], 'green')

    '''
    Plot the Keypoints
    '''
    for i in keypoints_list:
        plt.scatter(data_train[0][0, i], data_train[0][1, i], marker='o', color='blue')

    plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.xlabel('$X/mm$')
    plt.ylabel('$Y/mm$')
    plt.legend(loc='best')
    plt.axis('equal')
    plt.show()



