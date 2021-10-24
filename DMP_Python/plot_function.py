from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib.pyplot import MultipleLocator

'''
Plot the new generated trajectory
Parameters of function:
    param.nb_Samples: the number of used Demonstrations
    param: the initial settings
    s: data of Demonstrations
    sIn: Phase variable
    Force_NL: force calculate from demonstrations
    currF: force for new trajectory
    H: activation function of GMM
'''

def plot_traj_x(param, data_train, keypoints_list, keypoints_id_list):
    """
    plot the trajectory(motion) in x-axis (time, x)
    """
    # get the size of training trajectory and generated trajectory
    num_points = data_train[0].shape[1]
    # set the independent variable: time
    time_axis = range(num_points)  # from 0 to the last point
    '''
    plot training data in x-axis (time, x)
    '''
    plt.figure()
    for i in range(param.nb_Samples):
        if i == 0:
            plt.plot(time_axis, data_train[i][0, :], 'gray', label='training data')
        else:
            plt.plot(time_axis, data_train[i][0, :], 'gray')

    '''
    Plot the start and end position of training trajectory
    '''
    # # plot the start Position
    # plt.scatter(time_axis[0], data_train[0][0, 0],
    #             marker='X', color='blue', s=60, label='start position')
    # # Plot the end Position
    # plt.scatter(time_axis[-1], data_train[0][0, num_points - 1],
    #             marker='X', color='green', s=60, label='end position')

    '''
    plot keypoints
    '''
    # get the number of the keypoints_list
    num_keypoints = len(keypoints_list)
    # remove the first and last keypoint, because this two points are start and end points
    # keypoints_list = keypoints_list[1: num_keypoints-1]

    for i in range(num_keypoints):
        if i == 0:
            plt.scatter(keypoints_id_list[i], keypoints_list[i][0, 0], marker='o', color='orange', label='keypoints')
        else:
            plt.scatter(keypoints_id_list[i], keypoints_list[i][0, 0], marker='o', color='orange')

    # plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.tick_params(labelsize=14)  # Label Size
    plt.xlabel('$time/second$', fontsize=15)
    plt.ylabel('$X/mm$', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    # plt.axis('equal')


def plot_traj_z(param, data_train, keypoints_list, keypoints_id_list):
    """
    plot the trajectory(motion) in z-axis (time, z)
    """
    # get the size of training trajectory and generated trajectory
    num_points = data_train[0].shape[1]
    # set the independent variable: time
    time_axis = range(num_points)  # from 0 to the last point
    '''
    plot training data in x-axis (time, x)
    '''
    plt.figure()
    for i in range(param.nb_Samples):
        if i == 0:
            plt.plot(time_axis, data_train[i][2, :], 'gray', label='training data')
        else:
            plt.plot(time_axis, data_train[i][2, :], 'gray')

    '''
    Plot the start and end position of training trajectory
    '''
    # # plot the start Position
    # plt.scatter(time_axis[0], data_train[0][2, 0],
    #             marker='X', color='blue', s=60, label='start position')
    # # Plot the end Position
    # plt.scatter(time_axis[-1], data_train[0][2, num_points - 1],
    #             marker='X', color='green', s=60, label='end position')

    '''
    plot keypoints
    '''
    # get the number of the keypoints_list
    num_keypoints = len(keypoints_list)
    # remove the first and last keypoint, because this two points are start and end points
    # keypoints_list = keypoints_list[1: num_keypoints-1]

    for i in range(num_keypoints):
        if i == 0:
            plt.scatter(keypoints_id_list[i], keypoints_list[i][0, 2], marker='o', color='orange', label='keypoints')
        else:
            plt.scatter(keypoints_id_list[i], keypoints_list[i][0, 2], marker='o', color='orange')

    # plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.tick_params(labelsize=14)  # Label Size
    plt.xlabel('$time/second$', fontsize=15)
    plt.ylabel('$Z/mm$', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    # plt.axis('equal')


def plot_keypoint_2d(param, data_train, keypoints_list):
    """
    plot the trajectory in 2D (x,y plane)
    """

    # get the size of training trajectory and generated trajectory
    num_points = data_train[0].shape[1]
    '''
    plot training data
    '''
    plt.figure()
    for i in range(param.nb_Samples):
        if i == 0:
            plt.plot(data_train[i][0, :], data_train[i][1, :], 'gray', label='training data')
        else:
            plt.plot(data_train[i][0, :], data_train[i][1, :], 'gray')

    '''
    Plot the start and end position of training trajectory
    '''
    # plot the start Position
    plt.scatter(data_train[0][0, 0], data_train[0][1, 0],
                marker='X', color='blue', s=60, label='start position')
    # Plot the end Position
    plt.scatter(data_train[0][0, num_points-1], data_train[0][1, num_points-1],
                marker='X', color='green', s=60, label='end position')

    '''
    plot keypoints
    '''
    # get the number of the keypoints_list
    num_keypoints = len(keypoints_list)
    # remove the first and last keypoint, because this two points are start and end points
    # keypoints_list = keypoints_list[1: num_keypoints-1]

    for keypoint in keypoints_list:
        if (keypoint == keypoints_list[0]).all():  # judge if all the element in array are same
            plt.scatter(keypoint[0, 0], keypoint[0, 1], marker='o', color='orange', label='keypoints')
        else:
            plt.scatter(keypoint[0, 0], keypoint[0, 1], marker='o', color='orange')

    plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.xlabel('$X/mm$')
    plt.ylabel('$Y/mm$')
    plt.legend(loc='best')
    plt.axis('equal')


def plot_keypoint_2d_yz(param, data_train, keypoints_list):
    """
    plot the trajectory in 2D (y-z plane)
    """
    # get the size of training trajectory and generated trajectory
    num_points = data_train[0].shape[1]

    '''
    plot training data
    '''
    plt.figure()
    for i in range(param.nb_Samples):
        if i == 0:
            plt.plot(data_train[i][1, :], data_train[i][2, :], 'gray', label='training data')
        else:
            plt.plot(data_train[i][1, :], data_train[i][2, :], 'gray')

    '''
    Plot the start and end position of training trajectory
    '''
    # plot the start Position
    plt.scatter(data_train[0][1, 0], data_train[0][2, 0],
                marker='X', color='blue', s=60, label='start position')
    # Plot the end Position
    plt.scatter(data_train[0][1, num_points-1], data_train[0][2, num_points-1],
                marker='X', color='green', s=60, label='end position')

    '''
    plot keypoints
    '''
    # get the number of the keypoints_list
    num_keypoints = len(keypoints_list)
    # remove the first and last keypoint, because this two points are start and end points
    # keypoints_list = keypoints_list[1: num_keypoints-1]

    for keypoint in keypoints_list:
        if (keypoint == keypoints_list[0]).all():  # judge if all the element in array are same
            plt.scatter(keypoint[0, 1], keypoint[0, 2], marker='o', color='orange', label='keypoints')
        else:
            plt.scatter(keypoint[0, 1], keypoint[0, 2], marker='o', color='orange')

    plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.xlabel('$Y/mm$')
    plt.ylabel('$Z/mm$')
    plt.legend(loc='best')
    # plt.axis('equal')


def plot_keypoint_3d(param, data_train, keypoints_list):
    """
    plot the trajectories in 3D
    """
    # get the size of training trajectory and generated trajectory
    num_points = data_train[0].shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    '''
    plot training data in 3D
    '''
    for i in range(param.nb_Samples):
        if i == 0:
            ax.plot(data_train[i][0, :], data_train[i][1, :], data_train[i][2, :], 'gray', label='training data')
        else:
            ax.plot(data_train[i][0, :], data_train[i][1, :], data_train[i][2, :], 'gray')

    '''
    plot start position and end position of training data
    '''
    ax.scatter(data_train[0][0, 0], data_train[0][1, 0], data_train[0][2, 0], marker='o', color='blue', linewidth=3.5)
    ax.scatter(data_train[0][0, num_points-1],
               data_train[0][1, num_points-1],
               data_train[0][2, num_points-1], marker='o', color='green', linewidth=3.5)

    '''
    plot keypoints of training trajectory in 3D
    '''
    # get the number of the keypoints_list
    num_keypoints = len(keypoints_list)
    # remove the first and last keypoint, because this two points are start and end points
    # keypoints_list = keypoints_list[1: num_keypoints-1]

    for keypoint in keypoints_list:
        ax.scatter(keypoint[0, 0], keypoint[0, 1], keypoint[0, 2], marker='o', color='orange', linewidth=3.5)

    # plt.title('Trajectory in 3D', fontsize='medium', fontweight='bold')
    plt.tick_params(labelsize=13)  # Label Size
    # ax.set_xlabel('x(mm)', fontsize=15)
    # ax.set_ylabel('y(mm)', fontsize=15)
    # ax.set_zlabel('z(mm)', fontsize=15)
    # Set the axis coordinate scale interval
    # x_major_locator = MultipleLocator(150)
    # ax.xaxis.set_major_locator(x_major_locator)
    #
    # y_major_locator = MultipleLocator(3)
    # ax.yaxis.set_major_locator(y_major_locator)
    #
    # z_major_locator = MultipleLocator(0.5)
    # ax.zaxis.set_major_locator(z_major_locator)


def plot2d_xy(param, data_train, trajectory_gen, position_Obstacle, plot_obstacle, keypoints_list):
    """
    plot the trajectory in 2D (x,y plane)
    """
    # get the size of training trajectory and generated trajectory
    num_points = data_train[0].shape[1]
    size_traj_gen = trajectory_gen.shape[1]

    '''
    plot training data
    '''
    plt.figure()
    for i in range(param.nb_Samples):
        if i == 0:
            plt.plot(data_train[i][0, :], data_train[i][1, :], 'gray', label='training data')
        else:
            plt.plot(data_train[i][0, :], data_train[i][1, :], 'gray')

    # plot generate trajectory in x-y plane
    plt.plot(trajectory_gen[0, :], trajectory_gen[1, :], 'red', label='generate trajectory')

    # plot start position and end position of training data
    # plt.scatter(data_train[0][0, 0], data_train[0][1, 0], marker='o', color='blue', label='start position')
    # plt.scatter(data_train[0][0, param.nb_Data-1], data_train[0][1, param.nb_Data-1],
    #             marker='o', color='green', label='end position')

    '''
    Plot the start and end position of training trajectory
    '''
    # plot the start Position
    plt.scatter(data_train[0][0, 0], data_train[0][1, 0],
                marker='X', color='blue', s=60, label='start position')
    # Plot the end Position
    plt.scatter(data_train[0][0, num_points-1], data_train[0][1, num_points-1],
                marker='X', color='green', s=60, label='end position')

    '''
    plot start position and end position of generated trajectory
    '''
    # plt.scatter(trajectory_gen[0, 0], trajectory_gen[1, 0], marker='o', color='blue')
    # plt.scatter(trajectory_gen[0, size_traj_gen-1], trajectory_gen[1, size_traj_gen-1], marker='o', color='green')

    '''
    plot keypoints
    '''
    # get the number of the keypoints_list
    num_keypoints = len(keypoints_list)
    # remove the first and last keypoint, because this two points are start and end points
    keypoints_list = keypoints_list[1: num_keypoints-1]

    for keypoint in keypoints_list:
        if (keypoint == keypoints_list[0]).all():  # judge if all the element in array are same
            plt.scatter(keypoint[0, 0], keypoint[0, 1], marker='o', color='orange', label='keypoints')
        else:
            plt.scatter(keypoint[0, 0], keypoint[0, 1], marker='o', color='orange')

    '''
    plot position of obstacle
    '''
    if plot_obstacle:
        plt.scatter(position_Obstacle[0, 0], position_Obstacle[1, 0], marker='o', color='magenta')

    # plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.xlabel('$X/mm$', fontsize=13)
    plt.ylabel('$Y/mm$', fontsize=13)
    plt.legend(loc='best')
    plt.axis('equal')


def plot2d_xz(param, s, trajectory_gen):
    # plot x_z plane
    size_traj_gen = trajectory_gen.shape[1]
    plt.figure()
    for i in range(param.nb_Samples):
        if i == 0:
            plt.plot(s[i][0], s[i][2], 'gray', label='training data')
        else:
            plt.plot(s[i][0], s[i][2], 'gray')
    # plot generate trajectory in x-y plane
    plt.plot(trajectory_gen[0, :], trajectory_gen[2, :], 'red', label='generate trajectory')
    # plot start position and end position of training data
    plt.scatter(s[0][0][0], s[0][2][0], marker='o', color='blue', label='start position')
    plt.scatter(s[0][0][param.nb_Data-1], s[0][2][param.nb_Data-1], marker='o', color='green', label='end position')
    # plot start position and end position of generated trajectory
    plt.scatter(trajectory_gen[0, 0], trajectory_gen[2, 0], marker='o', color='blue')
    plt.scatter(trajectory_gen[0, size_traj_gen-1], trajectory_gen[2, size_traj_gen-1], marker='o', color='green')

    plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.xlabel('$X/mm$')
    plt.ylabel('$Z/mm$')
    plt.legend(loc='best')


def plot_3d(param, data_train, trajectory_gen, position_Obstacle, plot_obstacle, keypoints_list):
    """
    plot the trajectories in 3D
    """
    # get the size of training trajectory and generated trajectory
    num_points = data_train[0].shape[1]
    size_traj_gen = trajectory_gen.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    '''
    plot training data in 3D
    '''
    for i in range(param.nb_Samples):
        ax.plot(data_train[i][0, :], data_train[i][1, :], data_train[i][2, :], 'gray',  linewidth=1.8)

    '''
    plot generated trajectory in 3D
    '''
    ax.plot(trajectory_gen[0, :], trajectory_gen[1, :], trajectory_gen[2, :], 'red', linewidth=1.8)

    '''
    plot start position and end position of training data
    '''
    ax.scatter(data_train[0][0, 0], data_train[0][1, 0], data_train[0][2, 0], marker='o', color='blue', linewidth=3.5)
    ax.scatter(data_train[0][0, num_points-1],
               data_train[0][1, num_points-1],
               data_train[0][2, num_points-1], marker='o', color='green', linewidth=3.5)

    '''
    plot start position and end position of generated trajectory
    '''
    # ax.scatter(trajectory_gen[0, 0],
    #            trajectory_gen[1, 0],
    #            trajectory_gen[2, 0], marker='o', color='blue', label='start position')  # start position
    # ax.scatter(trajectory_gen[0, size_traj_gen-1],
    #            trajectory_gen[1, size_traj_gen-1],
    #            trajectory_gen[2, size_traj_gen-1], marker='o', color='green', label='start position')  # end position

    '''
    plot keypoints of training trajectory in 3D
    '''
    # get the number of the keypoints_list
    num_keypoints = len(keypoints_list)
    # remove the first and last keypoint, because this two points are start and end points
    keypoints_list = keypoints_list[1: num_keypoints-1]

    for keypoint in keypoints_list:
        ax.scatter(keypoint[0, 0], keypoint[0, 1], keypoint[0, 2], marker='o', color='orange', linewidth=3)

    '''
    plot position of obstacle:
    '''
    # if plot_obstacle:
    #     ax.scatter(position_Obstacle[0, 0],
    #                position_Obstacle[1, 0],
    #                position_Obstacle[2, 0], marker='o', color='magenta', label='start position')  # start position

    # plt.title('Trajectory in 3D', fontsize='medium', fontweight='bold')
    plt.tick_params(labelsize=15)  # Label Size
    ax.set_xlabel('x(mm)', fontsize=15)  # Font Size
    ax.set_ylabel('y(mm)', fontsize=15)
    ax.set_zlabel('z(mm)', fontsize=15)

    # Set the axis coordinate scale interval
    # x_major_locator = MultipleLocator(150)
    # ax.xaxis.set_major_locator(x_major_locator)
    #
    # y_major_locator = MultipleLocator(3)
    # ax.yaxis.set_major_locator(y_major_locator)
    # #
    # z_major_locator = MultipleLocator(0.5)
    # ax.zaxis.set_major_locator(z_major_locator)


def plot_force(param, s, trajectory_gen, sIn, Force_NL, currF, H, position_Obstacle, plot_obstacle):
    """
    plot the force term
    """
    '''
    force in X-axis
    '''
    plt.figure()
    # show image from negative x-axis
    ax = plt.gca()
    ax.invert_xaxis()
    for i in range(param.nb_Samples):
        if i == 0:
            plt.plot(sIn, Force_NL[i][0], 'gray', label='force_x')
        else:
            plt.plot(sIn, Force_NL[i][0], 'gray')
    plt.plot(sIn, currF[0, :], 'red', label='force for new trajectory')
    plt.title('External Force in X-axis', fontsize='medium', fontweight='bold')
    plt.xlabel('$phase\ variable$')
    plt.ylabel('$force/N$')
    plt.legend(loc='best')

    '''
    force in Y-axis
    '''
    plt.figure()
    # show image from negative x-axis
    ax = plt.gca()
    ax.invert_xaxis()
    for i in range(param.nb_Samples):
        if i == 0:
            plt.plot(sIn, Force_NL[i][1], 'gray', label='force_y')
        else:
            plt.plot(sIn, Force_NL[i][1], 'gray')
    plt.plot(sIn, currF[1, :], 'red', label='force for new trajectory')
    plt.title('External Force in Y-axis', fontsize='medium', fontweight='bold')
    plt.xlabel('$phase\ variable$')
    plt.ylabel('$force/N$')
    plt.legend(loc='best')

    '''
    force in Z-axis
    '''
    plt.figure()
    # show image from negative x-axis
    ax = plt.gca()
    ax.invert_xaxis()
    for i in range(param.nb_Samples):
        if i == 0:
            plt.plot(sIn, Force_NL[i][2], 'gray', label='force_z')
        else:
            plt.plot(sIn, Force_NL[i][2], 'gray')
    plt.plot(sIn, currF[2, :], 'red', label='force for new trajectory')
    plt.title('External Force in Z-axis', fontsize='medium', fontweight='bold')
    plt.xlabel('$phase\ variable$')
    plt.ylabel('$force/N$')
    plt.legend(loc='best')


def plot_GMM(param, sIn, H):
    """
    plot the GMM activation function
    """
    plt.figure()
    # show image from negative x-axis
    ax = plt.gca()
    ax.invert_xaxis()
    for i in range(param.nbStates):
        plt.plot(sIn, H[i, :])
    plt.title('Activation function of GMM', fontsize='medium', fontweight='bold')
    plt.xlabel('$phase\ variable$')
    plt.ylabel('$Activation\ function$')
    # plt.legend(loc='best')

def plot_main(param, data_train, trajectory_gen, sIn, Force_NL, currF, H, position_Obstacle, plot_obstacle,
              keypoints_list, keypoints_list_x, keypoints_list_y, keypoints_list_z, keypoints_ID_list):
    """
    plot the results
    """
    '''
    plot the training trajectory with keypoints in different axis (x-axis, y-axis, z-axis)
    '''
    # plot the training trajectory in x axis (time, x)
    # plot_traj_x(param, data_train, keypoints_list, keypoints_ID_list)
    # plot the training trajectory in z axis (time, z)
    # plot_traj_z(param, data_train, keypoints_list, keypoints_ID_list)

    '''
    plot the training trajectory with keypoints in different plane: x-y, y-x, ...
    '''
    # plot trajectory with keypoints in 2D
    # plot_keypoint_2d(param, data_train, keypoints_list)  # in x-y plane
    # plot_keypoint_2d_yz(param, data_train, keypoints_list)  # in y-z plane

    # plot trajectory with keypoints in 3D
    plot_keypoint_3d(param, data_train, keypoints_list)

    '''
    plot the trajectory in 2D (x,y plane)
    '''
    # Plot trajectory in x-y plane:
    plot2d_xy(param, data_train, trajectory_gen, position_Obstacle, plot_obstacle, keypoints_list)

    # Plot trajectory in x-z plane:
    # plot2d_xz()

    '''
    plot the trajectory in 3D
    '''
    plot_3d(param, data_train, trajectory_gen, position_Obstacle, plot_obstacle, keypoints_list)

    '''
    plot the Force Term
    '''
    # plot_force()

    '''
    plot the GMM function
    '''
    # plot_GMM()

    plt.show()




