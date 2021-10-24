from matplotlib import pyplot as plt
import numpy as np
from scipy import stats  # for calculating gaussian distribution
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


def plot_result3d(param, s, trajectory_gen, sIn, Force_NL, currF, H, position_Obstacle,
                  plot_obstacle, force_mean, data_train_raw):
    '''
    Set Parameters for plotting
    '''
    xy_axis_fontsize = 16  # size of the axis label
    scale_size = 15  # size of the scale on axis
    legend_size = 14  # size of the legend

    """
    plot the trajectory in 2D (x,y plane)
    """
    # sIn = sIn[::-1]
    # plot training data
    plt.figure()
    size_traj_gen = trajectory_gen.shape[1]
    for i in range(param.nb_Samples):
        if i == 0:
            plt.plot(s[i][0], s[i][1], 'gray', label='training data')
        else:
            plt.plot(s[i][0], s[i][1], 'gray')

    # plot generate trajectory in x-y plane
    # plt.plot(trajectory_gen[0, :], trajectory_gen[1, :], 'red', label='generate \ntrajectory')
    # plot start position and end position of training data
    plt.scatter(s[0][0][0], s[0][1][0], marker='o', color='blue', label='start position')
    plt.scatter(s[0][0][param.nb_Data-1], s[0][1][param.nb_Data-1], marker='o', color='green', label='end position')
    # plot start position and end position of generated trajectory
    plt.scatter(trajectory_gen[0, 0], trajectory_gen[1, 0], marker='o', color='blue')
    plt.scatter(trajectory_gen[0, size_traj_gen-1], trajectory_gen[1, size_traj_gen-1], marker='o', color='green')
    # plot position of obstacle
    if plot_obstacle:
        plt.scatter(position_Obstacle[0, 0], position_Obstacle[1, 0], marker='o', color='magenta')

    # plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.tick_params(labelsize=14)  # Label Size
    plt.xlabel('$X/mm$', fontsize=15)
    plt.ylabel('$Y/mm$', fontsize=15)

    # Set the range of the axis
    # plt.xlim(-10, 18)
    # plt.ylim(-16, 16)

    # plt.legend(loc='lower right', fontsize=12)
    plt.legend(loc='best', fontsize=15)
    plt.axis('equal')

    # plot x_z plane
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

    # plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.xlabel('$X/mm$')
    plt.ylabel('$Z/mm$')
    plt.legend(loc='best')

    """
    plot the trajectories in 3D
    """
    z = np.ones(param.nb_Data)*3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot training data in 3D (Filtered)
    # for i in range(param.nb_Samples):
    #     if i == 0:
    #         ax.plot(s[i][0], s[i][1], s[i][2], 'orange', linewidth=2)
    #     else:
    #         ax.plot(s[i][0], s[i][1], s[i][2], 'orange', linewidth=2)

    # plot training data in 3D (Raw data not use kalman filter)
    for i in range(param.nb_Samples):
        ax.plot(data_train_raw[i][0], data_train_raw[i][1], data_train_raw[i][2], 'gray',  linewidth=2)

    # plot generated trajectory in 3D
    ax.plot(trajectory_gen[0, :], trajectory_gen[1, :], trajectory_gen[2, :], 'red', linewidth=2)

    # plot start position and end position of training data
    ax.scatter(s[0][0][0], s[0][1][0], s[0][2][0], marker='o', color='blue', linewidth=3.5)
    ax.scatter(s[0][0][param.nb_Data-1],
               s[0][1][param.nb_Data-1],
               s[0][2][param.nb_Data-1], marker='o', color='green', linewidth=3.5)

    # plot start position and end position of generated trajectory
    ax.scatter(trajectory_gen[0, 0],
               trajectory_gen[1, 0],
               trajectory_gen[2, 0], marker='o', color='blue', linewidth=3.5)  # start position
    ax.scatter(trajectory_gen[0, size_traj_gen-1],
               trajectory_gen[1, size_traj_gen-1],
               trajectory_gen[2, size_traj_gen-1], marker='o', color='green', linewidth=3.5)  # end position

    # plot position of obstacle:
    if plot_obstacle:
        ax.scatter(position_Obstacle[0, 0],
                   position_Obstacle[1, 0],
                   position_Obstacle[2, 0], marker='o', color='magenta', label='start position')  # start position

    # plt.title('Trajectory in 3D', fontsize='medium', fontweight='bold')
    plt.tick_params(labelsize=12)  # Label Size
    ax.set_xlabel('x(mm)', fontsize=14)  # 15
    ax.set_ylabel('y(mm)', fontsize=14)
    ax.set_zlabel('z(mm)', fontsize=14)

    # Set the axis coordinate scale interval
    # x_major_locator = MultipleLocator(2)
    # ax.xaxis.set_major_locator(x_major_locator)
    # #
    # y_major_locator = MultipleLocator(3)
    # ax.yaxis.set_major_locator(y_major_locator)
    # #
    # z_major_locator = MultipleLocator(0.5)
    # ax.zaxis.set_major_locator(z_major_locator)

    '''
    plot the force term
    '''
    # get the id of Gaussian basis functions
    id_gbf_max = np.argmax(H, axis=0)  # the id of gaussian basis function, which is the maximum weight at each time step

    # force in X-axis
    plt.figure()
    # show image from negative x-axis
    ax = plt.gca()
    ax.invert_xaxis()
    for i in range(param.nb_Samples):
        if i == 0:
            # plot the target force (calculated from training trajectory)
            plt.plot(sIn, Force_NL[i][0], 'gray', label='target force')
            # plot the mean force of each gaussian basis function
            for k in range(param.nbStates):
                id_weight_max = np.argwhere(id_gbf_max == k)  # ID of the maximum weight at each time step
                f_mean = np.ones(id_weight_max.shape[0]-1)*force_mean[0][k]
                plt.plot(sIn[id_weight_max[0][0]:id_weight_max[-1][0]], f_mean)
        else:
            plt.plot(sIn, Force_NL[i][0], 'gray')
    # plot calculated force by using DMP
    plt.plot(sIn, currF[0, :], 'red', label='calculated force by DMP')

    # plt.title('External Force in X-axis', fontsize='medium', fontweight='bold')
    plt.xlabel('$phase\ variable$')
    plt.ylabel('$force\ /\ N$')
    plt.legend(loc='best')

    # force in Y-axis
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

    # force in Z-axis
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

    '''
    plot the GMM activation function
    '''
    plt.figure()
    # show image from negative x-axis
    ax = plt.gca()
    ax.invert_xaxis()
    for i in range(param.nbStates):
        plt.plot(sIn, H[i, :])
    # plt.title('Activation function of GMM', fontsize='medium', fontweight='bold')
    plt.xlabel('$phase\ variable$')
    plt.ylabel('$Gaussian \ basis \ functions$')
    # plt.legend(loc='best')

    '''
    Plot the phase variable
    '''
    plt.figure()
    time_axis = np.arange(param.nb_Data)*0.01
    plt.plot(time_axis, sIn)
    plt.xlabel('$time\ (seconds)$', fontsize=12)
    plt.ylabel('$phase\ variable$', fontsize=12)

    '''
    Plot the gaussian basis functions with Force Term (Subplot)
    '''
    plt.figure()

    # Sub image 1
    plt.subplot(2, 1, 1)  # (number of row for sub images, number of column for sub images, ID of current Image)
    # show image from negative x-axis
    ax = plt.gca()
    ax.invert_xaxis()

    for i in range(param.nb_Samples):
        if i == 0:
            # plot the target force (calculated from training trajectory)
            plt.plot(sIn, Force_NL[i][0], 'gray', label='target force')
            # plot the mean force of each gaussian basis function
            for k in range(param.nbStates):
                id_weight_max = np.argwhere(id_gbf_max == k)  # ID of the maximum weight at each time step
                f_mean = np.ones(id_weight_max.shape[0]-1)*force_mean[0][k]
                plt.plot(sIn[id_weight_max[0][0]:id_weight_max[-1][0]], f_mean)
            # For just show one segment of the function
            # k = 4
            # id_weight_max = np.argwhere(id_gbf_max == k)  # ID of the maximum weight at each time step
            # f_mean = np.ones(id_weight_max.shape[0]-1)*force_mean[0][k]
            # plt.plot(sIn[id_weight_max[0][0]:id_weight_max[-1][0]], f_mean)

        else:
            plt.plot(sIn, Force_NL[i][0], 'gray')

    # plot calculated force by using DMP
    plt.plot(sIn, currF[0, :], 'red', label='calculated force by DMP')

    # Title of Axis:
    # plt.title('External Force in X-axis', fontsize='medium', fontweight='bold')

    # Set the Range of Axis:
    plt.xlim(1, 0)

    # Set the Label of Axis:
    plt.tick_params(labelsize=scale_size)  # Label Size of Axis
    # plt.xlabel('$phase\ variable$', fontsize=16)
    plt.ylabel('$Force\ /\ N$', fontsize=xy_axis_fontsize)

    plt.legend(loc='best', fontsize=legend_size)

    # Sub image 2
    plt.subplot(2, 1, 2)  # (number of row for sub images, number of column for sub images, ID of current Image)
    # show image from negative x-axis
    ax = plt.gca()
    ax.invert_xaxis()
    for i in range(param.nbStates):
        plt.plot(sIn, H[i, :])
    # plt.plot(sIn, H[4, :])  # for just show one GBF

    # Title of Axis:
    # plt.title('Activation function of GMM', fontsize='medium', fontweight='bold')

    # Set the Range of Axis:
    plt.xlim(1, 0)

    # Set the Label of Axis:
    plt.tick_params(labelsize=scale_size)  # Label Size
    plt.xlabel('$phase\ variable$', fontsize=xy_axis_fontsize)
    plt.ylabel('$Gaussian \ basis \ functions$', fontsize=xy_axis_fontsize)
    # plt.legend(loc='best')

    '''
    Plot a gaussian basis function
    '''
    plt.figure()
    # show image from negative x-axis
    ax = plt.gca()
    ax.invert_xaxis()
    plt.xlim((1, 0))
    my_x_ticks = np.arange(1, 0, -0.1)
    plt.xticks(my_x_ticks)

    # Set the Gaussian Basis Function
    mean_value = 0.5
    variance = 0.02
    standard_deviation = np.sqrt(variance)
    GBF1 = stats.norm.pdf(sIn, mean_value, standard_deviation)  # gaussian basis function
    GBF2 = stats.norm.pdf(sIn, 0.55, np.sqrt(0.03))  # gaussian basis function
    # GBF = np.random.normal(sIn, mean_value, standard_deviation)
    plt.plot(sIn, GBF1, 'blue', label='gaussian basis function 1')
    plt.plot(sIn, GBF2, 'red', label='gaussian basis function 2 ')

    plt.tick_params(labelsize=15)  # Label Size
    plt.xlabel('$phase\ variable$', fontsize=16)
    plt.ylabel('$Gaussian \ basis \ functions$', fontsize=16)
    # plt.legend(loc='best')

    '''
    Gaussian distribution
    '''
    # plt.figure()
    # plt.xlim((-2, 2))
    # # Set the Gaussian Basis Function
    # mean_value = 0
    # variance = 0.02
    # standard_deviation = np.sqrt(variance)
    # x_ = np.linspace(-2, 2, 200)
    # GBF = stats.norm.pdf(x_, mean_value, standard_deviation)  # gaussian basis function
    # plt.plot(x_, GBF, 'blue', label='gaussian basis function')
    #
    # plt.xlabel('$phase\ variable$', fontsize=13)
    # plt.ylabel('$Gaussian \ basis \ functions$', fontsize=13)

    '''
    Plot the position, velocity and acceleration 
    '''
    plt.figure()
    time_axis = np.arange(param.nb_Data)/100

    # Position in x-axis
    plt.subplot(4, 1, 1)
    plt.plot(time_axis, s[0][0], 'red', label='position in x-axis')
    plt.tick_params(labelsize=15)  # Label Size
    # plt.xlabel('$time\ seconds$', fontsize=13)
    plt.ylabel('$position \ (mm)$', fontsize=16)

    # velocity in x-axis
    plt.subplot(4, 1, 2)
    plt.plot(time_axis, s[0][3], 'green', label='velocity in x-axis')
    plt.tick_params(labelsize=15)  # Label Size
    # plt.xlabel('$time\ seconds$', fontsize=13)
    plt.ylabel('$velocity \ (mm/s)$', fontsize=16)

    # acceleration in x-axis
    plt.subplot(4, 1, 3)
    plt.plot(time_axis, s[0][6], 'blue', label='acceleration in x-axis')
    plt.tick_params(labelsize=15)  # Label Size
    # plt.xlabel('$time\ (seconds)$', fontsize=13)
    plt.ylabel('$acceleration \ (mm/s^2)$', fontsize=16)

    # target force in x-axis
    plt.subplot(4, 1, 4)
    for i in range(param.nb_Samples):
        if i == 0:
            # plot the target force (calculated from training trajectory)
            plt.plot(time_axis, Force_NL[i][0], 'orange', label='target force in x-axis')
        else:
            plt.plot(time_axis, Force_NL[i][0], 'orange')

    # plt.title('External Force in X-axis', fontsize='medium', fontweight='bold')
    plt.tick_params(labelsize=15)  # Label Size
    plt.xlabel('$time\ (seconds)$', fontsize=16)
    plt.ylabel('$target \ force\ (N)$', fontsize=16)
    # plt.legend(loc='best')




    plt.show()
