from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

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


def plot_result3d(param, s, trajectory_gen, sIn, Force_NL, currF, H, position_Obstacle, plot_obstacle):
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
    plt.plot(trajectory_gen[0, :], trajectory_gen[1, :], 'red', label='generate trajectory')
    # plot start position and end position of training data
    plt.scatter(s[0][0][0], s[0][1][0], marker='o', color='blue', label='start position')
    plt.scatter(s[0][0][param.nb_Data-1], s[0][1][param.nb_Data-1], marker='o', color='green', label='end position')
    # plot start position and end position of generated trajectory
    plt.scatter(trajectory_gen[0, 0], trajectory_gen[1, 0], marker='o', color='blue')
    plt.scatter(trajectory_gen[0, size_traj_gen-1], trajectory_gen[1, size_traj_gen-1], marker='o', color='green')
    # plot position of obstacle
    if plot_obstacle:
        plt.scatter(position_Obstacle[0, 0], position_Obstacle[1, 0], marker='o', color='magenta')

    plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.xlabel('$X/mm$')
    plt.ylabel('$Y/mm$')
    plt.legend(loc='best')

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

    plt.title('Trajectory', fontsize='medium', fontweight='bold')
    plt.xlabel('$X/mm$')
    plt.ylabel('$Z/mm$')
    plt.legend(loc='best')

    """
    plot the trajectories in 3D
    """
    z = np.ones(param.nb_Data)*3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot training data in 3D
    for i in range(param.nb_Samples):
        if i == 0:
            ax.plot(s[i][0], s[i][1], s[i][2], 'gray', label='training data')
        else:
            ax.plot(s[i][0], s[i][1], s[i][2], 'gray')
    # plot generated trajectory in 3D
    ax.plot(trajectory_gen[0, :], trajectory_gen[1, :], trajectory_gen[2, :], 'red', label='generate trajectory')

    # plot start position and end position of training data
    ax.scatter(s[0][0][0], s[0][1][0], s[0][2][0], marker='o', color='blue', label='start position')
    ax.scatter(s[0][0][param.nb_Data-1],
               s[0][1][param.nb_Data-1],
               s[0][2][param.nb_Data-1], marker='o', color='green', label='start position')

    # plot start position and end position of generated trajectory
    ax.scatter(trajectory_gen[0, 0],
               trajectory_gen[1, 0],
               trajectory_gen[2, 0], marker='o', color='blue', label='start position')  # start position
    ax.scatter(trajectory_gen[0, size_traj_gen-1],
               trajectory_gen[1, size_traj_gen-1],
               trajectory_gen[2, size_traj_gen-1], marker='o', color='green', label='start position')  # end position

    # plot position of obstacle:
    if plot_obstacle:
        ax.scatter(position_Obstacle[0, 0],
                   position_Obstacle[1, 0],
                   position_Obstacle[2, 0], marker='o', color='magenta', label='start position')  # start position

    plt.title('Trajectory in 3D', fontsize='medium', fontweight='bold')
    ax.set_xlabel('x(mm)')
    ax.set_ylabel('y(mm)')
    ax.set_zlabel('z(mm)')

    '''
    plot the force term
    '''
    # force in X-axis
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
    plt.title('Activation function of GMM', fontsize='medium', fontweight='bold')
    plt.xlabel('$phase\ variable$')
    plt.ylabel('$Activation\ function$')
    # plt.legend(loc='best')

    plt.show()
