import numpy as np
from scipy.interpolate import interp1d
import numpy.matlib  # Matrix library
from scipy import stats  # for calculating gaussian distribution
import copy
from Obstacle_avoid import obstacleAvoidance

def trajectory_generator(param, sIn, Force_adjust, position, position_target,
                         position_Obstacle, Gamma, Beta, dot_phi_max, Obstacle_detect):
    """
    :@brief: Use the learned skill model to create a new Trajectory (starting from x_0 toward x_goal)
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
        else:
            # original DMP
            acceleration = PD_gain.dot(np.vstack((position_target - position, -velocity)))

            # modified DMP
            # acceleration = PD_gain.dot(np.vstack((position_target - position, -velocity))) \
            #                 - param.KP * (position_target - position_start) * sIn[num_plan]

        velocity = velocity + acceleration * param.dt
        position = position + velocity * param.dt

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


def trajectory_generator2(param, sIn, Force_adjust, position, position_target, threshold,
                          position_Obstacle, Gamma, Beta, dot_phi_max, Obstacle_detect):
    """
    :@brief: Use the learned skill model to create a new Trajectory (starting from x_0 toward x_goal)
    :param param:
    :param sIn:
    :param Force_adjust:
    :param position:
    :param position_target:
    :param threshold: threshold for judging if the end position is close enough to the goal position
    :param position_Obstacle:
    :param Gamma:
    :param Beta:
    :param dot_phi_max:
    :param Obstacle_detect:
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

    # the new start position for the next trajectory segment
    position_start_new = copy.deepcopy(position_target)

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
        if np.linalg.norm(position - position_target) <= threshold:
            position_start_new = position  # the new start position for the next trajectory segment
            break

        # If plan is at least minimum length, check if the end position is close enough to goal position
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

    return plan_gen, position_start_new


def trajectory_connector(trajectory_gen_list):
    """
    Connect trajectory segments
    :param trajectory_gen_list: the list of generated trajectory parts
    :return: connected generated trajectory
    """
    number_segments = len(trajectory_gen_list)
    trajectory_gen = trajectory_gen_list[0]
    for i in range(1, number_segments):
        trajectory_gen = np.hstack((trajectory_gen, trajectory_gen_list[i]))

    return trajectory_gen
