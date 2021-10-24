import numpy as np
from scipy.interpolate import interp1d

def resample(data, num_data, num_resample):
    """
    Interpolate and Resample
    :param data:
    :param num_data: number of the data
    :param num_resample: number used for resampling
    :return: resampled data
    """
    x_train = np.arange(num_data)  # independent variable of interpolation function
    # construct the interp
    f_interpolate = interp1d(x_train, data, kind='cubic')  # f(x) is data_train

    # the part of the trajectory that needed (here is the total trajectory)
    x_interpolate = np.linspace(x_train.min(), x_train.max(), num_resample)

    # resampling
    data_resampled = f_interpolate(x_interpolate)  # position(x,y,z): use interpolation function

    return data_resampled


def deviation_calculator(traj_train, traj_gen):
    """
    Calculate the deviation between the training trajectory and generated trajectory for evaluate the accuracy of
    the learned motion model
    :param traj_train: the training trajectory
    :param traj_gen: the generated trajectory
    :return: the deviation
    """
    '''
    Resample the two trajectories
    '''
    # get the number of the two trajectories and set parameters:
    num_train = traj_train[0].shape[1]  # number of training trajectory
    num_gen = traj_gen.shape[1]  # number of generated trajectory
    num_resample = 200

    # Resample
    traj_train_resampled = resample(traj_train[0], num_train, num_resample)
    traj_gen_resampled = resample(traj_gen, num_gen, num_resample)

    # for test
    # print(np.linalg.norm(traj_gen_resampled[:, 0] - traj_train_resampled[:, 0]))
    # print(traj_train_resampled[:, 0])

    '''
    Calculate the Deviation
    '''
    deviation_total = 0  # the sum of the absolute value of the deviation between two correspond point
    for i in range(num_resample):
        deviation_total = deviation_total + np.linalg.norm(traj_gen_resampled[:, i] - traj_train_resampled[:, i])
    deviation = deviation_total / num_resample

    return deviation

# if __name__ == '__main__':
