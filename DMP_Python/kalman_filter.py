import cv2
import numpy as np
from matplotlib import pyplot as plt
from Load_Save_Data import load_data

'''
Reference: https://docs.opencv.org/master/dd/d6a/classcv_1_1KalmanFilter.html
'''

def kalman_filter_0(data_train):
    """
    Model of Kalman is: x_t = x_t-1
    :param data_train: raw training data
    :return: filtered data
    """
    # create a kalman object
    kalman = cv2.KalmanFilter(3, 3)  # (Dimensionality of the state, Dimensionality of the measurement, Dimensionality of the control vector)
    '''
    Set the parameter matrix for kalman filter: 
    '''
    # state transition matrix(A)
    kalman.transitionMatrix = np.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]], np.float32)
    # process noise covariance matrix (Q)
    processNoiseCov = 0.0001
    kalman.processNoiseCov = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]], np.float32) * processNoiseCov
    # measurement matrix (H)
    kalman.measurementMatrix = np.array([[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]], np.float32)
    # measurement noise covariance matrix (R)
    measurementNoiseCov = 0.01  # If the value is small, the measured value is very accurate, and the predicted result is closer to the measured value
    kalman.measurementNoiseCov = np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]], np.float32) * measurementNoiseCov

    # Set the initial state (State 0)
    kalman.statePre = np.array([[data_train[0][0, 0]],
                                [data_train[0][1, 0]],
                                [data_train[0][2, 0]]], np.float32)  # predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)

    # Get the length of the training data
    size_data_train = data_train[0].shape[1]
    # data_corrected = []  # A list to save the corrected data (data after filtering)
    data_filtered = np.zeros((3, size_data_train), np.float32)

    '''
    Filter the Raw Data
    '''
    for i in range(size_data_train):
        # current measurement
        measurement_current = np.array([[data_train[0][0, i]],
                                        [data_train[0][1, i]],
                                        [data_train[0][2, i]]], np.float32)

        # Updates the predicted state from the measurement
        state_corrected = kalman.correct(measurement_current)
        kalman.predict()
        for j in range(3):  # dimension is 3 (x, y, z)
            data_filtered[j, i] = state_corrected[j]

    data_train_filtered = []
    data_train_filtered.append(data_filtered)

    return data_train_filtered


def kalman_filter(data_train):
    """
    Model of Kalman is: x_t = x_t-1 + v_t-1 * dt
    :param data_train: raw training data
    :return: filtered data
    """
    # create a kalman object
    kalman = cv2.KalmanFilter(6, 3)  # (Dimensionality of the state, Dimensionality of the measurement, Dimensionality of the control vector)
    '''
    Set the parameter matrix for kalman filter: 
    '''
    # Calculate the time step
    fps = 3.5  # frame rate of the ZED camera
    dt = np.float32(1/fps)
    v = dt
    # calculate the velocity at first time step
    v_x_0 = (data_train[0][0, 1] - data_train[0][0, 0]) / dt
    v_y_0 = (data_train[0][1, 1] - data_train[0][1, 0]) / dt
    v_z_0 = (data_train[0][2, 1] - data_train[0][2, 0]) / dt


    # state transition matrix(A)
    kalman.transitionMatrix = np.array([[1, 0, 0, v, 0, 0],
                                        [0, 1, 0, 0, v, 0],
                                        [0, 0, 1, 0, 0, v],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]], np.float32)
    # process noise covariance matrix (Q)
    processNoiseCov = 0.0001
    kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]], np.float32) * processNoiseCov
    # measurement matrix (H)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0]], np.float32)
    # measurement noise covariance matrix (R)
    measurementNoiseCov = 0.1  # If the value is small, the measured value is very accurate, and the predicted result is closer to the measured value
    kalman.measurementNoiseCov = np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]], np.float32) * measurementNoiseCov

    # Set the initial state (State 0)
    kalman.statePre = np.array([[data_train[0][0, 0]],
                                [data_train[0][1, 0]],
                                [data_train[0][2, 0]],
                                [v_x_0],
                                [v_y_0],
                                [v_z_0]], np.float32)  # predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)

    # Get the length of the training data
    size_data_train = data_train[0].shape[1]
    # data_corrected = []  # A list to save the corrected data (data after filtering)
    data_filtered = np.zeros((3, size_data_train), np.float32)

    '''
    Filter the Raw Data
    '''
    for i in range(size_data_train):
        # current measurement
        measurement_current = np.array([[data_train[0][0, i]],
                                        [data_train[0][1, i]],
                                        [data_train[0][2, i]]], np.float32)

        # Updates the predicted state from the measurement
        state_corrected = kalman.correct(measurement_current)
        kalman.predict()
        for j in range(3):  # dimension is 3 (x, y, z)
            data_filtered[j, i] = state_corrected[j]

    data_train_filtered = []
    data_train_filtered.append(data_filtered)

    return data_train_filtered


def plot_data_filtered(data_raw, data_filtered):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot raw training data in 3D
    ax.plot(data_raw[0][0, :], data_raw[0][1, :], data_raw[0][2, :], 'gray', label='training data')

    # plot filtered training data in 3D
    ax.plot(data_filtered[0][0, :], data_filtered[0][1, :], data_filtered[0][2, :], 'royalblue', label='generate trajectory')

    ax.set_xlabel('x(mm)', fontsize=13)
    ax.set_ylabel('y(mm)', fontsize=13)
    ax.set_zlabel('z(mm)', fontsize=13)

    plt.show()

if __name__ == '__main__':
    # load  training data:
    data_train = load_data(1)

    # Use kalman filter to preprocess the raw training data
    data_filtered = kalman_filter(data_train)

    # plot the raw training trajectory and preprocessed training trajectory
    plot_data_filtered(data_train, data_filtered)
    # print(np.eye(3, 3))
