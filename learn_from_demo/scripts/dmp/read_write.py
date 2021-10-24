import numpy as np

def read_traj_gen(name_file):
    """
    :@brief: Load generated trajectory from Text File (.txt)
    :param name_file: name of loaded text file
    :return: loaded trajectory
    """
    traj_gen = np.loadtxt(name_file, dtype=float, delimiter=';')
    print('Info: generated trajectory is loaded')
    return traj_gen