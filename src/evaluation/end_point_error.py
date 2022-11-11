import numpy as np

from src.evaluation.variables import UNKNOWN_FLOW_THRESH


def read_flo_file(filename, verbose=False):
    """
    Read from .flo optical flow file (Middlebury format)
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    adapted from https://github.com/liruoteng/OpticalFlowToolkit/
    """
    f = open(filename, "rb")
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        raise TypeError("Magic number incorrect. Invalid .flo file")
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        if verbose:
            print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()

    return data2d


def flow_error(tu, tv, u, v):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :return: End point error of the estimated flow
    """
    smallflow = 0.0

    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]

    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = [(np.absolute(stu) > smallflow) | (np.absolute(stv) > smallflow)]

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    epe = epe[tuple(ind2)]
    sdEpe = round(np.std(epe), 2)
    mepe = round(np.mean(epe), 2)

    return mepe, sdEpe


def calc_MEPE_directory(folder_name: str, u: np.ndarray, v: np.ndarray):
    flo_in_path = "ground_truth_flow/" + folder_name + "/flow10.flo"
    # Read flo file
    flow = read_flo_file(flo_in_path)
    # Extract the true u
    tu = flow[:, :, 0]
    # Extract the true v
    tv = flow[:, :, 1]

    # calulate Average End Point Error
    mepe, sdEpe = flow_error(tu, tv, u, v)
    # print("The Average End Point Error for "+folder_name+ " is: " + str(mepe)+" and the standard deviation is: "+str(sdEpe))

    return mepe, sdEpe
