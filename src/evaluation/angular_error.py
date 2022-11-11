import numpy as np

from src.evaluation.variables import UNKNOWN_FLOW_THRESH
from src.evaluation.end_point_error import read_flo_file


def angle_flow_error(tu, tv, u, v):
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
    index_su = su[tuple(ind2)]
    index_sv = sv[tuple(ind2)]
    an = 1.0 / np.sqrt(index_su**2 + index_sv**2 + 1)
    un = index_su * an
    vn = index_sv * an

    index_stu = stu[tuple(ind2)]
    index_stv = stv[tuple(ind2)]
    tn = 1.0 / np.sqrt(index_stu**2 + index_stv**2 + 1)
    tun = index_stu * tn
    tvn = index_stv * tn

    angle = un * tun + vn * tvn + (an * tn)
    mang = np.mean(angle)

    sdAngular = round(np.std(angle), 2)
    AAE = round(mang, 2)

    return AAE, sdAngular


def calc_AAE_directory(folder_name: str, u: np.ndarray, v: np.ndarray):
    flo_in_path = "ground_truth_flow/" + folder_name + "/flow10.flo"
    # Read flo file
    flow = read_flo_file(flo_in_path)
    # Extract the true u
    tu = flow[:, :, 0]
    # Extract the true v
    tv = flow[:, :, 1]

    # calulate Average Angular error
    mang, sdAngular = angle_flow_error(tu, tv, u, v)
    # print("The Average Angular error for "+folder_name+ " is: " + str(mang)+" and the standard deviation is: "+str(sdAngular))

    return mang, sdAngular
