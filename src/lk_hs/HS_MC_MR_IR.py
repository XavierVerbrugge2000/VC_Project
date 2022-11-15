import numpy as np
import cv2 as cv
from scipy import signal
import math

from src.evaluation.angular_error import calc_AAE_directory
from src.evaluation.end_point_error import calc_MEPE_directory
from src.representation.colorwhele import visualize_flow
from src.lk_hs.horn_shunck import hs


def gmask(x, y, s):  # the function for gaussian filter
    gmask = np.exp(-((x**2) + (y**2)) // 2 // s**2)
    return gmask


size = 2
# Utilize the standard deviation from the paper.
s = sigma = math.sqrt(2 / (4 * math.pi))
G = []  # Gaussian Kernel
for i in range(-size, size + 1):
    G.append(gmask(i, 0, s))  # equating y to 0 since we need a 1D matri


def DoLow(I):  # the function down samples the intput image by 2
    """Algorithm: For an input image of size rxc
    1) Apply x mask
    2) Delete Alternate Columns
    3) Apply y mask
    4) delete alternate rows
    The ouput image is of size (r//2)x(c//2)"""
    # ========= Applying X mask ====================

    Ix = []
    for i in range(len(I[:, 0])):
        Ix.extend([signal.convolve(I[i, :, 0], G, "same")])  # Ix*Gx = Ix
        Ix.extend([signal.convolve(I[i, :, 1], G, "same")])  # Ix*Gx = Ix
        Ix.extend([signal.convolve(I[i, :, 2], G, "same")])  # Ix*Gx = Ix
    Ix = np.array(np.matrix(Ix))

    # ========selecting alternate columns===========
    Ix = I
    Ix = Ix[:, ::2]
    # ========= Applying Y mask ====================

    Ixy = []
    for i in range(len(Ix[0, :])):
        Ixy.extend([signal.convolve(Ix[:, i, 0], G, "same")])  # Ix * Gy = Ixy
        Ixy.extend([signal.convolve(Ix[:, i, 1], G, "same")])  # Ix * Gy = Ixy
        Ixy.extend([signal.convolve(Ix[:, i, 2], G, "same")])  # Ix * Gy = Ixy
    Ixy = np.array(np.matrix(np.transpose(Ixy)))

    # ========selecting alternate rows===========
    Ixy = Ix
    Ixy = Ixy[::2, :]
    return Ixy  # Returning Ixy...


def DoHigh(I, G):  # the function up samples the intput image by 2
    """Algorithm: For an input image of size rxc
    1) Insert zero column on every alternate columns
    2) Apply x mask
    3) Insert zero row on every alternate rows
    4) Apply y mask
    The ouput image is of size (2r)x(2c)"""
    G = np.dot(
        2, G
    )  # Doubing the Guassian kernel since we later use alternate rows and columns
    # =========== Inserting alternate columns of zeros ========
    newI = np.zeros(shape=(np.shape(I)[0], 2 * np.shape(I)[1]))
    newI[:, ::2] = I
    # ========= Applying X mask ====================

    Ix = []
    for i in range(len(newI[:, 0])):
        Ix.extend([signal.convolve(newI[i, :], G, "same")])  # newI*G ----> x direction
        # Ix.extend([signal.convolve(newI[i,:,1],G,'same')]) # newI*G ----> x direction
        # Ix.extend([signal.convolve(newI[i,:,2],G,'same')]) # newI*G ----> x direction
    Ix = np.array(np.matrix(Ix))

    # =========== Inserting alternate rows of zeros ========
    Ix = newI
    newI = np.zeros(shape=(2 * np.shape(Ix)[0], np.shape(Ix)[1]))
    newI[::2] = Ix
    # ========= Applying Y mask ====================
    Ixy = newI

    Ixy = []
    for i in range(len(newI[0, :])):
        Ixy.extend([signal.convolve(newI[:, i], G, "same")])  # Ixy
        # Ixy.extend([signal.convolve(newI[:,i,1],G,'same')]) # Ixy
        # Ixy.extend([signal.convolve(newI[:,i,2],G,'same')]) # Ixy
    Ixy = np.array(np.matrix(np.transpose(Ixy)))

    return Ixy  # Return Ixy...


def hs_multi(prevImg, nextImg, winSize=39, threshD=1e-9):
    prevImg = prevImg / 1.0
    nextImg = nextImg / 1.0

    # PrevImgBlue,PrevImgGreen,PrevImgrRed = extractChannels(prevImg)
    # NextImgBlue,NextImgGreen,NextImgrRed = extractChannels(prevImg)

    # Calculate derivatives alond x and y

    u, v = hs(img_t=prevImg, img_t1=nextImg, multi_channel=True)

    return u, v


def hs_section(
    I1,  # frame 1
    I2,  # frame 2
    uin,  # u from previous level
    vin,  # v from previous level
):
    """This function runs the HS Algorithm for the current section of pyramid iteratively.
    We take a window of 5x5 and move from left top corner to right bottom corner in order to calculate the vectors for that window.
    Once the whole image is done, we sum the vectors of current and previous levels.
    """

    uin = np.round(uin)
    vin = np.round(vin)
    u = np.zeros([len(I1[:, 0]), len(I1[0, :])])
    v = np.zeros([len(I2[:, 0]), len(I2[0, :])])

    for i in range(2, len(I1[:, 0]) - 2):
        for j in range(2, len(I2[0, :]) - 2):
            I1current = I1[i - 2 : i + 3, j - 2 : j + 3]  # picking 5x5 pixels at a time
            lri = (i - 2) + vin[i, j]  # Low Row Index of the selected window
            hri = (i + 2) + vin[i, j]  # High Row Index of the selected window
            lci = (j - 2) + uin[i, j]  # Low Column Index of the selected window
            hci = (j + 2) + uin[i, j]  # High Column Index of the selected window

            # ============= 5 x 5 Window search ===============
            """
            When the 5x5 window goes beyond the resolution of the concerned image
            we choose the ending 5x5 window for that image.
            """

            if lri < 0:  # if the window goes towards the left of the image
                lri = 0
                hri = 4
            if lci < 0:  # if the window goes above the image
                lci = 0
                hci = 4
            if (
                hri > (len(I1[:, 0])) - 1
            ):  # if the window goes towards the right of the image
                lri = len(I1[:, 0]) - 5
                hri = len(I1[:, 0]) - 1
            if hci > (len(I1[0, :])) - 1:  # if the window goes below the image
                lci = len(I1[0, :]) - 5
                hci = len(I1[0, :]) - 1
            if np.isnan(lri):
                lri = i - 2
                hri = i + 2
            if np.isnan(lci):
                lci = j - 2
                hci = j + 2

            hci = int(hci)
            hri = int(hri)
            lci = int(lci)
            lri = int(lri)

            # Selecting the same window for the second frame

            I2current = I2[lri : (hri + 1), lci : (hci + 1)]

            # Now applying LK for each window of the 2 images
            I1current = I1current / 1.0
            I2current = I2current / 1.0

            u_temp, v_temp = hs(img_t=I1current, img_t1=I2current, multi_channel=True)

            u[i, j] = u_temp[2, 2]
            v[i, j] = v_temp[2, 2]

    u = u + uin
    v = v + vin

    return u, v


def runproglk(
    I1,
    I2,
    iternum,  # Number of iterations per level -- we are using 3
    nlev,  # Number of levels -- we are using 3
):

    I1 = cv.imread(I1)
    I2 = cv.imread(I2)
    I1 = cv.cvtColor(I1, cv.COLOR_BGR2RGB)
    I2 = cv.cvtColor(I2, cv.COLOR_BGR2RGB)
    """ The function uses all the above defined function to implement HS algorithm in a multi-resolution Gaussian pyramid
    framework. We use the following :
    window size of 3x3
    Pyramid size = 3 levels
    Starting from the lower most level, at each level we use the iterative LK algorithm and then warp and upsample it so that it can be used for the next level.
    """
    p1 = np.empty(
        (len(I1[:, 0]), len(I1[0, :]), 3, nlev)
    )  # creating 4d array with different levels for frame 1
    p2 = np.empty(
        (len(I2[:, 0]), len(I2[0, :]), 3, nlev)
    )  # creating 4d array with different levels for frame 2
    p1[:, :, :, 0] = I1  # assign values for Highest level
    p2[:, :, :, 0] = I2  # assign values for Highest level
    """
    p[:,:,0] ---> Level 2 ---> Highest resolution
    p[:,:,2] ---> Level 0 ---> Least resolution
    """
    # Defining the lower levels
    for i in range(1, nlev):
        I1 = DoLow(I1)
        I2 = DoLow(I2)
        p1[0 : int((len(I1[:, 0]))), 0 : int((len(I1[0, :]))), :, i] = I1
        p2[0 : int((len(I2[:, 0]))), 0 : int((len(I2[0, :]))), :, i] = I2

    # ===================== level 0 - Base====================

    l0I1 = p1[0 : int((len(p1[:, 0]) // 4)), 0 : int((len(p1[0, :]) // 4)), :, 2]
    l0I2 = p2[0 : int((len(p2[:, 0]) // 4)), 0 : int((len(p2[0, :]) // 4)), :, 2]
    (u, v) = hs_multi(I1, I2)
    # ============= Iterative LK for that section============

    for j in range(1, iternum + 1):
        (u, v) = hs_section(l0I1, l0I2, u, v)
    # ============= Store U and V values ===================

    ul0 = u
    vl0 = v
    Il0 = l0I1
    ul0[np.where(ul0 == 0)] = np.nan
    vl0[np.where(vl0 == 0)] = np.nan
    # ====================Level 1===================
    k = 1
    ue = DoHigh(u, G)
    ve = DoHigh(v, G)
    I1current = p1[
        0 : (len(p1[:, 0]) // (2 ** (nlev - k - 1))),
        0 : (len(p1[0, :]) // (2 ** (nlev - k - 1))),
        :,
        nlev - k - 1,
    ]
    I2current = p2[
        0 : (len(p2[:, 0]) // (2 ** (nlev - k - 1))),
        0 : (len(p2[0, :]) // (2 ** (nlev - k - 1))),
        :,
        nlev - k - 1,
    ]
    (u, v) = hs_section(I1current, I2current, ue, ve)

    # ========== Iterative HS for that section ===========
    for l in range(1, iternum + 1):
        (u, v) = hs_section(I1current, I2current, ue, ve)

    # ============= Store U and V values ===================
    ul1 = u
    vl1 = v
    Il1 = I1current
    ul1[np.where(ul1 == 0)] = np.nan
    vl1[np.where(vl1 == 0)] = np.nan
    # ====================Level 2=========================
    k = 2
    ue = DoHigh(u, G)
    ve = DoHigh(v, G)
    I1current = p1[
        0 : (len(p1[:, 0]) // (2 ** (nlev - k - 1))),
        0 : (len(p1[0, :]) // (2 ** (nlev - k - 1))),
        :,
        nlev - k - 1,
    ]
    I2current = p2[
        0 : (len(p2[:, 0]) // (2 ** (nlev - k - 1))),
        0 : (len(p2[0, :]) // (2 ** (nlev - k - 1))),
        :,
        nlev - k - 1,
    ]
    (u, v) = hs_section(I1current, I2current, ue, ve)
    # ========== Iterative HS for that section ===========
    for l in range(1, iternum + 1):
        (u, v) = hs_section(I1current, I2current, ue, ve)
    # ============= Store U and V values ===================
    ul2 = u
    vl2 = v
    Il2 = I1current

    return (ul0, vl0, Il0, ul1, vl1, Il1, ul2, vl2, Il2)


def hs_mc_mr_ir(folder):

    (ul01, vl01, Il01, ul11, vl11, Il11, ul21, vl21, Il21) = runproglk(
        "dataset/other-data-color/" + folder + "/frame10.png",
        "dataset/other-data-color/" + folder + "/frame11.png",
        3,
        3,
    )
    print(folder + ": Succesfully calcualted optical flow")
    out = visualize_flow(u=ul21, v=vl21)
    # Save the flow image
    # folder_out = "results/results-other-color-MR-IR-HS/" + folder + "/"
    # cv.imwrite(
    #     f"results/results-other-color-MR-IR-HS/{folder}/{str(folder)}-MR+IR.png", out
    # )

    # Replace Nan's
    u = np.nan_to_num(ul21)
    v = np.nan_to_num(vl21)

    # Save the flow
    np.savetxt(
        "results/results-other-color-pyramid-flow-HS/" + folder + "-U.txt", u, fmt="%d"
    )
    np.savetxt(
        "results/results-other-color-pyramid-flow-HS/" + folder + "-V.txt", v, fmt="%d"
    )

    # Calculate statistics
    mepe, sdEpe = calc_MEPE_directory(folder, u=u, v=v)
    print(
        "The Average End Point Error for "
        + folder
        + " is: "
        + str(mepe)
        + " and the standard deviation is: "
        + str(sdEpe)
    )

    mang, sdAngular = calc_AAE_directory(folder, u=u, v=v)

    print(
        "The Average Angular error for "
        + folder
        + " is: "
        + str(mang)
        + " and the standard deviation is: "
        + str(sdAngular)
    )

    return out
