import matplotlib.pyplot as plt
from scipy import signal
import math
import cv2
from pylab import *
from src.lk_hs.LK_MC import  *
from src.representation.colorwhele import visualize_flow
from src.evaluation.end_point_error import calc_MEPE_directory
from src.evaluation.angular_error import calc_AAE_directory


# the function for gaussian filter
def gmask(x, y, s):
    gmask = np.exp(-((x ** 2) + (y ** 2)) // 2 // s ** 2)
    return gmask


# Utilize the standard deviation from the paper.
def kernel(size=2):
    s = math.sqrt(2 / (4 * math.pi))
    G = []  # Gaussian Kernel
    for i in range(-size, size + 1):
        G.append(gmask(i, 0, s))  # equating y to 0 since we need a 1D matrix
    return G

def DownSample (I): # the function applies gaussian filter and down samples the intput image by 2

    tau = 0.5
    sigma = math.sqrt(2/(4*tau))
    I = cv2.GaussianBlur(I,(5,5),sigma)

    # Resize

    width = int(I.shape[1]  /2 )
    height = int(I.shape[0] /2)
    dim = (width, height)

    # resize image
    I_resized = cv2.resize(I, dim, interpolation = cv2.INTER_CUBIC)

    return I_resized

'''

def DownSample(I):  # the function down samples the intput image by 2
     Algorithm: For an input image of size rxc
           1) Apply x mask
           2) Delete Alternate Columns
           3) Apply y mask
           4) delete alternate rows
           The ouput image is of size (r//2)x(c//2)

    # Gather Kernel:
    G = kernel()

    # ========= Applying X mask ====================

    Ix = []
    for i in range(len(I[:, 0])):
        Ix.extend([signal.convolve(I[i, :, 0], G, 'same')])  # Ix*Gx = Ix
        Ix.extend([signal.convolve(I[i, :, 1], G, 'same')])  # Ix*Gx = Ix
        Ix.extend([signal.convolve(I[i, :, 2], G, 'same')])  # Ix*Gx = Ix

    Ix = array(matrix(Ix))


    # ========selecting alternate columns===========
    Ix = I
    Ix = Ix[:, ::2, :]
    # ========= Applying Y mask ====================

    Ixy = []
    for i in range(len(Ix[0, :])):
        Ixy.extend([signal.convolve(Ix[:, i, 0], G, 'same')])  # Ix * Gy = Ixy
        Ixy.extend([signal.convolve(Ix[:, i, 1], G, 'same')])  # Ix * Gy = Ixy
        Ixy.extend([signal.convolve(Ix[:, i, 2], G, 'same')])  # Ix * Gy = Ixy

    Ixy = array(matrix(Ixy))

    # ========selecting alternate rows===========
    Ixy = Ix
    Ixy = Ixy[::2, :, :]

    return Ixy
'''

def UpSample(I, G):  # the function up samples the intput image by 2
    ''' Algorithm: For an input image of size rxc
           1) Insert zero column on every alternate columns
           2) Apply x mask
           3) Insert zero row on every alternate rows
           4) Apply y mask
           The ouput image is of size (2r)x(2c)'''

    G = np.dot(2, G)  # Doubing the Guassian kernel since we later use alternate rows and columns
    # =========== Inserting alternate columns of zeros ========
    newI = np.zeros(shape=(shape(I)[0], 2 * shape(I)[1]))
    newI[:, ::2] = I
    # ========= Applying X mask ====================

    Ix = []
    for i in range(len(newI[:, 0])):
        Ix.extend([signal.convolve(newI[i, :], G, 'same')])  # newI*G ----> x direction

    Ix = array(matrix(Ix))

    # =========== Inserting alternate rows of zeros ========

    newI = np.zeros(shape=(2 * shape(Ix)[0], shape(Ix)[1]))
    newI[::2] = Ix
    # ========= Applying Y mask ====================

    Ixy = []
    for i in range(len(newI[0, :])):
        Ixy.extend([signal.convolve(newI[:, i], G, 'same')])  # Ixy

    Ixy = array(matrix(transpose(Ixy)))

    return Ixy  # Return Ixy...


def LK_Iterative(I1  # frame 1
                 , I2  # frame 2
                 , uin  # u from previous level
                 , vin  # v from previous level
                 ):
    '''This function runs the LK Algorithm for the current section of pyramid iteratively.
    We take a window of 5x5 and move from left top corner to right bottom corner in order to calculate the vectors for that window.
    Once the whole image is done, we sum the vectors of current and previous levels.
    '''

    uin = np.round(uin)
    vin = np.round(vin)
    u = np.zeros([len(I1[:, 0]), len(I1[0, :])])
    v = np.zeros([len(I2[:, 0]), len(I2[0, :])])

    for i in range(2, len(I1[:, 0]) - 2):
        for j in range(2, len(I2[0, :]) - 2):

            I1current = I1[i - 2:i + 3, j - 2:j + 3]  # picking 5x5 pixels at a time
            lri = (i - 2) + vin[i, j]  # Low Row Index of the selected window
            hri = (i + 2) + vin[i, j]  # High Row Index of the selected window
            lci = (j - 2) + uin[i, j]  # Low Column Index of the selected window
            hci = (j + 2) + uin[i, j]  # High Column Index of the selected window

            # ============= 5 x 5 Window search ===============
            '''
            When the 5x5 window goes beyond the resolution of the concerned image
            we choose the ending 5x5 window for that image.
            '''

            if (lri < 0):  # if the window goes towards the left of the image
                lri = 0
                hri = 4
            if (lci < 0):  # if the window goes above the image
                lci = 0
                hci = 4
            if (hri > (len(I1[:, 0])) - 1):  # if the window goes towards the right of the image
                lri = len(I1[:, 0]) - 5
                hri = len(I1[:, 0]) - 1
            if (hci > (len(I1[0, :])) - 1):  # if the window goes below the image
                lci = len(I1[0, :]) - 5
                hci = len(I1[0, :]) - 1
            if (np.isnan(lri)):
                lri = i - 2
                hri = i + 2
            if (np.isnan(lci)):
                lci = j - 2
                hci = j + 2

            hci = int(hci)
            hri = int(hri)
            lci = int(lci)
            lri = int(lri)

            # Selecting the same window for the second frame
            # Wrap
            I2current = I2[lri:(hri + 1), lci:(hci + 1)]
            #plt.imshow(I2current.astype('uint8'))
            #plt.show()

            u_iter, v_iter,_ = LK_Multi_Channel(I1current, I2current)

            u[i, j] = u_iter[2, 2]
            v[i, j] = v_iter[2, 2]

    u = u + uin
    v = v + vin

    return u, v


def LK_MC_MR_IR(I1,
                I2,
                iternum,  # Number of iterations per level -- we are using 3
                nlev  # Number of levels -- we are using 3
                ):
    I1 = cv2.imread(I1)
    I2 = cv2.imread(I2)
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
    ''' The function uses all the above defined function to implement LK algorithm in a multi-resolution Gaussian pyramid
    framework. We use the following :
    window size of 3x3
    Pyramid size = 3 levels
    Starting from the lower most level, at each level we use the iterative LK algorithm and then warp and upsample it so that it can be used for the next level.
    '''
    p1 = np.empty((len(I1[:, 0]), len(I1[0, :]), 3, nlev))  # creating 4d array with different levels for frame 1
    p2 = np.empty((len(I2[:, 0]), len(I2[0, :]), 3, nlev))  # creating 4d array with different levels for frame 2
    p1[:, :, :, 0] = I1  # assign values for Highest level
    p2[:, :, :, 0] = I2  # assign values for Highest level
    '''
    p[:,:,0] ---> Level 2 ---> Highest resolution
    p[:,:,2] ---> Level 0 ---> Least resolution
    '''
    # Defining the lower levels
    for i in range(1, nlev):
        I1 = DownSample(I1)
        I2 = DownSample(I2)
        p1[0:int((len(I1[:, 0]))), 0:int((len(I1[0, :]))), :, i] = I1
        p2[0:int((len(I2[:, 0]))), 0:int((len(I2[0, :]))), :, i] = I2

    # ===================== level 0 - Base====================

    l0I1 = p1[0:int((len(p1[:, 0]) // 4)), 0:int((len(p1[0, :]) // 4)), :, 2]
    l0I2 = p2[0:int((len(p2[:, 0]) // 4)), 0:int((len(p2[0, :]) // 4)), :, 2]
    u, v,_ = LK_Multi_Channel(I1, I2)

    # ============= Iterative LK for that section============

    for j in range(1, iternum + 1):
        (u, v) = LK_Iterative(l0I1, l0I2, u, v)
    # ============= Store U and V values ===================

    ul0 = u
    vl0 = v
    Il0 = l0I1
    ul0[np.where(ul0 == 0)] = nan
    vl0[np.where(vl0 == 0)] = nan
    # ====================Level 1===================
    k = 1
    G = kernel()
    ue = UpSample(u, G)
    ve = UpSample(v, G)
    I1current = p1[0:(len(p1[:, 0]) // (2 ** (nlev - k - 1))), 0:(len(p1[0, :]) // (2 ** (nlev - k - 1))), :,
                nlev - k - 1]
    I2current = p2[0:(len(p2[:, 0]) // (2 ** (nlev - k - 1))), 0:(len(p2[0, :]) // (2 ** (nlev - k - 1))), :,
                nlev - k - 1]
    (u, v) = LK_Iterative(I1current, I2current, ue, ve)

    # ========== Iterative LK for that section ===========
    for l in range(1, iternum + 1):
        (u, v) = LK_Iterative(I1current, I2current, ue, ve)

    # ============= Store U and V values ===================
    ul1 = u
    vl1 = v
    Il1 = I1current
    ul1[np.where(ul1 == 0)] = nan
    vl1[np.where(vl1 == 0)] = nan
    # ====================Level 2=========================
    k = 2
    ue = UpSample(u, G)
    ve = UpSample(v, G)
    I1current = p1[0:(len(p1[:, 0]) // (2 ** (nlev - k - 1))), 0:(len(p1[0, :]) // (2 ** (nlev - k - 1))), :,
                nlev - k - 1]
    I2current = p2[0:(len(p2[:, 0]) // (2 ** (nlev - k - 1))), 0:(len(p2[0, :]) // (2 ** (nlev - k - 1))), :,
                nlev - k - 1]
    (u, v) = LK_Iterative(I1current, I2current, ue, ve)
    # ========== Iterative LK for that section ===========
    for l in range(1, iternum + 1):
        (u, v) = LK_Iterative(I1current, I2current, ue, ve)
    # ============= Store U and V values ===================
    u_final = u
    v_final = v
    Il2 = I1current

    return u_final,v_final


def LK_MC_MR_IR_calc_all_flows():
    folders = ["Dimetrodon", "Grove2", "Grove3", "Hydrangea", "RubberWhale", "Urban2", "Urban3", "Venus"]

    for folder in folders:

        u,v = LK_MC_MR_IR(
            'dataset/other-data-color/' + folder + '/frame10.png',
            'dataset/other-data-color/' + folder + '/frame11.png', 3, 3)
        print(folder + ": Succesfully calcualted optical flow")
        out = visualize_flow(u=u, v=u)
        # Save the flow image
        folder_out = "results/results-other-color-MR-IR/" + str(folder) + "/"
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        cv2.imwrite(folder_out + str(folder) + "-6-MR+IR.png", out)
        #Show plot
        plt.imshow(out)
        plt.show()
        # Replace Nan's
        u = np.nan_to_num(u)
        v = np.nan_to_num(u)

        # Save the flow
        np.savetxt('results/results-other-color-pyramid-flow/' + folder + '-U.txt', u, fmt='%d')
        np.savetxt('results/results-other-color-pyramid-flow/' + folder + '-V.txt', v, fmt='%d')

        # Calculate statistics
        mepe, sdEpe = calc_MEPE_directory(folder, u=u, v=v)
        print("The Average End Point Error for " + folder + " is: " + str(
            mepe) + " and the standard deviation is: " + str(sdEpe))

        mang, sdAngular = calc_AAE_directory(folder, u=u, v=v)

        print(
            "The Average Angular error for " + folder + " is: " + str(mang) + " and the standard deviation is: " + str(
                sdAngular))
        print("\n")
