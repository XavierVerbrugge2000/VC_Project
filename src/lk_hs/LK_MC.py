import numpy as np
from matplotlib import pyplot as plt
import sys, os, cv2, time
from src.lk_hs.lukas_kanade import *
from src.representation.colorwhele import visualize_flow


def calcDerivativesForEachChannel(img):
    # Extract the channels
    red = img[:, :, 2]
    green = img[:, :, 1]
    blue = img[:, :, 0]

    # Calculate X derivatives
    I0x = cv2.Sobel(blue, cv2.CV_64F, 1, 0, ksize=3)
    I1x = cv2.Sobel(green, cv2.CV_64F, 1, 0, ksize=3)
    I2x = cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=3)

    # Calculate Y derivatives
    I0y = cv2.Sobel(blue, cv2.CV_64F, 0, 1, ksize=3)
    I1y = cv2.Sobel(green, cv2.CV_64F, 0, 1, ksize=3)
    I2y = cv2.Sobel(red, cv2.CV_64F, 0, 1, ksize=3)

    return I0x, I1x, I2x, I0y, I1y, I2y


def LK_Multi_Channel(prevImg, nextImg, winSize=39, threshD=1e-9):
    prevImg = prevImg / 1.
    nextImg = nextImg / 1.

    # Calculate derivatives x and y

    prevI0x, prevI1x, prevI2x, prevI0y, prevI1y, prevI2y = calcDerivativesForEachChannel(prevImg)
    nextI0x, nextI1x, nextI2x, nextI0y, nextI1y, nextI2y = calcDerivativesForEachChannel(nextImg)

    # compromise solution by the least squares principle

    sigma1 = prevI0x ** 2 + prevI1x ** 2 + prevI2x ** 2 + nextI0x ** 2 + nextI1x ** 2 + nextI2x ** 2

    sigma2 = sigma4 = prevI0x * prevI0y + prevI1x * prevI1y + prevI2x * prevI2y + nextI0x * nextI0y + nextI1x * nextI1y + nextI2x * nextI2y

    sigma3 = (nextImg[:, :, 0] - prevImg[:, :, 0]) * (prevI0x + nextI0x) + (nextImg[:, :, 1] - prevImg[:, :, 1]) * (
            prevI1x + nextI1x) + (nextImg[:, :, 2] - prevImg[:, :, 2]) * (prevI2x + nextI2x)

    sigma5 = prevI0y ** 2 + prevI1y ** 2 + prevI2y ** 2 + nextI0y ** 2 + nextI1y ** 2 + nextI2y ** 2

    sigma6 = (nextImg[:, :, 0] - prevImg[:, :, 0]) * (prevI0y + nextI0y) + (nextImg[:, :, 1] - prevImg[:, :, 1]) * (
            prevI1y + nextI1y) + (nextImg[:, :, 2] - prevImg[:, :, 2]) * (prevI2y + nextI2y)

    sigma = [sigma1, sigma2, sigma3, sigma4, sigma5, sigma6]

    for g in range(len(sigma)):
        # apply average blurring
        # sigma[i] = cv2.blur(sigma[i], (winSize, winSize))
        # Gaussian blurring. The sigma determines the contribution of neighbours.
        sigma[g] = cv2.GaussianBlur(sigma[g], (winSize, winSize), 0)

    # Use Cramer's rule to solve the equation

    # Calculate determinant of the coefficients
    D = sigma[1] ** 2 - sigma[0] * sigma[4]
    D[np.abs(D) < threshD] = np.Inf

    # Cramers rule
    # https://pressbooks.bccampus.ca/algebraintermediate/chapter/solve-systems-of-equations-using-determinants/
    # Replace the X coefficients with the constants and take the determinant 
    u = (sigma[4] * sigma[2] - sigma[1] * sigma[5]) / D
    # Replace the Y coefficients with the constants and take the determinant 
    v = (sigma[1] * sigma[2] - sigma[0] * sigma[5]) / D

    # length -> euclidean distance
    arrow = np.sqrt(u ** 2 + v ** 2)

    return u, -v, arrow


def run_color(prevImg, nextImg):
    prevRGB = cv2.cvtColor(prevImg, cv2.COLOR_BGR2RGB)
    nextRGB = cv2.cvtColor(nextImg, cv2.COLOR_BGR2RGB)
    u, v, arrow = LK_Multi_Channel(prevRGB, nextRGB)
    coor = getCoor(u, v, arrow)
    out = drawArrow(prevImg, u, v, coor)
    return out


def main_color(folder_name):
    folder_in = 'dataset/other-data-color/' + folder_name + '/'
    folder_out = 'results/result-other-color-LK/' + folder_name + '/'
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    files = os.listdir(folder_in)
    files.sort()
    for i in range(len(files) - 1):
        prevImg = cv2.imread(folder_in + files[i])
        nextImg = cv2.imread(folder_in + files[i + 1])
        out = run_color(prevImg, nextImg)
        cv2.imwrite(folder_out + str(i) + '.png', out)


def LK_MC_calc_all_flow():
    folders = ["Dimetrodon","Grove2","Grove3","Hydrangea","RubberWhale","Urban2","Urban3","Venus"]

    for folder in folders:
        prevImg= cv2.imread("dataset/other-data-color/"+folder+"/frame10.png")
        nextImg= cv2.imread("dataset/other-data-color/"+folder+"/frame11.png")
        prevRGB = cv2.cvtColor(prevImg, cv2.COLOR_BGR2RGB)
        nextRGB = cv2.cvtColor(nextImg, cv2.COLOR_BGR2RGB)
        # Calculating optical flow
        u, v, arrow= LK_Multi_Channel(prevRGB, nextRGB)
        print(folder +": Succesfully calcualted optical flow")
        # Visualizing flow
        out = visualize_flow(u = u, v = v)
        cv2.imshow(folder, out)
        folder_out = 'results/result-other-color-LK/'+folder+'/'
        cv2.imwrite(folder_out+str(folder)+'-colorwheel.png', out)

        #Show plot
        plt.imshow(out)
        plt.show()

        # Replace Nan's
        u = np.nan_to_num(u)
        v = np.nan_to_num(v)

        # Calculate statistics
        mepe, sdEpe = calc_MEPE_directory(folder, u = u, v = v)
        print("The Average End Point Error for "+folder+ " is: " + str(mepe)+" and the standard deviation is: "+str(sdEpe))
        mang, sdAngular = calc_AAE_directory(folder, u= u, v = v)
        print("The Average Angular error for "+folder+ " is: " + str(mang)+" and the standard deviation is: "+str(sdAngular)+"/n")#calc_all_flow()
        print("\n")

def video(infile, outfile='MultiChannel_result.avi'):
    assert os.path.exists(infile), "video doesn't exist"
    cap = cv2.VideoCapture(infile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out_video = cv2.VideoWriter(outfile, fourcc, fps, (int(width), int(height)))
    _, old_frame = cap.read()
    while 1:
        ret, frame = cap.read()
        if ret:
            out = run_color(old_frame, frame)
            cv2.imshow('frame', out)
            out_video.write(out)
        else:
            break
        old_frame = frame
    cv2.destroyAllWindows()
    out_video.release()
    cap.release()
