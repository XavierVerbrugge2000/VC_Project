

import numpy as np
from matplotlib import pyplot as plt
import sys, os, cv2, time
from src.representation.colorwhele import visualize_flow
from src.evaluation.end_point_error import calc_MEPE_directory
from src.evaluation.angular_error import calc_AAE_directory
import cv2


def LK(prevImg, nextImg, winSize = 39, threshD = 1e-9):
    prevImg = prevImg/1.
    nextImg = nextImg/1.

    # Calculate derivatives alond x and y
    prevDx = cv2.Sobel(prevImg,cv2.CV_64F,1,0, ksize=3)
    prevDy = cv2.Sobel(prevImg,cv2.CV_64F,0,1, ksize=3)
    nextDx = cv2.Sobel(nextImg,cv2.CV_64F,1,0, ksize=3)
    nextDy = cv2.Sobel(nextImg,cv2.CV_64F,0,1, ksize=3)

    # compromise solution by the least squares principle

    sigma1 = prevDx**2 + nextDx**2
    sigma2 = sigma4 = prevDx*prevDy+nextDx*nextDy
    sigma3 = (nextImg-prevImg)*(prevDx+nextDx)
    sigma5 = prevDy**2 + nextDy**2
    sigma6 = (nextImg-prevImg)*(prevDy+nextDy)
    sigma = [sigma1, sigma2, sigma3, sigma4, sigma5, sigma6]

    for i in range(len(sigma)):
        # apply average blurring
        sigma[i] = cv2.blur(sigma[i], (winSize, winSize))

    # Use Cramer's rule to solve the equation

    # Calculate determinant of the coefficients
    D = sigma[1]**2 - sigma[0]*sigma[4]
    D[np.abs(D) < threshD] = np.Inf


    # Cramers rule
    # https://pressbooks.bccampus.ca/algebraintermediate/chapter/solve-systems-of-equations-using-determinants/
    # Replace the X coefficients with the constants and take the determinant
    u = (sigma[4]*sigma[2]-sigma[1]*sigma[5]) / D
    # Replace the Y coefficients with the constants and take the determinant
    v = (sigma[1]*sigma[2]-sigma[0]*sigma[5]) / D

    # length -> euclidean distance
    arrow = np.sqrt(u**2+v**2)

    return u, -v, arrow


def getCoor(u, v, arrow, step = 20, percent = 0.1):
    y, x = np.meshgrid(np.arange(0, u.shape[0], step), np.arange(0, u.shape[1], step))
    coor = np.vstack((x.flatten(), y.flatten()))

    arrow = arrow[y, x]
    index = np.argsort(arrow.flatten())
    coor = coor[:, index[int(-len(index)*percent):]]
    return coor


def drawArrow(img, u, v, coor, scale=50):
    color = (0, 255, 255)
    mask = np.zeros_like(img)
    m = np.max((np.abs(u), np.abs(v)))
    if m < 0.5:
        scale /=0.5
    for i in range(coor.shape[1]):
        x1 = int(coor[0, i])
        y1 = int(coor[1, i])
        x2 = int(coor[0, i]+u[y1, x1]*scale)
        y2 = int(coor[1, i]+v[y1, x1]*scale)
        mask = cv2.line(mask, (x1, y1), (x2, y2), color, thickness=1)
        mask = cv2.circle(mask,(x1,y1), 3, color, -1)
    out = cv2.add(img, mask)
    return out


def run(prevImg, nextImg):
    prevGray = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
    nextGray = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
    u, v, arrow= LK(prevGray, nextGray)
    coor = getCoor(u, v, arrow)
    out = drawArrow(prevImg, u, v, coor)
    return out


def main(folder_name):
    folder_in = 'dataset/other-data-gray/'+folder_name+'/'
    folder_out = 'results/result-other-grey-LK/'+folder_name+'/'
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    files = os.listdir(folder_in)
    files.sort()
    for i in range(len(files)-1):
        prevImg = cv2.imread(folder_in+files[i])
        nextImg = cv2.imread(folder_in+files[i+1])
        out = run(prevImg, nextImg)
        cv2.imwrite(folder_out+str(i)+'.png', out)


def calc_all():
    folders = ["Army","Backyard","Basketball","Dumptruck","Evergreen","Grove","Mequon","Schefflera","Teddy","Urban","Wooden","Yosemite"]
    for folder in folders:
        main(folder)
        print(folder +": Succesfully calcualted optical flow")


def calc_all_other():
    folders = ["Beanbags","Dimetrodon","DogDance","Grove2","Grove3","vdrangea","MiniCooper","RubberWhale","Urban2","Urban3","Venus","Walking"]
    for folder in folders:
        main(folder)
        print(folder +": Succesfully calcualted optical flow")


def video(infile, outfile):
    assert os.path.exists(infile), "video doesn't exist"
    cap = cv2.VideoCapture(infile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out_video = cv2.VideoWriter(outfile, fourcc, fps, (int(width), int(height)))
    _, old_frame = cap.read()
    while 1:
        ret, frame = cap.read()
        if ret:
            out = run(old_frame, frame)
            cv2.imshow('frame', out)
            out_video.write(out)
        else: break
        old_frame = frame
    cv2.destroyAllWindows()
    out_video.release()
    cap.release()


def LK_calc_all_flow():
    folders = ["Dimetrodon","Grove2","Grove3","hydrangea","RubberWhale","Urban2","Urban3","Venus"]
    

    for folder in folders:
        prevImg= cv2.imread("dataset/other-data-gray/"+folder+"/frame10.png")
        nextImg= cv2.imread("dataset/other-data-gray/"+folder+"/frame11.png")
        prevGray = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
        nextGray = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
        u, v, arrow = LK(prevGray, nextGray)
        print(folder +": Succesfully calcualted optical flow")
        out = visualize_flow(u = u, v = v)
        plt.imshow(out)
        plt.show()
        # Save the flow image
        folder_out = "results/result-other-grey-LK/"+str(folder)+"/"
        if not os.path.exists(folder_out):
            os.mkdir(folder_out)
        cv2.imwrite(folder_out+ str(folder)+".png", out)
        # Replace Nan's
        u = np.nan_to_num(u)
        v = np.nan_to_num(v)

        # Calculate statistics
        mepe, sdEpe = calc_MEPE_directory(folder, u = u, v = v)
        print("The Average End Point Error for "+ folder + " is: " + str(mepe)+" and the standard deviation is: "+str(sdEpe))

        mang, sdAngular = calc_AAE_directory(folder, u= u, v = v)

        print("The Average Angular error for "+folder+ " is: " + str(mang)+" and the standard deviation is: "+str(sdAngular))
        print("\n")














