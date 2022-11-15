import cv2 as cv
import numpy as np


def images(path: str, multi_channel: bool) -> np.ndarray:
    img = cv.imread(path)

    if not multi_channel:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
    else:
        for i in range(img.shape[2]):
            img[:, :, i] = img[:, :, i].astype(np.float32)

    img = cv.GaussianBlur(img, (3, 3), 2)

    return img


# Derivatives computation
def derivatives_computation(
    img1: np.ndarray, img2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    fx = cv.Sobel(img1, cv.CV_64F, 1, 0, ksize=3)
    fy = cv.Sobel(img1, cv.CV_64F, 0, 1, ksize=3)
    ft = img2 - img1

    return fx, fy, ft


# Defferentiate the calculations between multichannel and single channel
def sigle_multi_channel_derivatives(
    multi_channel: bool, img1: np.ndarray, img2=np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if multi_channel:
        fx, fy, ft = [np.zeros(shape=(img1.shape[0], img1.shape[1])) for _ in range(3)]
        for i in range(img1.shape[2]):
            [fx_c, fy_c, ft_c] = derivatives_computation(
                img1=img1[:, :, i], img2=img2[:, :, i]
            )
            fx = fx + fx_c
            fy = fy + fy_c
            ft = ft + ft_c
    else:
        [fx, fy, ft] = derivatives_computation(img1=img1, img2=img2)
    return fx, fy, ft


# main function
def hs(
    multi_channel: bool,
    img_t_path: str = None,
    img_t1_path: str = None,
    img_t: np.ndarray = None,
    img_t1: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    img_t_path: path to the image in time t
    img_t1_path: path to the image in time t+1
    multi_channel: True if colored image, False for gray images
    """
    if img_t_path and img_t1_path:
        img1 = images(path=img_t_path, multi_channel=multi_channel)
        img2 = images(path=img_t1_path, multi_channel=multi_channel)
    else:
        img1 = img_t
        img2 = img_t1

    fx, fy, ft = sigle_multi_channel_derivatives(multi_channel, img1=img1, img2=img2)

    # Initialize the velocities as zeros (u and v)
    images_shape = img1.shape
    u = np.zeros(shape=(images_shape[0], images_shape[1]), dtype=np.float32)
    v = np.zeros(shape=(images_shape[0], images_shape[1]), dtype=np.float32)

    # Laplace kernel
    hs_kernel = np.array(
        [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], float
    )

    alpha = 10
    itNum = 50
    for i in range(itNum):
        # Computation of the local average
        uAvg = cv.filter2D(u, -1, cv.flip(hs_kernel, -1), borderType=cv.BORDER_CONSTANT)
        vAvg = cv.filter2D(v, -1, cv.flip(hs_kernel, -1), borderType=cv.BORDER_CONSTANT)

        der = (fx * uAvg + fy * vAvg + ft) / (alpha + fx**2 + fy**2)

        u = uAvg - fx * der
        v = vAvg - fy * der

        if i < (itNum - 1):

            u = u * 1000
            v = v * 1000

            u = u.astype(np.uint8)
            v = v.astype(np.uint8)

            u = cv.medianBlur(u, 5)
            u = cv.medianBlur(u, 3)
            v = cv.medianBlur(v, 5)
            v = cv.medianBlur(v, 3)

            u = u / 1000
            v = v / 1000

    return u, v
