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

    return img


# Derivatives computation
def derivatives_computation(
    img1: np.ndarray, img2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # X, Y, T kernel
    kernelX = np.array([[-1, 1], [-1, 1]]) * 0.25
    kernelY = np.array([[1, 1], [-1, -1]]) * 0.25
    kernelT = np.ones((2, 2)) * 0.25

    fx = cv.filter2D(
        img1, -1, cv.flip(kernelX, -1), borderType=cv.BORDER_CONSTANT
    ) + cv.filter2D(img2, -1, cv.flip(kernelX, -1), borderType=cv.BORDER_CONSTANT)
    fy = cv.filter2D(
        img1, -1, cv.flip(kernelY, -1), borderType=cv.BORDER_CONSTANT
    ) + cv.filter2D(img2, -1, cv.flip(kernelY, -1), borderType=cv.BORDER_CONSTANT)
    ft = cv.filter2D(
        img1, -1, cv.flip(kernelT, -1), borderType=cv.BORDER_CONSTANT
    ) + cv.filter2D(img2, -1, cv.flip(-kernelT, -1), borderType=cv.BORDER_CONSTANT)

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
def hs(img_t: str, img_t1: str, multi_channel: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    img_t: path to the image in time t
    img_t1: path to the image in time t+1
    multi_channel: True if colored image, False for gray images
    """
    img1 = images(path=img_t, multi_channel=multi_channel)
    img2 = images(path=img_t1, multi_channel=multi_channel)

    fx, fy, ft = sigle_multi_channel_derivatives(multi_channel, img1=img1, img2=img2)

    # Initialize the velocities as zeros (u and v)
    images_shape = img1.shape
    u = np.zeros(shape=(images_shape[0], images_shape[1]), dtype=np.float32)
    v = np.zeros(shape=(images_shape[0], images_shape[1]), dtype=np.float32)

    # Laplace kernel
    hs_kernel = np.array(
        [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], float
    )

    alpha = 0.001
    for _ in range(1000):
        # Computation of the local average
        uAvg = cv.filter2D(u, -1, cv.flip(hs_kernel, -1), borderType=cv.BORDER_CONSTANT)
        vAvg = cv.filter2D(v, -1, cv.flip(hs_kernel, -1), borderType=cv.BORDER_CONSTANT)

        der = (fx * uAvg + fy * vAvg + ft) / (alpha**2 + fx**2 + fy**2)

        u = uAvg - fx * der
        v = vAvg - fy * der

    return u, v
