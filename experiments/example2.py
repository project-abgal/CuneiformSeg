import cv2
import numpy as np

path_to_characters = "../hccc/images/"


def subtract_char(background, patch, x, y):
    cv2.subtract(background[x:x+patch.shape[0], y:y+patch.shape[1]],
                 255-patch, background[x:x+patch.shape[0], y:y+patch.shape[1]])
    #  background[x:x+patch.shape[0],y:y+patch.shape[1]]-=255-patch


def dilate_and_diff(img):
    img = img.copy()
    neiborhood24 = np.ones((5, 5))
    dilated = cv2.dilate(img, neiborhood24, iterations=1)
    return cv2.absdiff(img, dilated)


def canny_edge(img):
    img = img.copy()
    img = cv2.blur(img, (kernel_size, kernel_size))
    img = img + np.random.normal(0, 0.1, img.shape)
    img = np.uint8(img)
    return cv2.Canny(img, 10, 20)


def ridge(img):
    img = img.copy()
    sigmaX = 1
    sigmaY = 1

    img = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=sigmaX, sigmaY=sigmaY)
    filter_ = cv2.ximgproc.RidgeDetectionFilter_create(
        ksize=3, dx=1, dy=1)
    img = filter_.getRidgeFilteredImage(img)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3)))
    #  img = cv2.erode(img, kernel=np.ones((3,3)))
    return img


if __name__ == "__main__":
    #  img = cv2.imread('../data/image/P499212.jpg')
    img = cv2.imread('../data-collection/images/P100004.jpg')
    kernel_size = 3
    #  img = canny_edge(img)
    img = ridge(img)
    cv2.imwrite("./out/out5.jpg", img)
