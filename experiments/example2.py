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
    outimg = img.copy()
    sigmaX = 1
    sigmaY = 1

    outimg = cv2.GaussianBlur(outimg, ksize=(
        9, 9), sigmaX=sigmaX, sigmaY=sigmaY)
    ridge_detection_filter = cv2.ximgproc.RidgeDetectionFilter_create(
        ksize=3, dx=1, dy=1)
    outimg = ridge_detection_filter.getRidgeFilteredImage(outimg)
    outimg = cv2.morphologyEx(outimg, cv2.MORPH_OPEN, np.ones((3, 3)))
    cv2.imwrite("./out/out6.jpg", outimg)
    #  img = cv2.erode(img, kernel=np.ones((3,3)))
    mser = cv2.MSER_create(_min_area=100, _max_area=500, _max_variation=0.1)
    regions, _ = mser.detectRegions(255-outimg)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(img, hulls, 1, (0, 255, 0))
    return img


def mser(img):
    outimg = img.copy()
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(outimg)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(img, hulls, 1, (0, 255, 0))
    return img


if __name__ == "__main__":
    #  img = cv2.imread('../data/image/P499212.jpg')
    img = cv2.imread('../data-collection/images/P100004.jpg')
    kernel_size = 3
    print(img.shape)
    #  img = canny_edge(img)
    img = ridge(img)
    #  img = mser(img)
    cv2.imwrite("./out/out5.jpg", img)
