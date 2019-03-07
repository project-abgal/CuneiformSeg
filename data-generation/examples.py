import cv2
from glob import glob
import numpy as np

path_to_characters = "../hccc/images/"


def subtract_char(background, patch, x, y):
    cv2.subtract(background[x:x+patch.shape[0], y:y+patch.shape[1]],
                 255-patch, background[x:x+patch.shape[0], y:y+patch.shape[1]])
    #  background[x:x+patch.shape[0],y:y+patch.shape[1]]-=255-patch


if __name__ == "__main__":
    characters = [cv2.cvtColor(cv2.imread(name), cv2.COLOR_RGB2GRAY)
                  for name in sorted(glob(path_to_characters+"*/*"))]
    imgsize = (300, 300)
    img = np.full(imgsize, 255, np.uint8)

    #  print cuneiform characters onto white back
    x = 0
    y = 0
    for i in range(200):
        char_id = np.random.randint(len(characters))
        if x+characters[char_id].shape[0] > imgsize[0]:
            y += characters[char_id].shape[1]
            x = 0
        if y+characters[char_id].shape[1] > imgsize[1]:
            break
        subtract_char(img, characters[char_id], x, y)
        x += characters[char_id].shape[0]

    # turn around black and white
    img = 255-img

    #  zoom up image
    zoom = 5
    img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)

    kernel_size = 5
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # erode
    img = cv2.erode(img, kernel, iterations=3)
    cv2.imwrite("./out/out2.jpg", img)

    # dilate and paint little by little
    iteration = 30
    img = img/iteration
    imgbuf = img
    for i in range(iteration):
        imgbuf = cv2.dilate(imgbuf, kernel, iterations=2)
        #  M = np.float32([[1, 0, -kernel_size//2], [0, 1, -kernel_size//2]])
        #  imgbuf = cv2.warpAffine(imgbuf, M,imgbuf.shape[::-1])
        #  img = np.maximum(img, imgbuf)
        img = cv2.addWeighted(img, 1, imgbuf, 1, 0)



    #  optional lise detection. doesn't work well.
    detectLines = False
    if detectLines:
        gray = img
        minLineLength = 100
        maxLineGap = 1
        lines = cv2.HoughLinesP(
            gray, 1, np.pi/360, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imwrite("./out/out.jpg", img)
