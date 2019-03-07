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
    imgsize = (1000, 300)
    img = np.full(imgsize, 255, np.uint8)

    #  print cuneiform characters onto white back
    x = 0
    y = 0
    for i in range(200):
        if x+characters[i].shape[0] > imgsize[0]:
            y += characters[i].shape[1]
            x = 0
        if y+characters[i].shape[1] > imgsize[1]:
            break
        subtract_char(img, characters[i], x, y)
        x += characters[i].shape[0]

    # turn around black and white
    img = 255-img

    #  zoom up image
    zoom = 5
    img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)

    # erode
    kernel = np.ones((2, 2), np.uint8)
    #  img = cv2.erode(img, kernel, iterations=3)

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
