import numpy as np
from glob import glob
from tqdm import tqdm
from util import *
import cv2

if __name__ == '__main__':
    serv = ".."
    pathimg = serv+"/data/image/"
    pathans = serv+"/data/image red/"

    size = 64
    resize = 64
    img = [cv2.imread(name)[:, :, ::-1]
           for name in sorted(glob(pathimg + "*"))]
    ans = [cv2.imread(name)[:, :, ::-1]
           for name in sorted(glob(pathans + "*"))]
    for i in range(len(ans)):
        ans[i] = redcut(ans[i])

    imagelist, ansimagelist = imageProcessor(
        img, ans, size, resize, division=3500)
    np.save("imagelist", imagelist)
    np.save("ansimagelist", ansimagelist)
