from glob import glob
from tqdm import tqdm
import cv2
from multiprocessing import Pool


def create_data(name):
    imageID = name[name.find('/P')+1:]
    print(imageID)
    gray_img = cv2.imread(name, 0)
    cv2.imwrite("../images/"+imageID, gray_img)
    line_img = cv2.Canny(gray_img, 150,200)
    cv2.imwrite("../line_images/"+imageID, line_img)


if __name__ == '__main__':
    pathimg = "../../data-collection/images/"

    p = Pool(7)
    p.map(create_data, sorted(glob(pathimg + "*")))

    #  for name in tqdm(sorted(glob(pathimg + "*"))):
