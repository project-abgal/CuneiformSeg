import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from glob import glob
import cv2


class TabletDataset(Dataset):
    """
    assumes the image formats(shape, name)
    of img_path and line_img_path is completely the same.
    """

    def __init__(self, img_path, line_img_path, crop_size=(300, 300)):
        #  for name in sorted(glob(img_path + "/*")):
        #      print(name)
        #      cv2.imread(name, 0).astype(np.float32)
        self.img = [cv2.imread(name, 0).astype(np.float32)
                    for name in sorted(glob(img_path + "/*")) if cv2.imread(name, 0).shape[0] > crop_size[0] and cv2.imread(name, 0).shape[1] > crop_size[1]]
        self.line_img = [cv2.imread(name, 0).astype(np.float32)
                         for name in sorted(glob(line_img_path + "/*")) if cv2.imread(name, 0).shape[0] > crop_size[0] and cv2.imread(name, 0).shape[1] > crop_size[1]]
        self.crop_size = crop_size

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        imagesize = self.img[idx].shape
        if imagesize[0] > self.crop_size[0]:
            x = np.random.randint(imagesize[0]-self.crop_size[0])
        else:
            x = 0
        if imagesize[1] > self.crop_size[1]:
            y = np.random.randint(imagesize[1]-self.crop_size[1])
        else:
            y = 0
        return [self.img[idx][x:x+self.crop_size[0], y:y+self.crop_size[1]], self.line_img[idx][x:x+self.crop_size[0], y:y+self.crop_size[1]]]
