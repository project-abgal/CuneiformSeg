import matplotlib.pyplot as plt
# import skimage.data
# import skimage.color
# import skimage.filters
import skimage.util
import skimage.segmentation
import numpy as np
import cv2
from math import ceil
from tqdm import tqdm, tqdm_notebook
import multiprocessing as mp


def slicplus(img, seg=100):
    # plt.imshow(img)
    # plt.show()
    a = skimage.segmentation.slic(img, seg)
    # plt.imshow(a)
    # plt.show()
    # plt.imshow(skimage.segmentation.mark_boundaries(img, a))
    # plt.show()

    lenua = len(np.unique(a))
    x = np.zeros(lenua, dtype=np.uint32)
    y = np.zeros(lenua, dtype=np.uint32)

    for i in range(lenua):
        ai = np.where(a == i)
        x[i] = np.sum(ai[0])//len(ai[0])
        y[i] = np.sum(ai[1])//len(ai[1])

    # print(x)
    # print(y)
    return x, y, a


def redcut(img):
    img[:, :, 0] = np.where(img[:, :, 0] > 0.93, 1, 0)
    img[:, :, 1] = np.where(img[:, :, 1] == 0, 1, 0)
    img[:, :, 2] = np.where(img[:, :, 2] == 0, 1, 0)
    img = np.all(img, 2)
    return img


def squarecut(img, x, y, size=28):
    sizehalf = size//2
    return img[x-sizehalf:x+sizehalf, y-sizehalf:y+sizehalf]


def cropper(image, ansimage, x, y, size, resize):
    imlist = []
    anslist = []
    for i in range(len(x)):
        if x[i]-size//2 >= 0 and y[i]-size//2 >= 0:
            imlist.append(cv2.resize(
                squarecut(image, x[i], y[i], size), (resize, resize)))
            anslist.append(ansimage[x[i], y[i]])
    return(np.array(imlist), np.array(anslist))

def cropperWithOutBlack(image, ansimage, x, y, size, resize):
    # cropperだが真っ黒のものを抜いて返す
    images, answers = cropper(image, ansimage, x, y, size, resize)
    print("images is", len(images))
    blackpart = np.where(images == 0, 1, 0)
    notblack = np.where(np.all(np.all(np.all(blackpart,3),2),1)==False)
    print((np.any(np.any(np.any(blackpart,3),2),1).shape))
    print("notblack is ", notblack, len(notblack))
    images,answers = images[notblack],answers[notblack]
    return images,answers

def imageGenerator(imagelist, ansimagelist, size=28, resize=28, division=1000):
    for image, ansimage in zip(imagelist, ansimagelist):
        x, y, _ = slicplus(image, division)
        x, y = x.astype('int32'), y.astype('int32')
        images, answers = cropperWithOutBlack(image, ansimage, x, y, size, resize)
        yield(images, answers)
    return


class Prefetcher(object):
    def __init__(self, generator, queue_size=2):
        self.queue = mp.Queue(queue_size)
        self.generator = generator
        self.process = mp.Process(target=self.worker)
        self.process.start()

    def worker(self):
        for batch in self.generator:
            self.queue.put(batch)
        self.queue.put(None)

    def __iter__(self):
        return iter(self.queue.get, None)


def visualize(model, image, size=28, resize=28, division=1000, batch_size=128):

    mapping = np.zeros_like(image[:, :, 0], np.float32)
    H, W = image.shape[:2]

    def gen(x, y):
        images = []
        for i, j in zip(tqdm_notebook(x), y):
            sizehalf = size//2
            if i-sizehalf >= 0 and j-sizehalf >= 0:
                images.append(cv2.resize(
                    squarecut(image, i, j, size), (resize, resize)))
                if len(images) >= batch_size:
                    yield [np.array(images)]
                    images = []
        if images != []:
            yield [np.array(images)]
        return

    x, y, a = slicplus(image, division)

    batch_num = ceil(len(x)/batch_size)
    pred = []
    t = tqdm_notebook(total=batch_num)
    for batch in Prefetcher(gen(x, y)):
        t.update()
        pred += list(model.predict_on_batch(batch))
    print(len(pred))

    pred = pred[::-1]

    for j in (range(len(x))):
        if pred != []:
            p = pred.pop()
            mapping[np.where(a == j)] += p

    return(mapping)


if __name__ == "__main__":
    img = cv2.imread("../data/image red/P128014.jpg")[:, :, ::-1]
    img=redcut(img)
    plt.imshow(img)
    plt.show()
    print(img[1682][600])
