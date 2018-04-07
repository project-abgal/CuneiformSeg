import os
# change the value according to the GPU you want to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
from glob import glob
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from keras.callbacks import TensorBoard
from model import *
from util import *
import cv2

serv = ".."
pathimg = serv+"/data/image/"
pathans = serv+"/data/image red/"
size = 40
resize = 30

img = [cv2.imread(name)[:, :,::-1] for name in sorted(glob(pathimg + "*"))]
ans = [cv2.imread(name)[:, :,::-1] for name in sorted(glob(pathans + "*"))]
for i in range(len(ans)):
    ans[i] = redcut(ans[i])

# ready for training
import tensorflow as tf
f_log = './log'
tb_cb = TensorBoard(log_dir=f_log, histogram_freq=10000, write_graph=False)
# config = tf.ConfigProto(gpu_options=tf.GPUOptions(
#    per_process_gpu_memory_fraction=0.45))
#session = tf.Session(config=config)
session = tf.Session()
model_size = resize
num_of_classes = 1
mod = get_net(model_size, num_of_classes, session=session)
session.run(tf.global_variables_initializer())

# training
epochs = 10

for e in tqdm(range(epochs)):
    for x, y in BackgroundGenerator(imageGenerator(img, ans, size=size, resize=resize, division=3500), max_prefetch=1):
        mod.fit(x, y, epochs=1, shuffle=True, verbose=1,
                batch_size=1024, callbacks=[tb_cb], validation_split=0.1)
        mod.save_weights(serv + "/model/model-" + str(e) + ".h5")
print('done!')
