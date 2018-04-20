#requires running preprocessor.py beforehand to create data.
import os
# change the value according to the GPU you want to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from glob import glob
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from keras.callbacks import TensorBoard
from model import *
from util import *
import cv2

serv = ".."


# ready for training
import tensorflow as tf
f_log = './log'
tb_cb = TensorBoard(log_dir=f_log, histogram_freq=10000, write_graph=False)
# config = tf.ConfigProto(gpu_options=tf.GPUOptions(
#    per_process_gpu_memory_fraction=0.45))
#session = tf.Session(config=config)
session = tf.Session()
size = 64
resize = size
model_size = resize
num_of_classes = 1
mod = get_net2(model_size, num_of_classes, session=session)
mod.load_weights(serv + "/model/model.h5")
session.run(tf.global_variables_initializer())

# training
epochs = 100

#requires running preprocessor.py beforehand to create data.
imagelist = np.load("imagelist.npy")
ansimagelist = np.load("ansimagelist.npy")

mod.fit(imagelist, ansimagelist, epochs=epochs, shuffle=True, verbose=1,
        batch_size=128, callbacks=[tb_cb], validation_split=0.1)
mod.save_weights(serv + "/model/model.h5")
print('done!')
