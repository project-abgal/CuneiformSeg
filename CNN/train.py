# requires running preprocessor.py beforehand to create data.
import os
# change the value according to the GPU you want to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from glob import glob
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from keras.callbacks import TensorBoard, EarlyStopping
from model import *
from util import *
from sklearn.model_selection import train_test_split
import cv2

serv = ".."


# ready for training
import tensorflow as tf
f_log = './log'
tb_cb = TensorBoard(log_dir=f_log, histogram_freq=10000, write_graph=False)
es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
# use this to limit gpu memory to half of it
# config = tf.ConfigProto(gpu_options=tf.GPUOptions(
#    per_process_gpu_memory_fraction=0.45))
#session = tf.Session(config=config)
session = tf.Session()
size = 64
resize = size
model_size = resize
num_of_classes = 1
mod = get_net2(model_size, num_of_classes, session=session)
# comment out the following line if this is the first train
# mod.load_weights(serv + "/model/model.h5")
session.run(tf.global_variables_initializer())

# training
epochs = 100

# requires running preprocessor.py beforehand to create data.
imagelist = np.load("imagelist.npy")
ansimagelist = np.load("ansimagelist.npy")

X_train, X_test, y_train, y_test = train_test_split(
    imagelist, ansimagelist, test_size=0.10, random_state=42)


mod.fit(X_train, y_train, epochs=epochs, shuffle=True, verbose=1,
        batch_size=128, callbacks=[tb_cb, es_cb], validation_data=[X_test, y_test])
mod.save_weights(serv + "/model/model.h5")
print('done!')
