import os
import argparse
import numpy as np
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from scipy.misc import imsave

parser = argparse.ArgumentParser(description='Get data from matrix')
parser.add_argument('path_to_Matrix', metavar='base', type=str, help='Path to Matrix.')
parser.add_argument('group_of_Matrix', metavar='base', type=str, help='group of the Matrix.')
args = parser.parse_args()
path_to_Matrix = args.path_to_Matrix
group_of_Matrix = args.group_of_Matrix

resample_length = 25
class_num = 43
img_nrows = 118
img_ncols = 118
img_channel = 3
npy_matrixs = list()
for i in range(resample_length):
    npy_matrixs.append(np.load(path_to_Matrix + "/input/input_g" + group_of_Matrix.zfill(3) + "_" + str(i).zfill(2) + ".npy"))
npy_Y = np.load(path_to_Matrix + "/Y/Y_g" + group_of_Matrix.zfill(3) +".npy")
m,_ = npy_Y.shape
for i in range(m):
    for j in range(class_num):
        if npy_Y[i,j] == 1:
            print("The example",str(i+1),"belong to class",str(j+1))

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

for i in range(m):
    for j in range(len(npy_matrixs)):
        img = deprocess_image(npy_matrixs[j][i])
        fname = './picture/example_' + str(i+1) + "_" + str(j).zfill(2) + '.png'
        imsave(fname, img)
