#对输入数据进行resample，在resample后对其进行矩阵化
import pickle
import argparse
import numpy as np
import math
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array
import time
from numpy import random

parser = argparse.ArgumentParser(description='数据resample与矩阵化')
parser.add_argument('path_to_ACT', metavar='base', type=str, help='Path to ACT dataset.')
parser.add_argument('path_to_python_list', metavar='base', type=str, help='Path to ACT python list.')
parser.add_argument('path_to_store_matrix', metavar='base', type=str, help='Path to store output matrix.')
args = parser.parse_args()
path_to_ACT = args.path_to_ACT
path_to_python_list = args.path_to_python_list
path_to_store_matrix = args.path_to_store_matrix
resample_length = 25
class_num = 43
img_nrows,img_ncols,img_channel = 118,118,3

with open(path_to_python_list, "rb") as fp:
    python_list = pickle.load(fp)

#对帧长度不足resample_length的视频进行扩充
for i in range(len(python_list)):
    (temp_address,temp_count,temp_files,temp_class) = python_list[i]
    temp_files.sort()
    if temp_count < resample_length:
        step = int(math.ceil(resample_length/temp_count))
        new_temp_files = list()
        for j in range(len(temp_files)):
            for k in range(step):
                new_temp_files.append(temp_files[j])
        python_list[i] = (temp_address,temp_count*step,new_temp_files,temp_class)
        '''
        print("CHange: ",temp_address,temp_count,len(temp_files),temp_class)
        print("After cHange: ",temp_address,temp_count*step,len(new_temp_files),temp_class)
        '''
'''
#用于测试扩充是否正确
for i in range(len(python_list)):
    (temp_address,temp_count,temp_files,temp_class) = python_list[i]
    if i%100 == 0:
        print(temp_address,temp_count,len(temp_files),temp_class)
    if temp_count < resample_length:
        print("Error!")
        print(temp_address,temp_count,len(temp_files),temp_class)
'''

#对每个视频内的帧文件根据时序进行排序
for i in range(len(python_list)):
    (temp_address,temp_count,temp_files,temp_class) = python_list[i]
    temp_files.sort()

#对视频进行固定长度的重采样
def resample(input_list,result_length):
    output_list = list()
    step = len(input_list) / result_length
    for i in range(result_length):
        output_list.append(input_list[math.floor( i * step + 0.5 )])
    output_list.sort()
    return output_list

for i in range(len(python_list)):
    (temp_address,temp_count,temp_files,temp_class) = python_list[i]
    new_temp_files = resample(temp_files,resample_length)
    python_list[i] = (temp_address,temp_count,new_temp_files,temp_class)

'''
#用于验证采样结果的代码
for i in range(len(python_list)):
    if i < 32:
        (temp_address,temp_count,temp_files,temp_class) = python_list[i]
        print(temp_address,temp_count,len(temp_files),temp_class)
#用于验证采样结果的代码
'''

#将图片转换为矩阵
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def create_inputs(matrix_list):
    m = len(matrix_list)
    X_list = list()
    for i in range(resample_length):
        X_list.append(np.zeros((m,img_nrows,img_ncols,img_channel)))
    Y = np.zeros((m,class_num))
    for i in range(m):
        (temp_address,temp_count,temp_files,temp_class) = matrix_list[i]
        Y[i,temp_class-1] = 1
        for j in range(resample_length):
            X_list[j][i] = preprocess_image(path_to_ACT + temp_address + temp_files[j])[0]
    return X_list,Y

count_group = 1
temp_python_list = list()
#将数据打乱
random.shuffle(python_list)
for i in range(len(python_list)):
    temp_python_list.append(python_list[i])
    if i == len(python_list) - 1:
        matrix_inputs,matrix_Y = create_inputs(temp_python_list)
        for j in range(len(matrix_inputs)):
            matrix_input = matrix_inputs[j]
            np.save(path_to_store_matrix + "/input/input_g" + str(count_group).zfill(3) + "_" + str(j).zfill(2), matrix_input)
            print("input_g" + str(count_group).zfill(3) + "_" + str(j).zfill(2), matrix_input.shape)
        np.save(path_to_store_matrix + "/Y/Y_g" + str(count_group).zfill(3), matrix_Y)
        print("Y_g" + str(count_group).zfill(3),matrix_Y.shape)
        count_group += 1
    else:
        if len(temp_python_list) == 16:
            matrix_inputs,matrix_Y = create_inputs(temp_python_list)
            for j in range(len(matrix_inputs)):
                matrix_input = matrix_inputs[j]
                np.save(path_to_store_matrix + "/input/input_g" + str(count_group).zfill(3) + "_" + str(j).zfill(2), matrix_input)
                print("input_g" + str(count_group).zfill(3) + "_" + str(j).zfill(2), matrix_input.shape)
            np.save(path_to_store_matrix + "/Y/Y_g" + str(count_group).zfill(3), matrix_Y)
            print("Y_g" + str(count_group).zfill(3),matrix_Y.shape)
            '''
            #测试生成的Y矩阵是否正确
            print("-----------------------")
            print("count_group:",count_group)
            print(matrix_Y)
            print("-----------------------")
            #测试生成的Y矩阵是否正确
            '''
            count_group += 1
            '''
            #测试每次写入的temp_python_list
            print("-----------------------")
            for j in range(len(temp_python_list)):
                (temp_address,temp_count,temp_files,temp_class) = temp_python_list[j]
                print(temp_address,temp_count,len(temp_files),temp_class)
            print("-----------------------")
            #测试每次写入的temp_python_list
            '''
            temp_python_list = list()
