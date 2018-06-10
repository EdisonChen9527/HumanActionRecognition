import EModelCosDistance
from numpy import random
import argparse
import numpy as np
import math
import keras.optimizers

parser = argparse.ArgumentParser(description='Get data to train')
parser.add_argument('path_to_train', metavar='base', type=str, help='Path to ACT train data.')
args = parser.parse_args()
path_to_train = args.path_to_train

#模型的基本参数
precondition_index_list = [6,7,8,9]
effect_index_list = [18,19,20,21]
index_list = [6,7,8,9,18,19,20,21]
input_length = len(index_list)
epoch_total = 20
group_count = 454
class_num = 43
learn_rate = 0.01
img_nrows = 118
img_ncols = 118
img_channel = 3

mymodel = EModelCosDistance.EModelCosDistance(precondition_length=len(precondition_index_list), effect_length=len(effect_index_list))
mymodel.summary()


my_optimizer = keras.optimizers.SGD(lr=learn_rate, momentum=0.0, decay=0.0, nesterov=False)
mymodel.compile(loss="categorical_crossentropy", optimizer=my_optimizer, metrics=["categorical_accuracy"])


def create_input(group, precondition_list, effect_list):
    X_list = list()
    Y = np.load(path_to_train + "/Y/" + "Y_g" + str(group).zfill(3) + ".npy")
    for i in range(len(precondition_list)):
        temp = np.load(path_to_train + "/input/" + "input_g" + str(group).zfill(3) + "_" + str(precondition_list[i]).zfill(2) +".npy")
        X_list.append(temp)
    for i in range(len(effect_list)):
        temp = np.load(path_to_train + "/input/" + "input_g" + str(group).zfill(3) + "_" + str(effect_list[i]).zfill(2) +".npy")
        X_list.append(temp)
    return X_list,Y

for i in range(epoch_total):
    shuffle_list = [j for j in range(group_count)]
    random.shuffle(shuffle_list)
    for j in shuffle_list:
        X,Y = create_input(j+1, precondition_index_list, effect_index_list)
        print("Recurrent epoch:",str(i+1),"group_count:",str(j+1))
        mymodel.fit(X,Y,batch_size=16,epochs=1)
    '''
    #按照随机顺序训练样本
    shuffle_list = [j for j in range(group_count*16)]
    random.shuffle(shuffle_list)
    X = list()
    for j in range(input_length):
        X.append(np.zeros((16,img_nrows,img_ncols,img_channel)))
    Y = np.zeros((16,class_num))
    train_flag = 0
    for j in shuffle_list:
        print("group_count:",str(math.floor(j/16) + 1))
        temp_X,temp_Y = create_input(math.floor(j/16) + 1, precondition_index_list, effect_index_list)
        for k in range(input_length):
            X[k][train_flag] = temp_X[k][j%16]
        Y[train_flag] = temp_Y[j%16]
        train_flag += 1
        if train_flag == 16:
            train_flag = 0
            mymodel.fit(X,Y,batch_size=16,epochs=1)
    '''
mymodel.save('./Model/EModelCosDistance.h5')
