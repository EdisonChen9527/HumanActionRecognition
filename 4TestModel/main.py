from keras.models import load_model
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Get model to test')
parser.add_argument('path_to_Model', metavar='base', type=str, help='Path to Model.')
parser.add_argument('path_to_Test', metavar='base', type=str, help='Path to test data.')
args = parser.parse_args()
path_to_Model = args.path_to_Model
path_to_Test = args.path_to_Test

#模型的基本参数
precondition_index_list = [6,7,8,9]
effect_index_list = [18,19,20,21]
index_list = [6,7,8,9,18,19,20,21]
input_length = len(index_list)
group_count = 249

mymodel = load_model(path_to_Model)
mymodel.summary()

def create_input(group, precondition_list, effect_list):
    X_list = list()
    Y = np.load(path_to_Test + "/Y/" + "Y_g" + str(group).zfill(3) + ".npy")
    for i in range(len(precondition_list)):
        temp = np.load(path_to_Test + "/input/" + "input_g" + str(group).zfill(3) + "_" + str(precondition_list[i]).zfill(2) +".npy")
        X_list.append(temp)
    for i in range(len(effect_list)):
        temp = np.load(path_to_Test + "/input/" + "input_g" + str(group).zfill(3) + "_" + str(effect_list[i]).zfill(2) +".npy")
        X_list.append(temp)
    return X_list,Y

for i in range(group_count):
    X,Y = create_input(i+1, precondition_index_list, effect_index_list)
    print("group_count:",str(i+1))
    print(mymodel.evaluate(X,Y,batch_size=16))
