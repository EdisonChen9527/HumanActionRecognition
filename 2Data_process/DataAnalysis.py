#用于计算视频中不足25帧的比率
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='数据预处理')
parser.add_argument('path_to_python_list', metavar='base', type=str, help='Path to ACT python list.')
args = parser.parse_args()
path_to_python_list = args.path_to_python_list

resample_length = 20
class_num = 43

with open(path_to_python_list, "rb") as fp:
    python_list = pickle.load(fp)

count = np.zeros(class_num)
total_count = np.zeros(class_num)
for i in python_list:
    (temp_address,temp_count,temp_files,temp_class) = i
    if temp_count < resample_length:
        print(temp_address,temp_count,temp_class)
        count[temp_class-1] += 1
    total_count[temp_class-1] += 1
for i in range(class_num):
    print("The class "+ str(i+1) + " count and total_count:",count[i],total_count[i])
