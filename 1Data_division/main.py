import argparse
import pickle
import os

parser = argparse.ArgumentParser(description='数据划分')
parser.add_argument('path_to_ACT', metavar='base', type=str, help='Path to ACT dataset.')
parser.add_argument('data_record', metavar='base', type=str, help='Path to ACT dataset record.')
args = parser.parse_args()
path_to_ACT = args.path_to_ACT
data_record = args.data_record

lines = open(data_record,"r").readlines()
python_list = list()
for line in lines:
    temp_address = line.split(" ")[0]
    temp_class = int(line.split(" ")[1])
    temp_files = None
    temp_count = 0
    for root,dirs,files in os.walk(path_to_ACT + temp_address):
        temp_files = files
        temp_count = len(temp_files)
        break
    python_list.append((temp_address,temp_count,temp_files,temp_class))

if "train" in data_record:
    with open("train_list.txt", "wb") as fp:
        pickle.dump(python_list, fp)
else:
    with open("test_list.txt", "wb") as fp:
        pickle.dump(python_list, fp)
