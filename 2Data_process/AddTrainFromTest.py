import os

start_test = 71
start_train = 455
for i in range(179):
    os.rename("./test/Y/" + "Y_g" + str(start_test).zfill(3) + ".npy", "./train/Y/" + "Y_g" + str(start_train).zfill(3) + ".npy")
    for j in range(25):
        os.rename("./test/input/" + "input_g" + str(start_test).zfill(3) + "_" + str(j).zfill(2) + ".npy", "./train/input/" + "input_g" + str(start_train).zfill(3) + "_" + str(j).zfill(2) + ".npy")
    start_test += 1
    start_train += 1
