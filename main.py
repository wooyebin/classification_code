import matplotlib.pyplot as plt

import config
import make_samples
import scikits_learn
import matplotlib.pyplot as plt
import MNN
import numpy as np

FFT = 1
ML = 1

# AUDIO FILE PATH
audio_file_paths_GG = config.audio_file_paths_GG
audio_file_paths_MB = config.audio_file_paths_MB
audio_file_paths_DG = config.audio_file_paths_DG

# DATA FILE
data_file_path_GG = "data/GG.txt"
data_file_path_MB = "data/MB.txt"
data_file_path_DG = "data/DG.txt"
data_file_paths = [data_file_path_GG, data_file_path_MB, data_file_path_DG]

# MAKE SAMPLES ( AUDIO -> DATA )
if FFT == 1:
    make_samples.make_samples(audio_file_paths_GG, data_file_path_GG)
    make_samples.make_samples(audio_file_paths_MB, data_file_path_MB)
    make_samples.make_samples(audio_file_paths_DG, data_file_path_DG)

# MAKE DATA SET
if ML == 1:
    data, label = scikits_learn.make_data_set(data_file_paths, "012")
    #data_GG, label_GG = scikits_learn.make_data_set(data_file_paths, "100")
    #data_MB, label_MB = scikits_learn.make_data_set(data_file_paths, "x10")

# MACHINE LEARNING
f = open("ml list.txt", "w")
if ML == 1:
    loop = 3
    hidden_layer_list = [2, 4]
    node_num_list = [30, 100, 200, 300, 1000]
    avg_list = []
    min_list = []
    max_list = []
    for hidden_layer in hidden_layer_list:
        for node_num in node_num_list:
            accuracy_list = [0 for i in range(loop)]
            print("----------%dx%d----------" % (node_num, hidden_layer))
            print("----------%dx%d----------" % (node_num, hidden_layer), file=f)
            for i in range(loop):
                accuracy = MNN.MNN_keras(data, label, hidden_layer, node_num)
                accuracy_list[i] = accuracy
            for acc in accuracy_list:
                print(acc)
            print("avg : ", end="")
            print("avg : ", end="", file=f)
            print(sum(accuracy_list)/loop)
            print(sum(accuracy_list) / loop, file=f)
            avg_list += [sum(accuracy_list)/loop]
            print("min : ", end="")
            print("min : ", end="", file=f)
            print(min(accuracy_list))
            print(min(accuracy_list), file=f)
            print("max : ", end="")
            print("max : ", end="", file=f)
            print(max(accuracy_list))
            print(max(accuracy_list), file=f)
            min_list += [min(accuracy_list)]
            max_list += [max(accuracy_list)]

    l_n = ["30x2", "100x2", "200x2", "300x2", "1000x2",
           "30x4", "100x4", "200x4", "300x4", "1000x4"]
    y, min, max = avg_list, min_list, max_list
    for i in range(len(y) - 1, 0, -1):
        for j in range(i):
            if y[j] > y[j + 1]:
                y[j], y[j + 1] = y[j + 1], y[j]
                min[j], min[j + 1] = min[j + 1], min[j]
                max[j], max[j + 1] = max[j + 1], max[j]
                l_n[j], l_n[j + 1] = l_n[j + 1], l_n[j]
    t = open("sort.txt", "w")
    print(l_n, y, max, min, file=t)
    t.close()
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    min_yerr = []
    max_yerr = []
    for i in range(10):
        min_yerr += [y[i] - min[i]]
        max_yerr += [max[i] - y[i]]
    plt.errorbar(x, y, yerr=[min_yerr, max_yerr])
    bar = plt.scatter(x, y, marker="h")
    for i in range(10):
        plt.text(x[i], min[i], str(min[i] * 1000 // 1 / 1000), fontsize=7)
        plt.text(x[i], max[i], str(max[i] * 1000 // 1 / 1000), fontsize=7)
        plt.text(x[i], y[i], str(y[i] * 1000 // 1 / 1000), fontsize=8)
        plt.text(x[i], 0.68, l_n[i], fontsize=7)
    plt.show()
    f.close()
    #scikits_learn.scikits_learn(data, label)
    #print("[labeling]-----------------------\n0 : GG\n1 : MB\n2 : DG")
    #scikits_learn.scikits_learn(data, label)
    #print()
    #print("[labeling]-----------------------\n1 : GG\n0 : MB & DG")
    #scikits_learn.scikits_learn(data_GG, label_GG)
    #print()
    #print("[labeling]-----------------------\n1 : MB\n0 : DG")
    #scikits_learn.scikits_learn(data_MB, label_MB)
    #MNN.MNN(data_OX, label_OX, data_GG, label_GG)




