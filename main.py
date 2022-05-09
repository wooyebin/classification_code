import matplotlib.pyplot as plt

import config
import make_samples
import scikits_learn
import matplotlib.pyplot as plt
import MNN
import time
import numpy as np
import tensorflow as tf

FFT = 1
ML = 1
ML_STORE = 0
ML_VIEW = 1
ML_TEST = 0

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
    print(len(data[0]))
    #data_GG, label_GG = scikits_learn.make_data_set(data_file_paths, "100")
    #data_MB, label_MB = scikits_learn.make_data_set(data_file_paths, "x10")

if ML_STORE == 1:
    hidden_layer = 2
    node_num = 100
    accuracy = MNN.MNN_keras(data, label, hidden_layer, node_num)
    print(accuracy)


# MACHINE LEARNING RESULT TEST VIEW
f = open("ml list.txt", "w")
if ML_VIEW == 1:
    loop = 1
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
    plt.xticks(x, l_n)
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
        #plt.text(x[i], 1.1, l_n[i], fontsize=7)
    slice_list = config.slice_list
    num_list = config.num_list
    sec = config.sec
    plt.title("%ds %d~%d(%d), %d~%d(%d), %d~%d(%d) d%d m%d" % (sec, slice_list[0], slice_list[1], num_list[0], slice_list[2], slice_list[3], num_list[1], slice_list[4], slice_list[5], num_list[2], config.DROPOUT, config.DROPMIN))
    plt.savefig("%ds %d~%d(%d), %d~%d(%d), %d~%d(%d) d%d m%d.png" % (sec, slice_list[0], slice_list[1], num_list[0], slice_list[2], slice_list[3], num_list[1], slice_list[4], slice_list[5], num_list[2], config.DROPOUT, config.DROPMIN))
    f.close()

if ML_TEST == 1:
    model = tf.keras.models.load_model('beekeeping.h5')
    xhat = np.array([[0.11326716199318626,0.1907574974185933,0.2475400844854006,0.41042100979627005,0.29868986667127423,1.1029056543608229,2.4601690607017908,1.0718922698637356,1.05197530163604,0.6381436664481499,1.3982089889157283,1.0104356354219974,2.157312681804377,2.16048625170441,0.7153075859118208,0.3812814965439345,2.0235572274003264,3.253876973200956,9.285393627709736,1.5857843538139191,1.674517082689143,0.8859246275303645,0.20751807397385896,0.7869564524966645,0.7906388932420303,0.3553770435114375,0.73948309855865,0.884691680727144,0.19137270844819435,1.3545676215158349,1.8103402293027708,0.16584029542048323,0.7121514057185007,0.8564178781245478,1.016167232812416,2.989856434107035,1.0753208736477287,1.8438158406180993,1.3069220292012054,1.22069381232584,0.5094635365962317,1.5823195881470749,0.868114954831266,1.0076618769144152,1.1329700427875091,0.9199164576734432,2.12354313276014,0.28539122085119556,0.693716248897593,0.18908259616950657,2.468530112386929,1.5048773794509773,0.9827584218886192,0.4657771818914664,1.392315769916049,1.368838983578638,0.23659264998725568,0.5235298128501125,0.1513365985737372,0.19137975705145593,0.4143207839196595,1.3866054959939356,0.5826739243512483,1.0898680816113684,0.25944925973840355,0.628854414801056,0.037902138419651094,0.21816149093345086,0.6333350266921278,0.344758609772222,0.7047464476212015,0.6742062030358752,0.2695388322210946,0.04387709248178563,0.9420559207991163,0.48015430654283514,0.7863705782941086,0.4654094531381593,0.7216468562543312,0.3880635444453792,0.17469922945324654,0.0769690896487891,0.24448779644716756,0.587398621513649,0.6234970993566822,0.7394121060746879,0.47197189338434115,1.1970829006303338,0.7014172058073708,0.23511650449680024,0.08805741606461527,1.1038785998570229,1.068439528920253,1.2462393852681324,0.8154320960422479,0.5528070999110437,0.9282813767540747,0.7418516433751318,1.4769985662757827,0.505677518242794,1.8409883429471774,1.071921358947418,0.8861669329464642,0.26199170936515287,0.7529542423029997,0.5347917476110453,0.6915413728225597,0.27179371519557655,0.4493813674882061,0.17880237232445198,0.18868427789757758,0.41225798390438917,0.21089510751201868,0.41391455008349193,0.11904674002001459,0.13250572123384044,0.26202452909979546,0.18298942491413603,0.24775825607574167,0.37407676490413466,0.655543027540311,0.10078751149186478,0.11055942412885358,0.12436830243687615,0.2103132094810163,0.23345215704296277,0.24001117865185378,0.2070168190203069,0.06553088324335732,0.17879265513706175,0.18134806319147595,0.11345634845715345,0.2994254196827655,0.31075354470891603,0.16690668395398062,0.23535102004688443,0.23020303880972642,0.28251452445357267,0.09714295640285851,0.2688207913217532,0.155335157500463,0.16234914149471757,0.2530166598185815,0.10968193753000195,0.12577688641196955,0.0619051796066216,0.23577075722521323,0.333698482646465,0.19963087778535307,0.10066182529616494]])
    yhat = model.predict(xhat)
    print(np.argmax(yhat, axis=1))
