frequency = 1000
slice_list = [0, 100, 100, 150, 450, 550]
num_list = [50, 50, 100]
SR = 9600
sec = 1
sec_window = 5

DROPOUT = 0
DROPMIN = 0

FFT = 1
ML_DATASET = 1
ML_STORE = 1
ML_VIEW = 0
ML_TEST = 0

loop = 100

input_node = sum(num_list)
node_num = 200
hidden_layer = 2
output_node = 3



# AUDIO FILE PATH
audio_file_paths_GG = ["files/GGUL.wav",
                       "files/GGUL2.wav"]
audio_file_paths_MB = ["files/JS.wav"]
audio_file_paths_DG = ["files/DG.wav"]

print("[config]-------------------------")
print("%15s : %d" % ("Sampling Rate", SR))
print("%15s : %d" % ("Sampling Sec", sec))
print("%15s : %d~%d, %d~%d, %d~%d" % ("Sampling Freq", slice_list[0], slice_list[1], slice_list[2], slice_list[3], slice_list[4], slice_list[5]))
