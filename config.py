frequency = 1000
slice_list = [0, 100, 100, 200, 450, 550]
num_list = [50, 50, 50]
SR = 9600
sec = 1
sec_window = 5

# AUDIO FILE PATH
audio_file_paths_GG = ["files/GG.wav",
                       "files/GGulBul1.wav",
                       "files/GG1.wav",
                       "files/GGUL_1.wav",
                       "files/GGUL_2.wav",
                       "files/GGUL_3.wav",
                       "files/GGUL_4.wav",
                       "files/GGUL_5.wav",
                       "files/GGUL_6.wav",
                       "files/GGUL_7.wav",
                       "files/GGUL_8.wav",
                       "files/GGUL_9.wav"]
audio_file_paths_MB = ["files/MB.wav",
                       "files/MalBul1.wav",
                       "files/MalBul2.wav",
                       "files/MB1.wav",
                       "files/MB2.wav",
                       "files/MB3.wav"]
audio_file_paths_DG = ["files/DG.wav",
                       "files/DG1.wav",
                       "files/DG2.wav",
                       "files/DG3.wav",
                       "files/DG3.wav",
                       "files/DeungGum1.wav",
                       "files/DeungGum2.wav",
                       "files/DeungGum3.wav",
                       "files/DeungGum4.wav"]

print("[config]-------------------------")
print("%15s : %d" % ("Sampling Rate", SR))
print("%15s : %d" % ("Sampling Sec", sec))
print("%15s : %d~%d, %d~%d, %d~%d" % ("Sampling Freq", slice_list[0], slice_list[1], slice_list[2], slice_list[3], slice_list[4], slice_list[5]))
