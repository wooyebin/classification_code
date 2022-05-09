import config
import librosa.display
import numpy as np
#import matplotlib.pyplot as plt

frequency = 2*config.frequency
slice_list = config.slice_list
num_list = config.num_list
SR = config.SR
sec = config.sec
sec_window = config.sec_window-1
DROPMIN = config.DROPMIN

def make_samples(audio_file_paths, data_file):
    data = open(data_file, "w")
    for audio_file_path in audio_file_paths:
        audio, sr = librosa.load(audio_file_path, sr=SR)
        entire = len(audio)
        for i in range((entire//(sr*sec))):
            sig = audio[i*sr*sec:(i+1)*sr*sec]
            mag = np.abs(np.fft.fft(sig))
            size = len(mag)*sec // sr
            if DROPMIN == 0:
                sampling_1 = mag[size*slice_list[0]:size*slice_list[1]:size*((slice_list[1]-slice_list[0])//num_list[0])]
                sampling_2 = mag[size*slice_list[2]:size*slice_list[3]:size*((slice_list[3]-slice_list[2])//num_list[1])]
                sampling_3 = mag[size*slice_list[4]:size*slice_list[5]:size*((slice_list[5]-slice_list[4])//num_list[2])]
            else:
                sampling_1 = drop_min(mag[size * slice_list[0]:size * slice_list[1]:size * ((slice_list[1] - slice_list[0]) // num_list[0])])
                sampling_2 = drop_min(mag[size * slice_list[2]:size * slice_list[3]:size * ((slice_list[3] - slice_list[2]) // num_list[1])])
                sampling_3 = drop_min(mag[size * slice_list[4]:size * slice_list[5]:size * ((slice_list[5] - slice_list[4]) // num_list[2])])
            sampling = np.hstack([sampling_1, sampling_2, sampling_3])
            data.write(','.join(str(_) for _ in sampling))
            data.write('\n')
    data.close()

def drop_min(sampling):
    drop_min_sampling = []
    for i in range(len(sampling)//10):
        tmp = sampling[i*10:(i+1)*10]
        tmp = np.delete(tmp, (np.argmin(tmp)))
        tmp = np.delete(tmp, (np.argmin(tmp)))
        tmp = np.delete(tmp, (np.argmin(tmp)))
        drop_min_sampling = np.hstack([drop_min_sampling, tmp])
    return drop_min_sampling