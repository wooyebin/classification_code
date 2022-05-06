import config
import librosa.display
import numpy as np
#import matplotlib.pyplot as plt

frequency = 2*config.frequency
slice_list = config.slice_list
SR = config.SR
sec = config.sec
sec_window = config.sec_window-1

def make_samples(audio_file_paths, data_file):
    data = open(data_file, "w")
    for audio_file_path in audio_file_paths:
        audio, sr = librosa.load(audio_file_path, sr=SR)
        entire = len(audio)
        for i in range((entire//(sr*sec))):
            sig = audio[i*sr*sec:(i+1)*sr*sec]
            mag = np.abs(np.fft.fft(sig))
            size = len(mag) // sr
            sampling_1 = mag[size*slice_list[0]:size*slice_list[1]:2*size]
            sampling_2 = mag[size*slice_list[2]:size*slice_list[3]:2*size]
            sampling_3 = mag[size*slice_list[4]:size*slice_list[5]:2*size]
            sampling = np.hstack([sampling_1, sampling_2, sampling_3])
            data.write(','.join(str(_) for _ in sampling))
            data.write('\n')
    data.close()
