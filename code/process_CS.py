# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:11:15 2023

@author: miche
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn import model_selection
import scipy.io
from pydub import AudioSegment
from pydub.utils import make_chunks


# Define directories
base_dir = "E:/Data/DeepESN/"
chunk_dir = 'code/chunked/'
esc_dir = os.path.join(base_dir, "CS_dataset/")
meta_file = os.path.join(esc_dir, "meta/cantiere.csv")
audio_dir = os.path.join(esc_dir, "audio/")


# Load metadata
meta_data = pd.read_csv(meta_file)
meta_data

data_size = meta_data.shape
print(data_size)

class_dict = {}
for i in range(data_size[0]):
    if meta_data.loc[i,"target"] not in class_dict.keys():
        class_dict[meta_data.loc[i,"target"]] = meta_data.loc[i,"category"]
class_pd = pd.DataFrame(list(class_dict.items()), columns=["labels","classes"])
class_pd


# Split file audio into chunks
def split_audio_files(audio_dir, file_name, win):
    file_path = os.path.join(audio_dir, file_name)
    myaudio = AudioSegment.from_file(file_path, "wav")
    chunk_length_ms = win # pydub calculates in millisec
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec
    for i, chunk in enumerate(chunks):
        chunk_name = base_dir + chunk_dir + file_name + "_{0}.wav".format(i)
        print ("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")


# Load a wave data
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=44100)
    return x,fs


# Compute the Mel-STFT
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp


# Data augmentation: add white noise
def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))


# Display wave in plots
def show_wave(x):
    plt.plot(x)
    plt.show()


# Display STFT in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs, x_axis = 'log', y_axis = 'time')
    plt.colorbar()
    plt.show()


# 
def process_audio(file_name, win):
    myaudio = AudioSegment.from_file(file_name, "wav")
    chunk_length_ms = win # pydub calculates in millisec
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec
    for i, chunk in enumerate(chunks):
        chunk_name = base_dir + chunk_dir + file_name + "_{0}.wav".format(i)
        print ("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")


# ======================================================================
# Set here the: 
# - number of frequency bands (freq); 
# - number of time windows (time); and,
# - size of the windows in miliseconds (win).
freq = 128
time = 18
win  = 50

L = 1198  # Minimum number of chunks for each file
# ======================================================================


# Split all files into chunks and return the Mel-Spectrogram
def split_save_np_data_chunked(x, y):
    for i in range(len(y)): # all files in csv
        split_audio_files(audio_dir, x[i], win)

    np_data = np.zeros(L*freq*time*len(x)).reshape(L*len(x), time, freq)
    np_targets = np.zeros(L*len(y)).reshape(-1, 1)
    c = 0
    for i in range(len(y)): # all files in csv
        for j in range(L):
          _x, fs = load_wave_data(base_dir + chunk_dir, x[i]+ "_{0}.wav".format(j))
          _x = calculate_melsp(_x) # evaluate the Spectrogram
          np_data[c] = _x.T #[128x1723]->[freqxtime]
          np_targets[c] = y[i]
          c = c+1
        # show_melsp(np_data[i], fs)
    return [np_data, np_targets]


# Return the Mel-Spectrogram of all data chunks
def save_np_data_chunked(x, y):
    np_data = np.zeros(L*freq*time*len(x)).reshape(L*len(x), time, freq)
    np_targets = np.zeros(L*len(y)).reshape(-1, 1)
    c = 0
    for i in range(len(y)): # all files in csv
        for j in range(L):
          _x, fs = load_wave_data(base_dir + chunk_dir, x[i]+ "_{0}.wav".format(j))
          _x = calculate_melsp(_x) # evaluate the Spectrogram
          np_data[c] = _x.T  # [128x1723]->[freqxtime]
          np_targets[c] = y[i]
          c = c+1
        # show_melsp(np_data[i], fs)
    return [np_data, np_targets]



# %% Pre-process the CS dataset

# Get the dataset file list
x = list(meta_data.loc[:,"filename"])
y = list(meta_data.loc[:, "target"])


# Split files into training and test sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, stratify=y)
print("x train:{0}\ny train:{1}\nx test:{2}\ny test:{3}".format(len(x_train),
                                                                len(y_train),
                                                                len(x_test),
                                                                len(y_test)))


# Split all files into chunks and get the Mel-Spectrogram
[X, Y] = split_save_np_data_chunked(x_train, y_train)
[Xte, Yte] = split_save_np_data_chunked(x_test, y_test)


# Otherwise (if chunks are already available)
# Get the Mel-Spectrogram of all chunks
# [X, Y] = save_np_data_chunked(x_train, y_train)
# [Xte, Yte] = save_np_data_chunked(x_test, y_test)


# Save training and test sets into *.mat format
scipy.io.savemat('cantiere.mat', {'X': X, 'Y':Y, 'Xte':Xte, 'Yte':Yte})
