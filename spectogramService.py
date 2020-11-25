#!/usr/bin/env python
# coding: utf-8
# Tworzenie spectogramów z załadowanych dźwięków
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
from skimage import util
import gc
import glob
datatest = (glob.glob("./ptaki/test/wav/*.wav")) #czytanie plików z lokalizacji
# datatest #pokazuje liste plików
M=1024

for i in range(0, 1000):
    rate, audio = wavfile.read(datatest[i])
    freqs, times, Sx = signal.spectrogram(audio, fs=rate, window='hanning',
                                      nperseg=1024, noverlap=M - 100,
                                      detrend=False, scaling='spectrum')
    f, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
    ax.axis('off')
    f.savefig('./imageptaki/test/'+datatest[i][17:-4]+'.png', bbox_inches='tight', pad_inches=0)
    plt.close('all')
    plt.clf()
    gc.collect()