# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:51:37 2022

@author: Shang FENG
"""

import mne
import numpy as np
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
import matplotlib.pyplot as plt

hz_list = [2, 3, 4, 12, 16]

len_slice = 30 * 250  # sec * samplerate
num_hz = len(hz_list)

#%%
raw_data = np.loadtxt('D:\\3 DATA\\202208数眠M2测试数据\\9.13晚\\2022091426120001\\2022091426120001.txt',
                      dtype=np.floating, delimiter=',')

num_slice = int(len(raw_data) / len_slice)
max_len = num_slice * len_slice

filter_data = mne.filter.filter_data(raw_data[:max_len], 250, 1, 20, verbose=0)

sliced_data = filter_data.reshape(num_slice, len_slice)

PSD_table = np.zeros(shape=(num_slice, num_hz))

for ii in range(num_slice):

    current_data = sliced_data[ii, :]

    psd, freqs = psd_array_multitaper(current_data, 250, adaptive=True,
                                      normalization='full', verbose=0)

    freq_res = freqs[1] - freqs[0]

    for jj in range(num_hz):

        hz = hz_list[jj]

        idx_band = np.logical_and(freqs >= hz-0.5, freqs <= hz+0.5)
        band_power = simps(psd[idx_band], dx=freq_res)  # simps抛物线近似

        PSD_table[ii, jj] = band_power

plt.plot(PSD_table[:,4])
# plt.plot(filter_data)
# plt.hist(PSD_table[:,4])
