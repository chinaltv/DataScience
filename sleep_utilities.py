# 工作清单
# 3 单独函数（batch_figure）制定输出原始数据图、处理数据图并进行坏段标记（没什么好思路，暂时鸽）  
# 4 特征提取（batch_feature_extraction，基于generate_data0825.py），将研究算法并将该内容全部把控成自己的东西。暂时摘抄  

# 上述是自己写的函数调用的包

import numpy as np
import pandas as pd
import os
import mne
from mne.time_frequency import psd_array_welch

## 下述是刘方轩调用的包

import math
import os
from pyinform.dist import Dist
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import scipy.interpolate as spi
from scipy import stats  # scipy中的stats可以做统计推断
from pyinform.shannon import entropy



# 该函数对NE的单通道edf数据进行处理。背后逻辑同数眠的art_clean。

def art_clean_NE_single(path, slice_interval = 1, slice_disposal_ratio = 10, art_disposal_percentile = 10, s_rate = 250, l_freq = 1, h_freq = 20):
    
    raw = mne.io.read_raw_edf(path, preload = True, verbose = 0) # 读取文件
    raw = raw.filter(l_freq = l_freq, h_freq = h_freq, verbose = 0) # 1-20Hz，FIR方法滤波
    raw = raw.get_data()[0]

    def sum_of_squares(lst):  # 该函数返回长切片列表中每个元素的平方和，要求其为reshape后的列表
        return sum([x **2 for x in lst])

    def art_max_pos(ss, cut_number): # 该函数输入切片功率列表ss，并返回切片对象伪迹的相对位置（排序，确定前10%）
        maxlist = sorted(ss, reverse = True)[:cut_number]  # 创建一个基于自定义数字的最大值列表
        ss_dict = {}  # 创建一个空字典
        for i, b in enumerate(ss):
            ss_dict[b] = i
        max_indices = [ss_dict[a] for a in maxlist]
        return max_indices

    def art_return_ss(raw_sliced, cut_interval, num_sliced): # 该函数在正好整除情况下，输出切片片段的平方和列表
        list_of_ss = []
        i = 0
        for x in raw_sliced:
            list_of_ss.append(sum_of_squares(raw_sliced[i]))
            i += 1
        return list_of_ss
        
    def speclen_art_return_ss(raw_with_tail, num_slice, remainder, cut_inverval): # 在无法整除情况下，返回有尾切片平方和，列表的最后一个元素是尾端元素的平方和
        length = len(raw_with_tail)
        exp_len = length - remainder
        neat_slice = raw_with_tail[:exp_len]
        neat_slice = neat_slice.reshape(num_slice, cut_interval) 
        list_of_ss = []
        i = 0
        for x in neat_slice:
            list_of_ss.append(sum_of_squares(neat_slice[i]))
            i += 1
        extra_ss = sum(x ** 2 for x in raw_with_tail[exp_len + 1:])
        extra_slice = raw_with_tail[exp_len + 1:]
        list_of_ss.append(extra_ss)
        return list_of_ss, neat_slice, extra_slice

    # 定义局部变量
    cut_interval = s_rate * slice_interval # 切割间距
    remainder = int(len(raw)) % cut_interval  # 这个数为剩余秒数*采样率，不是单纯秒数
    num_sliced = int(len(raw) / cut_interval) # 完整切割片数，向下取整

    # 逻辑判断长度：平方和
    if remainder == 0: # 正好整除
        raw_sliced = raw.reshape(num_sliced, cut_interval)
        ss = art_return_ss(raw_sliced, cut_interval, num_sliced) # 切片片段的平方和列表
    else:
        if remainder < slice_disposal_ratio / 100 * int(len(raw)):  # 丢弃
            ss, neat_slice, extra_slice = speclen_art_return_ss(raw, num_sliced, remainder, cut_interval)
            ss.pop()
        elif remainder >= slice_disposal_ratio / 100 * int(len(raw)):  # 保留并缩放
            ss, neat_slice, extra_slice = speclen_art_return_ss(raw, num_sliced, remainder, cut_interval)
            ss[len(ss)-1] = ss[len(ss)-1] / remainder * cut_interval 

    # 定义另一块局部变量
    cut_number = int(len(ss)/ art_disposal_percentile) # 伪迹丢弃切割长度
    art_indices = art_max_pos(ss, cut_number) # 借用上面做的函数，输出最大值index列表

    # 逻辑判断长度：切片组合
    if remainder == 0: # 正好整除
        cleaned_sliced = [n for i, n in enumerate(raw_sliced) if i not in art_indices]
    else:
        if remainder < slice_disposal_ratio / 100 * int(len(raw)):  # 丢弃，用neat_slice挑选和art_indices挑选位置，此时和remainder == 0一样，只是数据经过处理。（speclen_art函数没有逻辑判断，对于任意长度尾端的，均会返回list_of_ss和两段slices）
            cleaned_sliced = [n for i, n in enumerate(neat_slice) if i not in art_indices]
        elif remainder >= slice_disposal_ratio / 100 * int(len(raw)):
            if len(ss) - 1 not in art_indices:   # 保留，伪迹正常
                cleaned_sliced = [n for i, n in enumerate(neat_slice) if i not in art_indices] + extra_slice
            elif len(ss) - 1 in art_indices:    # 保留，伪迹异常
                art_indices.remove(int(len(ss)-1))
                cleaned_sliced = [n for i, n in enumerate(neat_slice) if i not in art_indices]

    return np.ravel(cleaned_sliced, order = 'C').tolist()




# 该函数大概已稳定。该函数实现任意长度的单文件单通道txt文件脑电预处理，包括滤波和伪迹处理。

def art_clean(raw, slice_interval = 1, slice_disposal_ratio = 10, art_disposal_percentile = 10, s_rate = 250, l_freq = 1, h_freq = 20):  # 该函数将对text_raw对象进行预处理，并返回一个数据列表

    raw = mne.filter.filter_data(data = raw, sfreq = s_rate, l_freq = l_freq, h_freq = h_freq, verbose = 0) # 1-20Hz的FIR滤波。因为只选取了1-20Hz，50Hz的工频处理就不需要啦。另外由于是1s切片（蓝牙问题），不用特地设置数据长度（一定能被250整除）。verbose参数决定了不生成任何信息（以免太吵）。
    
    def sum_of_squares(lst):  # 该函数返回长切片列表中每个元素的平方和，要求其为reshape后的列表
        return sum([x **2 for x in lst])

    def art_max_pos(ss, cut_number): # 该函数输入切片功率列表ss，并返回切片对象伪迹的相对位置（排序，确定前10%）
        maxlist = sorted(ss, reverse = True)[:cut_number]  # 创建一个基于自定义数字的最大值列表
        ss_dict = {}  # 创建一个空字典
        for i, b in enumerate(ss):
            ss_dict[b] = i
        max_indices = [ss_dict[a] for a in maxlist]
        return max_indices

    def art_return_ss(raw_sliced, cut_interval, num_sliced): # 该函数在正好整除情况下，输出切片片段的平方和列表
        list_of_ss = []
        i = 0
        for x in raw_sliced:
            list_of_ss.append(sum_of_squares(raw_sliced[i]))
            i += 1
        return list_of_ss
        
    def speclen_art_return_ss(raw_with_tail, num_slice, remainder, cut_inverval): # 在无法整除情况下，返回有尾切片平方和，列表的最后一个元素是尾端元素的平方和
        length = len(raw_with_tail)
        exp_len = length - remainder
        neat_slice = raw_with_tail[:exp_len]
        neat_slice = neat_slice.reshape(num_slice, cut_interval) 
        list_of_ss = []
        i = 0
        for x in neat_slice:
            list_of_ss.append(sum_of_squares(neat_slice[i]))
            i += 1
        extra_ss = sum(x ** 2 for x in raw_with_tail[exp_len + 1:])
        extra_slice = raw_with_tail[exp_len + 1:]
        list_of_ss.append(extra_ss)
        return list_of_ss, neat_slice, extra_slice

    # 定义局部变量
    cut_interval = s_rate * slice_interval # 切割间距
    remainder = int(len(raw)) % cut_interval  # 这个数为剩余秒数*采样率，不是单纯秒数
    num_sliced = int(len(raw) / cut_interval) # 完整切割片数，向下取整

    # 逻辑判断长度：平方和
    if remainder == 0: # 正好整除
        raw_sliced = raw.reshape(num_sliced, cut_interval)
        ss = art_return_ss(raw_sliced, cut_interval, num_sliced) # 切片片段的平方和列表
    else:
        if remainder < slice_disposal_ratio / 100 * int(len(raw)):  # 丢弃
            ss, neat_slice, extra_slice = speclen_art_return_ss(raw, num_sliced, remainder, cut_interval)
            ss.pop()
        elif remainder >= slice_disposal_ratio / 100 * int(len(raw)):  # 保留并缩放
            ss, neat_slice, extra_slice = speclen_art_return_ss(raw, num_sliced, remainder, cut_interval)
            ss[len(ss)-1] = ss[len(ss)-1] / remainder * cut_interval 

    # 定义另一块局部变量
    cut_number = int(len(ss)/ art_disposal_percentile) # 伪迹丢弃切割长度
    art_indices = art_max_pos(ss, cut_number) # 借用上面做的函数，输出最大值index列表

    # 逻辑判断长度：切片组合
    if remainder == 0: # 正好整除
        cleaned_sliced = [n for i, n in enumerate(raw_sliced) if i not in art_indices]
    else:
        if remainder < slice_disposal_ratio / 100 * int(len(raw)):  # 丢弃，用neat_slice挑选和art_indices挑选位置，此时和remainder == 0一样，只是数据经过处理。（speclen_art函数没有逻辑判断，对于任意长度尾端的，均会返回list_of_ss和两段slices）
            cleaned_sliced = [n for i, n in enumerate(neat_slice) if i not in art_indices]
        elif remainder >= slice_disposal_ratio / 100 * int(len(raw)):
            if len(ss) - 1 not in art_indices:   # 保留，伪迹正常
                cleaned_sliced = [n for i, n in enumerate(neat_slice) if i not in art_indices] + extra_slice
            elif len(ss) - 1 in art_indices:    # 保留，伪迹异常
                art_indices.remove(int(len(ss)-1))
                cleaned_sliced = [n for i, n in enumerate(neat_slice) if i not in art_indices]

    return np.ravel(cleaned_sliced, order = 'C').tolist()




# 该函数大概已稳定。将遍历指定目录，返回filepath下全部文件名，对raw对象进行art_clean处理并返回txt对象。本函数依赖于art_clean并且只可处理数眠数据（txt格式）。

def batch_art_clean(input_path, output_path, slice_interval = 1, slice_disposal_ratio = 10, art_disposal_percentile = 10, s_rate = 250, l_freq = 1, h_freq = 20): 
    FileName_List = os.listdir(input_path) # 返回文件名列表
    for FileName in FileName_List: # 对于所有文件名
        InputPaths = os.path.join('%s/%s' % (input_path, FileName)) # 拼接路径
        OutputPaths = os.path.join('%s/%s' % (output_path, FileName)) 
        if os.path.isfile(InputPaths): # 判断是否为文件还是文件夹。
            if InputPaths.endswith('.txt'): # 判断文件扩展名是否为指定格式
                raw = np.loadtxt(InputPaths, dtype = np.floating, delimiter = ',') # 读取文件
                result = art_clean(raw, slice_interval = slice_interval, slice_disposal_ratio = slice_disposal_ratio, art_disposal_percentile = art_disposal_percentile, s_rate = s_rate, l_freq = l_freq, h_freq = h_freq)
                np.savetxt(OutputPaths, result)




# 该函数在有限使用中已稳定。该函数在指定原始数据raw，切片长度（秒数）参数、采样率参数、傅里叶变换参数、指定频段band_power（可用列表形式或文本），将输出对应频段切片的总功率，并作为字典输出。尾端数据（与切片长度有关）将被正常算出，因此视需求而定自行观察抛弃列表尾端的数据。

def band_return_ss(raw, slice_interval = 2, s_rate = 250, band_power = ["alpha", "beta", "gamma", "delta", "theta"], n_fft = 256): # 不滤波
    # slice_disposal_ratio还没用上，现在逻辑暂时只是不抛弃，算片段功率，尾端数据正常算，不缩放（请自行判断）

    def is_number(s): # 网上抄的
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    # 前置设置和数据导入
    raw = np.array(raw)  # 原数据txt读进来是列表，要转数组，要不然mne包会报错
    bandfreq4output = []
    band4output = []
    freq_band = {
        "delta": [0.5, 4.5], 
        "theta": [4.5, 8.5], 
        "alpha": [8.5, 11.5], 
        "sigma": [11.5, 15.5], 
        "beta": [15.5, 30]
        }

    # 参数判断
    if type(band_power) != list:  # band_power参数输错报错
        print ("Your band_power parameter is not a list. Please pass [fmin, fmax] or a string list like ['alpha', 'beta' ...].") # return

    else: # 正式函数
        if len(band_power) == 2 and is_number(band_power[0]) and is_number(band_power[1]): # band_power应该是一个列表。当列表等于2个的时候并且均为整数的时候，说明这是一个频段。后续应该加入报错命令。
            type_para = 1
            bandfreq4output = [band_power]
            band4output.append("Your set")
        else:
            type_para = 0
            for types in band_power:
                if types in freq_band:
                    bandfreq4output.append(freq_band[types]) # 调用字典内部文件，band4output会是一个复合列表。
                    band4output.append(types)

        
        # 正式输出函数，此时raw是一个数组而不是其他函数内的列表
        cut_interval = s_rate * slice_interval
        remainder = int(len(raw)) % cut_interval 
        remainder = int(len(raw)) % cut_interval  
        num_sliced = int(len(raw) / cut_interval)

        length = len(raw)
        exp_len = length - remainder
        neat_slice = raw[:exp_len]
        neat_slice = neat_slice.reshape(num_sliced, cut_interval)  # neat_slice是一个复合列表，内藏n个切片
        extra_slice = raw[exp_len + 1:] # 尾端数据将抛弃
        
        dict_of_ss = {}
        list_of_ss = []
        i = 0

        for band in bandfreq4output: # 每个频段的循环
            for slices in neat_slice: # 处理干净切片
                psd, freq = psd_array_welch(slices, sfreq = s_rate, fmin = band[0], fmax = band[1], n_fft = n_fft, verbose = 0)
                list_of_ss.append(sum(psd))
            if len(extra_slice) != 0: # 处理额外切片，请视情况删除
                psd_extra, freq_extra = psd_array_welch(extra_slice, sfreq = s_rate, fmin = band[0], fmax = band[1], n_fft = n_fft)
                list_of_ss.append(sum(psd_extra))
            if type_para == 1: # 切片后加入
                dict_of_ss[band4output[i]] = list_of_ss
                break
            elif type_para == 0:
                dict_of_ss[band4output[i]] = list_of_ss
                list_of_ss = []
                i += 1

    return dict_of_ss




# 该函数实现任意长度的单文件单通道txt文件脑电功率输出，可指定任意频段，并以字典形式输出在每个文件里（其实还是不直观，后续再看看怎么处理吧）。

def batch_band_ss(input_path, output_path, slice_interval = 2, s_rate = 250, band_power = ["alpha", "beta", "gamma", "delta", "theta"], n_fft = 256): # 该函数将遍历指定目录，返回filepath下全部文件名，对raw对象进行band_return_ss处理并存入txt对象。本函数依赖于band_return_ss并且只可处理数眠数据（txt格式）。
    FileName_List = os.listdir(input_path) # 返回文件名列表
    for FileName in FileName_List: # 对于所有文件名
        InputPaths = os.path.join('%s/%s' % (input_path, FileName)) # 拼接路径
        OutputPaths = os.path.join('%s/%s' % (output_path, FileName)) 
        if os.path.isfile(InputPaths): # 判断是否为文件还是文件夹。
            if InputPaths.endswith('.txt'): # 判断文件扩展名是否为指定格式
                raw = np.loadtxt(InputPaths, dtype = np.floating, delimiter = ',') # 读取文件
                result = band_return_ss(raw, slice_interval = 2, s_rate = 250, band_power = ["alpha", "beta", "gamma", "delta", "theta"], n_fft = 256)
                np.savetxt(OutputPaths, result)




# 还没动工，暂时摘抄

def feature_extraction(input_path):
    return input_path













# 下述所有函数由刘方轩开发，实现matlab功能。有一部分与上述函数重复。

## 该函数实现带通+陷波滤波

def bandpass_notch(raw_data):
    from scipy import signal
    import scipy.interpolate as spi

    raw_data_arr = np.array(raw_data, dtype=float)
    # 带通滤波：
    ab1 = signal.butter(N=6, Wn=[0.008, 0.288], btype='bandpass')
    b1, a1 = ab1[0], ab1[1]
    rawdata_filtered1 = signal.filtfilt(b1, a1, raw_data_arr)

    # 陷波滤波
    ab2 = signal.butter(N=6, Wn=[0.392, 0.408], btype='bandstop')
    b2, a2 = ab2[0], ab2[1]
    rawdata_filtered2 = signal.filtfilt(b2, a2, rawdata_filtered1)

    return rawdata_filtered2

## 该函数实现60个指定特征提取

def generate_features(data_list):
    # 先1~36Hz滤波再49~51陷波滤波得到的数组
    notch_filter = bandpass_notch(data_list)

    # 经过滤波处理后的信号最大值
    max_filter = np.max(notch_filter)
    # 经过滤波处理后的信号最小值
    min_filter = np.min(notch_filter)


    # 峰度
    kurtosis = stats.kurtosis(notch_filter)  # 求峰度 notch_filter
    # 2022-07-27 新增
    w = signal.get_window('hamming', 125, fftbins=False)
    F, T, S = signal.stft(np.array(notch_filter), 250, w, nperseg=125, noverlap=100)   #0922修改为滤波后再做STFT

    Mag = abs(np.transpose(S))
    seg = np.size(Mag, 1)  # matlab中索引是从1开始的，故此处将2改为1
    xx = np.arange(0, 30.01, 0.01)
    n_xx = len(xx)

    loc1 = int(0.06 * 100)
    loc2 = int(0.1 * 100)
    loc3 = int(0.3 * 100)
    loc4 = int(0.5 * 100)
    loc5 = int(1 * 100)
    loc6 = int(1.6 * 100)
    loc7 = int(2 * 100)
    loc8 = int(3 * 100)
    loc9 = int(4 * 100)
    loc10 = int(4.5 * 100)
    loc11 = int(7 * 100)
    loc12 = int(8 * 100)
    loc13 = int(11 * 100)
    loc14 = int(13 * 100)
    loc15 = int(15 * 100)
    loc16 = int(16 * 100)
    loc17 = int(30 * 100)


    yy = np.zeros((n_xx, seg))

    for i in range(seg):
        y = Mag[:, i]
        ipo3 = spi.splrep(T, y, k=3)  # 生成模型参数
        PSD = spi.splev(xx, ipo3)  # 生成插值点
        # print("PSD:",PSD)
        # print("PSD shape:",PSD.shape)
        yy[:, i] = PSD
        # print("yy:",yy)
        # print("yy shape:",yy.shape)

    # 计算0.06~0.1Hz的PSD
    PSD_bands1 = np.sum(yy[loc1:loc2], axis=0)
    # 计算0.1~0.3Hz的PSD
    PSD_bands2 = np.sum(yy[loc2:loc3], axis=0)
    # 计算0.3~0.5Hz的PSD
    PSD_bands3 = np.sum(yy[loc3:loc4], axis=0)
    # 计算0.5~1Hz的PSD
    PSD_bands4 = np.sum(yy[loc4:loc5], axis=0)
    # 计算0.5~2Hz的PSD
    PSD_bands5 = np.sum(yy[loc4:loc7], axis=0)
    # 计算1.6~4Hz的PSD
    PSD_bands6 = np.sum(yy[loc6:loc9], axis=0)
    # 计算3~4.5Hz的PSD
    PSD_bands7 = np.sum(yy[loc8:loc10], axis=0)
    # 计算4~7Hz的PSD
    PSD_bands8 = np.sum(yy[loc9:loc11], axis=0)
    # 计算8~13Hz的PSD
    PSD_bands9 = np.sum(yy[loc12:loc14], axis=0)
    # 计算11~16Hz的PSD
    PSD_bands10 = np.sum(yy[loc13:loc16], axis=0)
    # 计算15~30Hz的PSD
    PSD_bands11 = np.sum(yy[loc15:loc17], axis=0)

    # 经过短时傅里叶变换处理后0.06-0.1Hz频段功率的最大值
    max_PSD_bands1 = np.max(PSD_bands1)
    # 经过短时傅里叶变换处理后0.06-0.1Hz频段功率的最小值
    min_PSD_bands1 = np.min(PSD_bands1)
    # 经过短时傅里叶变换处理后0.06-0.1Hz频段功率的均值
    mean_PSD_bands1 = np.mean(PSD_bands1)
    # 经过短时傅里叶变换处理后0.06-0.1Hz频段功率的中位数
    median_PSD_bands1 = np.median(PSD_bands1)
    # 经过短时傅里叶变换处理后0.06-0.1Hz频段功率的标准差
    std_PSD_bands1 = np.std(PSD_bands1)


    # 经过短时傅里叶变换处理后0.1-0.3Hz频段功率的最大值
    max_PSD_bands2 = np.max(PSD_bands2)
    # 经过短时傅里叶变换处理后0.1-0.3Hz频段功率的最小值
    min_PSD_bands2 = np.min(PSD_bands2)
    # 经过短时傅里叶变换处理后0.1-0.3Hz频段功率的均值
    mean_PSD_bands2 = np.mean(PSD_bands2)
    # 经过短时傅里叶变换处理后0.1-0.3Hz频段功率的中位数
    median_PSD_bands2 = np.median(PSD_bands2)
    # 经过短时傅里叶变换处理后0.1-0.3Hz频段功率的标准差
    std_PSD_bands2 = np.std(PSD_bands2)


    # 经过短时傅里叶变换处理后0.3-0.5Hz频段功率的最大值
    max_PSD_bands3 = np.max(PSD_bands3)
    # 经过短时傅里叶变换处理后0.3-0.5Hz频段功率的最小值
    min_PSD_bands3 = np.min(PSD_bands3)
    # 经过短时傅里叶变换处理后0.3-0.5Hz频段功率的均值
    mean_PSD_bands3 = np.mean(PSD_bands3)
    # 经过短时傅里叶变换处理后0.3-0.5Hz频段功率的中位数
    median_PSD_bands3 = np.median(PSD_bands3)
    # 经过短时傅里叶变换处理后0.3-0.5Hz频段功率的标准差
    std_PSD_bands3 = np.std(PSD_bands3)

    # 经过短时傅里叶变换处理后0.5-1Hz频段功率的最大值
    max_PSD_bands4 = np.max(PSD_bands4)
    # 经过短时傅里叶变换处理后0.5-1Hz频段功率的最小值
    min_PSD_bands4 = np.min(PSD_bands4)
    # 经过短时傅里叶变换处理后0.5-1Hz频段功率的均值
    mean_PSD_bands4 = np.mean(PSD_bands4)
    # 经过短时傅里叶变换处理后0.5-1Hz频段功率的中位数
    median_PSD_bands4 = np.median(PSD_bands4)
    # 经过短时傅里叶变换处理后0.5-1Hz频段功率的标准差
    std_PSD_bands4 = np.std(PSD_bands4)

    # 经过短时傅里叶变换处理后0.5-2Hz频段功率的最大值
    max_PSD_bands5 = np.max(PSD_bands5)
    # 经过短时傅里叶变换处理后0.5-2Hz频段功率的最小值
    min_PSD_bands5 = np.min(PSD_bands5)
    # 经过短时傅里叶变换处理后0.5-2Hz频段功率的均值
    mean_PSD_bands5 = np.mean(PSD_bands5)
    # 经过短时傅里叶变换处理后0.5-2Hz频段功率的中位数
    median_PSD_bands5 = np.median(PSD_bands5)
    # 经过短时傅里叶变换处理后0.5-2Hz频段功率的标准差
    std_PSD_bands5 = np.std(PSD_bands5)


    # 经过短时傅里叶变换处理后1.6-4Hz频段功率的最大值
    max_PSD_bands6 = np.max(PSD_bands6)
    # 经过短时傅里叶变换处理后1.6-4Hz频段功率的最小值
    min_PSD_bands6 = np.min(PSD_bands6)
    # 经过短时傅里叶变换处理后1.6-4Hz频段功率的均值
    mean_PSD_bands6 = np.mean(PSD_bands6)
    # 经过短时傅里叶变换处理后1.6-4Hz频段功率的中位数
    median_PSD_bands6 = np.median(PSD_bands6)
    # 经过短时傅里叶变换处理后1.6-4Hz频段功率的标准差
    std_PSD_bands6 = np.std(PSD_bands6)


    # 经过短时傅里叶变换处理后3-4.5Hz频段功率的最大值
    max_PSD_bands7 = np.max(PSD_bands7)
    # 经过短时傅里叶变换处理后3-4.5Hz频段功率的最小值
    min_PSD_bands7 = np.min(PSD_bands7)
    # 经过短时傅里叶变换处理后3-4.5Hz频段功率的均值
    mean_PSD_bands7 = np.mean(PSD_bands7)
    # 经过短时傅里叶变换处理后3-4.5Hz频段功率的中位数
    median_PSD_bands7 = np.median(PSD_bands7)
    # 经过短时傅里叶变换处理后3-4.5Hz频段功率的标准差
    std_PSD_bands7 = np.std(PSD_bands7)



    # 经过短时傅里叶变换处理后4-7Hz频段功率的最大值
    max_PSD_bands8 = np.max(PSD_bands8)
    # 经过短时傅里叶变换处理后4-7Hz频段功率的最小值
    min_PSD_bands8 = np.min(PSD_bands8)
    # 经过短时傅里叶变换处理后4-7Hz频段功率的均值
    mean_PSD_bands8 = np.mean(PSD_bands8)
    # 经过短时傅里叶变换处理后4-7Hz频段功率的中位数
    median_PSD_bands8 = np.median(PSD_bands8)
    # 经过短时傅里叶变换处理后4-7Hz频段功率的标准差
    std_PSD_bands8 = np.std(PSD_bands8)




    # 经过短时傅里叶变换处理后8-13Hz频段功率的最大值
    max_PSD_bands9 = np.max(PSD_bands9)
    # 经过短时傅里叶变换处理后8-13Hz频段功率的最小值
    min_PSD_bands9 = np.min(PSD_bands9)
    # 经过短时傅里叶变换处理后8-13Hz频段功率的均值
    mean_PSD_bands9 = np.mean(PSD_bands9)
    # 经过短时傅里叶变换处理后8-13Hz频段功率的中位数
    median_PSD_bands9 = np.median(PSD_bands9)
    # 经过短时傅里叶变换处理后8-13Hz频段功率的标准差
    std_PSD_bands9 = np.std(PSD_bands9)



    # 经过短时傅里叶变换处理后11-16Hz频段功率的最大值
    max_PSD_bands10 = np.max(PSD_bands10)
    # 经过短时傅里叶变换处理后11-16Hz频段功率的最小值
    min_PSD_bands10 = np.min(PSD_bands10)
    # 经过短时傅里叶变换处理后11-16Hz频段功率的均值
    mean_PSD_bands10 = np.mean(PSD_bands10)
    # 经过短时傅里叶变换处理后11-16Hz频段功率的中位数
    median_PSD_bands10 = np.median(PSD_bands10)
    # 经过短时傅里叶变换处理后11-16Hz频段功率的标准差
    std_PSD_bands10 = np.std(PSD_bands10)



    # 经过短时傅里叶变换处理后15-30Hz频段功率的最大值
    max_PSD_bands11 = np.max(PSD_bands11)
    # 经过短时傅里叶变换处理后15-30Hz频段功率的最小值
    min_PSD_bands11 = np.min(PSD_bands11)
    # 经过短时傅里叶变换处理后15-30Hz频段功率的均值
    mean_PSD_bands11 = np.mean(PSD_bands11)
    # 经过短时傅里叶变换处理后15-30Hz频段功率的中位数
    median_PSD_bands11 = np.median(PSD_bands11)
    # 经过短时傅里叶变换处理后15-30Hz频段功率的标准差
    std_PSD_bands11 = np.std(PSD_bands11)



    # # 经过短时傅里叶变换处理后8-13Hz频段功率的平均值
    # mean_PSD_bands4 = np.mean(PSD_bands4)

    # Shannon Entropy
    aa = np.array([abs(int(i)) for i in notch_filter])
    d = Dist(np.max(aa)+1)
    # print(aa)
    for x in aa:
        d.tick(x)
    shannon_entropy = entropy(d, b=750)

    #return [max_filter, min_filter, kurtosis, max_PSD_bands1, min_PSD_bands1, min_PSD_bands2, min_PSD_bands3,
            #min_PSD_bands4, shannon_entropy]
    return [min_filter, max_filter,kurtosis, max_PSD_bands1, min_PSD_bands1, mean_PSD_bands1, median_PSD_bands1, std_PSD_bands1, max_PSD_bands2, min_PSD_bands2, mean_PSD_bands2, median_PSD_bands2, std_PSD_bands2,
            max_PSD_bands3, min_PSD_bands3, mean_PSD_bands3, median_PSD_bands3, std_PSD_bands4, max_PSD_bands4, min_PSD_bands4, mean_PSD_bands4, median_PSD_bands4, std_PSD_bands4,
            max_PSD_bands5, min_PSD_bands5, mean_PSD_bands5, median_PSD_bands5, std_PSD_bands5,max_PSD_bands6, min_PSD_bands6, mean_PSD_bands6, median_PSD_bands6, std_PSD_bands6,
            max_PSD_bands7, min_PSD_bands7, mean_PSD_bands7, median_PSD_bands7, std_PSD_bands7,max_PSD_bands8, min_PSD_bands8, mean_PSD_bands8, median_PSD_bands8, std_PSD_bands8,
            max_PSD_bands9, min_PSD_bands9, mean_PSD_bands9, median_PSD_bands9, std_PSD_bands9,max_PSD_bands10, min_PSD_bands10, mean_PSD_bands10, median_PSD_bands10, std_PSD_bands10,
            max_PSD_bands11, min_PSD_bands11, mean_PSD_bands11, median_PSD_bands11, std_PSD_bands11, shannon_entropy]

## 该函数处理数据并调用generate_features()函数生成指定数据

def generate_data_list(file_path):
    oc_map = {"o": 2, "c": 3, "m": 0}  # 将文件名的首字母进行映射
    # file_path = 'F:\\eegeyestatus\\train_data_files\\'
    file_path = file_path
    data_list = []  # 收集处理后的训练数据的容器
    for parent, dir_names, file_names in os.walk(file_path):
        # print("parent:", parent)
        # print("dir_names:", dir_names)
        # print("file_names:", file_names)
        for per_txt in file_names:
            # print("per_txt:", per_txt)
            eeg_data = np.loadtxt(parent + "\\" + per_txt, dtype=str).astype("float")
            # print("eeg_data:", eeg_data)
            data_array = generate_features(eeg_data)
            n = len(parent)
            data_array.append(parent[n-2:n+1])
            data_array.append(oc_map[per_txt[0]])
            if per_txt[0] == 'c' and per_txt[8] == '_':
                data_array.append(per_txt[7])
            elif per_txt[0] == 'c' and per_txt[9] == '_':
                data_array.append(per_txt[7:9])
            elif per_txt[0] == 'o' and per_txt[7] == '_':
                data_array.append(per_txt[6])
            else:
                data_array.append(per_txt[6:8])

            data_list.append(data_array)
    return data_list

## 该函数计算香农熵，可能有点问题

def calcShannonEnt(dataSet):
    numEntires = len(dataSet)  # 返回数据集的行数
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:  # 对每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # Label计数
    shannonEnt = 0.0  # 经验熵(香农熵)
    for key in labelCounts:  # 计算香农熵
        prob = float(labelCounts[key]) / numEntires  # 选择该标签(Label)的概率
        shannonEnt -= prob * math.log(prob, 2)  # 利用公式计算
    return shannonEnt

## 该函数可以绘制机器学习准确率（实际对预测）的比较图

def multi_column_plot(list, plt_title):
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(40, 13))
    total_rows = len(list[0])
    total_columns = len(list)

    subplot_index = 1

    # ic(total_rows)
    # ic(total_columns)

    for wave_list in list:
        for wave_obj in wave_list:
            # ic(type(wave_obj))
            # ic(wave_obj['title'])
            ax = plt.subplot(total_columns, total_rows, subplot_index)  # 行 列 当前
            # print(wave_obj['title'])
            ax.set_title(f"{wave_obj['title']} - {plt_title}")
            ax.plot(wave_obj['data'], label=wave_obj['title'], color=wave_obj['color'])
            # ax.set_xlim([0, 7500])
            # ax.set_ylim([-9e49, 9e49])
            # ax.set_xlabel('seconds')

            subplot_index += 1

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()








