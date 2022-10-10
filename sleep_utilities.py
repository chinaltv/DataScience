# 工作清单
   
# 3 单独函数（batch_figure）制定输出原始数据图、处理数据图并进行坏段标记（没什么好思路，暂时鸽）  
# 4 特征提取（batch_feature_extraction，基于generate_data0825.py）  


import numpy as np
import pandas as pd
import os
import mne
from mne.time_frequency import psd_array_welch

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

    def art_min_pos(ss, number): # 该函数输入切片功率列表ss，并返回切片对象伪迹的相对位置（排序，确定前10%）
        minlist = sorted(ss, reverse = False)[:cut_number] # 创建一个基于自定义数字的最小值列表
        ss_dict = {}  # 创建一个空字典
        for i, b in enumerate(ss):
            ss_dict[b] = i
        min_indices = [ss_dict[a] for a in minlist]
        return min_indices

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
    art_indices = art_max_pos(ss, cut_number) + art_min_pos(ss, cut_number) # 借用上面做的函数，输出最大/小值index列表

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



# 还没动工

def feature_extraction(raw):
    return raw