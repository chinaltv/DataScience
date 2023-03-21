import math
import os
from pyinform.dist import Dist
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import scipy.interpolate as spi
from scipy import stats  # scipy中的stats可以做统计推断
from pyinform.shannon import entropy

# 计算香农熵
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


# 先带通滤波再陷波滤波
def bandpass_notch(raw_data):
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


# 计算得出9个值
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
    aa=np.array([abs(int(i)) for i in notch_filter])
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
