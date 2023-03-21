import numpy as np
import mne
import matplotlib # 用于MNE的GUI窗口弹出，该功能不推荐在Jupyter notebook中使用
matplotlib.use('TKAgg')
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet

## 设置工作路径，读取数据与查看原始数据信息

data_path = "/..."
raw = read_function(data_path, preload = True)
# read_function include:
# mne.io.read_raw_  + 
# (.set)eeglab(EEGLAB); 
# (.vhdr)brainvision(BrainVision); 
# (.edf)edf(EDF)
# (.gdf)gdf(GDF)
# (.cnt)cnt(Neuroscan)
# (.egi/.mff)egi
# (.data)nicolet
# (.nxe)eximia(Nexstim eXimia)
# (.lay/.dat)persyst(Persyst)
# (.eeg)nihon(Nihon Kohden)

print(raw)
print(raw.info)

# 相关数据读取信息说明：
# 导联个数n，时间点t，对应长度，数据大小：n x t (..s), ~ MB
# 采样率Hz：sfreq
# 高通滤波Hz：highpass
# 低通滤波Hz：lowpass
# 导联名称：ch_names




## 修订相关信息

# 如有需要（如导联名称并非标准名称），导入.locs文件并重命名电极位置信息
locs_info_path = "/..."
montage = mne.channels.read_custom_montage(locs_info_path) # 读取电极位置信息
new_chan_names = np.loadtxt(locs_info_path, dtype = str, usecols = 3) # 读取正确导联名称
old_chan_names = raw.info["ch_names"] # 读取旧导联名称
chan_names_dict = {old_chan_names[i]:new_chan_names[i] for i in range(n)} # 创建字典并匹配新旧导联名称，请注意修改n为导联实际个数
raw.rename_channels(chan_names_dict) # 更新数据中的导联名称
raw.set_montage(montage) # 传入数据的电极位置信息
# 如有需要，使用特定函数改变电极位点，例如国际标准10-20系统：standard_1020
montage = mne.channels.make_standard_montage("...")
# 如有需要，设定导联类型为eeg和eog
chan_types_dict = {new_chan_names[i]:"eeg" for i in range(n)} # 请注意修改n为导联实际个数
chan_types_dict = {"EOG1":"eog", "EOG2":"eog"}
raw.set_channel_types(chan_types_dict)

## 可视化原始数据，请注意修改相应信息，如n为导联实际个数，ns为实际需要的秒数
raw.plot(duration = ns, n_channels = n, clippping = None) # 原始数据波形图
raw.plot_psd(average = True) # 原始数据功率谱图
raw.plot_sensors(ch_type = 'eeg', show_name = True) # 电极拓扑图
raw.plot_psd_topo() # 原始数据拓扑图




## 滤波
# 每次滤波后，可以再次绘制功率谱图查看结果
raw.plot_psd(average = True) # 原始数据功率谱图

# 陷波滤波：去除工频
raw = raw.notch_filter(freqs=(n)) # 请记住修改n为相应的Hz。通过原始数据功率谱图可以看出一些实际情况。如出现特殊的波形可认定为存在杂音，需要对数据进行工频过滤处理。国内和香港澳门的数据一般会出现在50Hz处。

# 高低通滤波：预处理
raw = raw.filter(l_freq = 0.1, h_freq = 30) # 高通滤波一般是为消除电压漂移，低通滤波为消除高频噪音，一般设定为30Hz低通和0.1Hz高通，另外默认使用FIR滤波方法，如想使用IIR滤波方法，可以修改参数method = 'iir'。




## 去除伪迹
# 去坏段
fig = raw.plot(duration = ns, n_channels = n, clippping = None)  # 原始数据波形图
fig.canvas.key_press_event('button_name') # 使用MNE建立一个交互式数据地形图界面，并按某个button（自己设置）打开相应的GUI，手动add new label进行坏段标记。

# 去坏道
raw.info['bads'].append('bad_track_name') # 请注意手动修改坏道名称
raw.info['bads'].extend(['bad_track_name_1', 'bad_track_name_2']) # 多个坏道的标记
print(raw.info['bads'])

# 坏道插值重建
# 本质是对相应标记'bads'的导联进行信号重建后，将原有标记删除。如果不需要去掉原坏道标记，可使用参数reset_bads = False
raw = raw.interpolate_bads()




## 独立成分分析（ICA）/盲源分离（Blind source separation, BSS）
'''
独立成分分析是一种信号处理的常见方法，在神经计算领域中本质上是非监督学习（unsupervised learning）问题，即从数据中得到单纯的结构（数据挖掘）。与主成分分析PCA相近，均是在多元（多维）统计数据中寻找潜在因子或成分，提取有效统计数据的一种降维技术。PCA/ICA个人可以认为是一种纯技术而非统计方法，与FA相比，在较为丧失解释性的情况下更有效地提取成分。PCA与ICA的主要差别在于评价准则，pca为最大化方差，使得残余方差最小，或信息损失最小（方差即信息）。ica为最大化独立性，使联合概率与各分量概率乘积最接近。ICA在信号处理中的主要作用主要是改变数据并产生更清晰的信号源。
另一个解释是ICA作为信号处理的常见方法，是经过多部迭代寻优，按照信号之间（而非方差）独立最大假设的原则将信号解混输出。在对信号进行ICA之前，可能需要进行PCA对数据进行预处理。
'''


ica = ICA(max_iter = 'auto') # 构造ICA analyst/object，可以使用参数n_componenets = n固定成分个数
raw_for_ica = raw.copy().filter(l_freq = 1, h_freq = None) # 复制后对高通1Hz的数据进行进行高低通滤波处理
ica.fit(raw_for_ica) # 进行ICA
ica.plot_sources(raw_for_ica) # 绘制各成分的时序信号图
ica.plot_components() # 绘制各成分地形图
# 单独对某些成分进行操作，列表填写相应序号，注意以0为起点
ica.plot_overlay(raw_for_ica, exclude = [special_value1, special_value2]) # 查看去掉某一成分前后信号差异
ica.plot_properties(raw, picks = [special_value1, special_value2]) # 单独可视化每个成分
ica.exclude = [special_value1, special_value2] # 单独剔除某些成分，设定序号
ica.apply(raw) # 应用到脑电数据中
raw.plot(duration = ns, n_channels = n, clipping = None) # 绘制各成分的数据波形图




## 进行重参考（Re-reference）
'''
EEG所测量的是某个电极的电位，又或者叫电势。但由于单个电极电势是无法测量的，所以需要搭配一个已知电极电势的电极作为参考电极。理想的参考电极的电势应为零（零电势电位，其电势应该是不变化的）。比较常见的参考方式有双侧乳突参考、平均参考、双极参考。参考导联的方式可以根据自己的研究灵活变动，目前关于参考电极的研究也仍在继续，比如参考电极标准化技术（ReferenceElectrode Standardization Technique，REST)。它不依赖于头皮上的中性参考电极位置或电位零点，而是将头皮上的一点或其他参考点近似地转换为以空间的无限远点为参考位置。
'''
raw.set_eeg_reference(ref_channels = [ref_elec]) # 使用特定参考
raw.set_eeg_reference(ref_channels = 'average') # 使用平均参考
raw.set_eeg_reference(ref_channels = 'REST') # 使用REST参考
raw_bip_ref = mne.set_bipolar_reference(raw, anode = ['EEG X'], cathode = ['EEG Y']) # 双极参考情况下，使用对应的阳极和阴极导联




## 分段
'''
MNE使用两种数据结构存储事件信息，分别为Events和Annotations。对于Annotations对象，用字符串表示时间类型，和为'square'与'rt'的描述信息（marker）。Event为我们进行数据分段需要用到的事件记录数据类型，用一个整形Event ID编码事件类型，用样本形式表示时间，并不含有Marker的持续时长，内部数据类型为NumPy Array。
'''
# 提取事件信息
print(raw.annotations) # 查看数据中的markers
print(raw.annotations.duration) # 查看基于Annotations打印数据的事件持续时长
print(raw.annotations.desscription) # 查看基于Annotations打印数据的事件的描述信息
print(raw.annotations.onset) # 查看基于Annotations打印数据的事件的开始时间
# 数据类型转换
events, event_id = mne.events_from_annotations(raw) # 将数据转换为Events类型
print(events.shape, event_id) # 打印出evetns矩阵的shape和event_id
# 基于Events对数据进行分段，并进行可视化
'''
该部分没有看懂，等候后续研究
'''
epochs = mne.Epochs(raw, events, event_id = n, tmin = n, tmax = n, baseline = (n, n), preload = True, reject = dict(eeg = 2e-4))
print(epochs)
epochs.plot(n_epochs = n) # 可视化n个分段数据
epochs.plot_psd(picks = 'eeg') # 绘制功率谱图（逐导联）
bands = [(n, n, 'Theta'), (n, n, 'Alpha'), (n, n, 'Beta')]
epochs.plot_psd_topomap(bands = bands, vlim = 'joint') # 绘制功率谱拓扑图（分Theta、Alpha和Beta频段）




## 叠加平均
'''
MNE使用Epochs类来存储分段数据，用Evoked类来存储叠加平均数据
'''
evoked = epochs.average() # 数据叠加平均
evoked.plot() # 绘制逐导联的时序信号图
times = np.linspace(ns, ns, ns, ...) 
evoked.plot_topomap(times = times, colorbar = True) # 绘制不同时间段的地形图
evoked.plot_topomap(times = ns, average = ns) # 绘制某一特定时刻（取某个时间段以内均值，差值为average）的地形图
evoked.plot_joint() # 绘制联合图
evoked.plot_image() # 绘制逐导联热力图
evoked.plot_topo() # 绘制拓扑时序信号图
mne.viz.plot_compare_evokeds(evokeds = evoked, combine = 'mean') # 绘制平均所有电极后的ERP
mne.viz.plot_compare_evokeds(evokeds = evoked, picks = ['a', 'b', 'c'], combine = 'mean') # 绘制枕叶电极的平均ERP




## 时频分析
'''
MNE提供三种时频分析方法：
1. Morlet wavelets: mne.time_frequency.tfr_morlet()
2. DPSS tapers: mne.time_frequency.tfr_multitaper()
3. Stockwell Transform: mne.time_frequency.tfr_stockwell()
'''
# 计算能量（Power）与试次间耦合（inter-trial coherence，ITC）。默认返回试次平均后结果，如果想获取单独的时频分析结果，将average = default(True)设置为False即可。
freqs = np.logspace(*np.log10([nHz, nHz]), num = n) # 设定相关参数，注意频段选择
n_cycles = freqs / 2
power, itc = tfr_morlet(epochs, freqs = freqs, n_cycles = n_cycles, use_fft = True)

# 绘制结果。其中method对应多种方法，如'mean'（减去baseline均值），'ratio'（除以basline均值），'logratio'（除以baseline均值并取log），'percent'（减去baseline均值并除以baseline均值），'zscore'（减去baseline均值再除以baseline标准差）和'zlogratio'（除以baseline均值并取log再除以baseline取log后的标准差），可根据自己方法选用。下面以power的结果绘制为例。如需要绘制ITC，则仅需要将power改成itc。
power.plot(picks = ['a', 'b', 'c'], baseline = (n, n), mode = 'method', title = 'auto') # 枕叶导联的power结果
power.plot_topo(baseline = (n, n), mode = 'method', title = 'Average power') # 绘制power拓扑图
power.plot_topomap(tmin = ns, tmax = ns, fmin = n, fmax = n, baseline = (n, n), mode = 'method', title = 'Theta/Alpha...') # 根据不同power（theta/alpha...），绘制tmin~tmax的power拓扑图
power.plot_joint(baseline = (n, n), mode = 'method', tmin = ns, tmax = ns, timefreqs = [(ns, nHz), (ns, nHz)]) # 绘制tmin~tmax的联合图，并绘制ns时nHz左右的结果




## 提取数据
# 使用get_data()对Raw类，Epochs类和Evocked类提取原始数据/分段数据/时频结果矩阵
epochs_array = epochs.get_data() # 以epochs为例，获取分段数据矩阵
power_array = power.data # 获取.data数据
print(epochs_array.shape) # 其shape会生成一个列表，分别代表试次，导联与时间点个数
print(epochs.array)
# 如想获取eog外的导联数据，则可将上述代码改为epochs_array = epochs.get_data(picks = ['eeg'])



