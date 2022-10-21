import os
import json
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipympl
from bisect import bisect_left

# 该函数已稳定，导入json文件并生成dataframe。可以用pd.read_json(input_path, lines = True)代替，除了展示的时候对齐不同。

def read_one(InputPaths):
    content = []
    raw = open(InputPaths, 'r', encoding = 'utf-8') # 读取文件
    for line in raw.readlines():
        if line.startswith(u'\ufeff'):
            line = line.encode('utf8')[3:].decode('utf8')
        dict = json.loads(line)
        content.append(dict)
    df = pd.DataFrame(content)
    raw.close()
    return df




# 该函数已稳定，使用已有的json导入文件（使用内置的read_one函数）并指定背景图和z/y轴理论长度（从程序那边报告来的，unity的东西），生成整个流程的注视点热力图

def json_2Dhm(json_filepath, background_filepath, z_len = [-3.302,3.395], y_len = [0,3.938], interest_vari = 'fp'): # Your json_filepath must be like: './paired_data/ASD/A1.json', your interest_vari must be like: 'fp'

    def read_one(InputPaths):
        content = []
        raw = open(InputPaths, 'r', encoding='utf-8') # 读取文件
        for line in raw.readlines():
            if line.startswith(u'\ufeff'):
                line = line.encode('utf8')[3:].decode('utf8')
            dict = json.loads(line)
            content.append(dict)
        df = pd.DataFrame(content)
        return df

    raw_json = read_one(json_filepath)[interest_vari]
    background = mpimg.imread(background_filepath)

    z_scale = (z_len[1] - z_len[0]) / background.shape[1] # 6.697 / 824
    y_scale = (y_len[1] - y_len[0]) / background.shape[0] # 3.938 / 490

    raw_json = raw_json.values.tolist()
    z,y,z_f,y_f = [],[],[],[]

    for dict in raw_json:
        z.append(-dict['Z'])
        y.append(dict['Y'])

    for elements in z:
        z_f.append((elements-z_len[0])/z_scale)
    for elements in y:
        y_f.append(elements/y_scale)

    zy_df = pd.DataFrame({'z': z_f, 'y': y_f})
    plt.imshow(background, extent = [min(z_f), max(z_f), min(y_f), max(y_f)])
    return plt.scatter(zy_df['z'], zy_df['y'], c='r', marker='.', s = 10, alpha = 0.1)




# 该函数已稳定，生成第一次和最后一次对话时间内的注视点热图，本质是对json_2Dhm()的修改

def json_2Dhm_fl(json_filepath, ts_filepath, background_filepath, z_len = [-3.302,3.395], y_len = [0,3.938], interest_vari = 'fp'): # Your raw_json must be like: read_one('./paired_data/ASD/A1.json')
    def read_one(InputPaths):
        content = []
        raw = open(InputPaths, 'r', encoding='utf-8') # 读取文件
        for line in raw.readlines():
            if line.startswith(u'\ufeff'):
                line = line.encode('utf8')[3:].decode('utf8')
            dict = json.loads(line)
            content.append(dict)
        df = pd.DataFrame(content)
        return df

    ts_df = read_one(ts_filepath)
    raw_json = read_one(json_filepath)
    background = mpimg.imread(background_filepath)
    z_scale = (z_len[1] - z_len[0]) / background.shape[1]
    y_scale = (y_len[1] - y_len[0]) / background.shape[0]

    SayHello_ts_start = bisect_left(raw_json['ts'].values, ts_df['SayHelloDialogueList'][0][0]['BeforeDialogueAtFirstTs'])
    SayHello_ts_end = bisect_left(raw_json['ts'].values, ts_df['SayHelloDialogueList'][0][0]['AfterDialogueAtFirstTs'])
    FinalDialogue_ts_start = bisect_left(raw_json['ts'].values, ts_df['FinalDialogueList'][0][0]['BeforeDialogueAtFirstTs'])
    FinalDialogue_ts_end = bisect_left(raw_json['ts'].values, ts_df['FinalDialogueList'][0][0]['AfterDialogueAtFirstTs'])
    SayHello_df = raw_json.iloc[SayHello_ts_start: SayHello_ts_end + 1][interest_vari].values.tolist()
    FinalDialogue_df = raw_json.iloc[FinalDialogue_ts_start: FinalDialogue_ts_end + 1][interest_vari].values.tolist()

    def zy_df(dataframe, z_len = z_len, y_len = y_len, z_scale = z_scale, y_scale = y_scale):
        z,y,z_f,y_f = [],[],[],[]
        for dict in dataframe:
            z.append(-dict['Z'])
            y.append(dict['Y'])
        for elements in z:
            z_f.append((elements-z_len[0])/z_scale)
        for elements in y:
            y_f.append(elements/y_scale)
        zy_df = pd.DataFrame({'z': z_f, 'y': y_f})
        return zy_df, z_f, y_f

    SayHello_df, SayHello_zf, SayHello_yf = zy_df(SayHello_df)
    FinalDialogue_df, FinalDialogue_zf, FinalDialogue_yf = zy_df(FinalDialogue_df)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('First and Last Dialogue Hotmap')
    axs[0].imshow(background, extent = [0, background.shape[1], 0, background.shape[0]])
    axs[0].scatter(SayHello_df['z'], SayHello_df['y'], c='r', marker='.', s = 10, alpha = 0.1)
    axs[1].imshow(background, extent = [0, background.shape[1], 0, background.shape[0]])
    axs[1].scatter(FinalDialogue_df['z'], FinalDialogue_df['y'], c='r', marker='.', s = 10, alpha = 0.1)
    
    return axs




# 该函数已稳定，实现批量提取多个json文件的某个变量

def batch_extra_vari(FileName_List, input_path, extvari = 'fon'):
    df_agg = pd.DataFrame()
    fon = []
    i = 0
    for FileName in FileName_List: # 对于所有文件名
        InputPaths = os.path.join('%s/%s' % (input_path, FileName)) # 拼接路径
        df_single = read_one(InputPaths)
        df_agg[i] = df_single[extvari]
        i += 1
    return df_agg