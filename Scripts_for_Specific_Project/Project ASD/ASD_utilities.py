import os
import json
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipympl
from bisect import bisect_left

# 导入json文件并生成dataframe。可以用pd.read_json(input_path, lines = True)代替，除了展示的时候对齐不同。

def read_one(InputPaths):
    content = []
    raw = open(InputPaths, 'r', encoding = 'utf-8') # 读取文件
    for line in raw.readlines():
        if line.startswith(u'\ufeff'):
            line = line.encode('utf8')[3:].decode('utf8')
        dict = json.loads(line)
        content.append(dict)
    df = pd.DataFrame(content)
    return df




# 使用已有的json导入文件（使用内置的read_one函数）并指定背景图和z/y轴理论长度（从程序那边报告来的，unity的东西），生成整个流程的注视点热力图

def json_2Dhm(json_filepath, background_filepath, z_len = [-3.34,3.49], y_len = [-0.02,3.938], interest_vari = 'fp'): # Your json_filepath must be like: './paired_data/ASD/A1.json', your interest_vari must be like: 'fp'

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
    raw_json = raw_json.values.tolist()
    z,y = [],[]

    for dict in raw_json:
        z.append(z_len[1] - dict['Z'] + z_len[0])
        y.append(dict['Y'])
    zy_df = pd.DataFrame({'z': z, 'y': y})
    fig, ax = plt.subplots()
    ax.imshow(background, extent = [z_len[0], z_len[1], y_len[0], y_len[1]])
    plt.scatter(zy_df['z'], zy_df['y'], c='r', marker='.', s = 10, alpha = 0.1)
    return plt.show()




# 生成第一次和最后一次对话时间内的注视点热图，本质是对json_2Dhm()的修改

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

    SayHello_ts_start = bisect_left(raw_json['ts'].values, ts_df['SayHelloDialogueList'][0][0]['BeforeDialogueAtFirstTs'])
    SayHello_ts_end = bisect_left(raw_json['ts'].values, ts_df['SayHelloDialogueList'][0][0]['AfterDialogueAtFirstTs'])
    FinalDialogue_ts_start = bisect_left(raw_json['ts'].values, ts_df['FinalDialogueList'][0][0]['BeforeDialogueAtFirstTs'])
    FinalDialogue_ts_end = bisect_left(raw_json['ts'].values, ts_df['FinalDialogueList'][0][0]['AfterDialogueAtFirstTs'])
    SayHello_df = raw_json.iloc[SayHello_ts_start: SayHello_ts_end + 1][interest_vari].values.tolist()
    FinalDialogue_df = raw_json.iloc[FinalDialogue_ts_start: FinalDialogue_ts_end + 1][interest_vari].values.tolist()

    def zy_df(dataframe, z_len = z_len, y_len = y_len):
        
        def scale(x, srcRange, dstRange):
            return (x - srcRange[0]) * (dstRange[1] - dstRange[0]) / (srcRange[1] - srcRange[0]) + dstRange[0]

        z,y,z_f,y_f = [],[],[],[]
        for dict in dataframe:
            z.append(dict['Z'])
            y.append(dict['Y'])
        for elements in z:
            elements_trans = z_len[1] - elements + z_len[0]
            z_f.append(scale(elements_trans, z_len, [0, background.shape[1]]))
        for elements in y:
            y_f.append(scale(elements, y_len, [0, background.shape[0]]))
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




# 批量提取多个json文件的某个变量

def batch_extra_vari(FileName_List, input_path, extvari = 'fon'):
    df_agg = pd.DataFrame()
    i = 0
    for FileName in FileName_List: # 对于所有文件名
        InputPaths = os.path.join('%s/%s' % (input_path, FileName)) # 拼接路径
        df_single = read_one(InputPaths)
        df_agg[i] = df_single[extvari]
        i += 1
    return df_agg




# 单个提取眼动数据注视点频次 TODO # Exp_Areas尚未确认

def Fon_count_one(json_dataframe, colname = 'count'):
    your_interest_vari = 'fon'
    Exp_Areas = ['NPCHair', 'NPCForehead', 'NPCBrowLeft', 'NPCBrowRight', 'NPCEyeLeft', 'NPCEyeRight', 'NPCEye', 'NPCEarLeft', 'NPCEarRight', 'NPCNose', 'NPCMouth', 'NPCCheekLeft', 'NPCCheekRight', 'NPCFace', 'NPCBody', 'NPCLeftUpperarm', 'NPCRightUpperarm', 'NPCLeftForearm', 'NPCRightForearm', 'NPCLeftHand', 'NPCRightHand', 'NPCHip', 'NPCLeftThigh', 'NPCRightThigh', 'NPCLeftShank', 'NPCRightShank', 'NPCLeftFeet', 'NPCRightFeet', 'LeftHand', 'RightHand']

    # 以下正式函数
    dataframe_selected = json_dataframe.loc[json_dataframe[your_interest_vari].isin(Exp_Areas)] # 排除无关数据
    count_no = dataframe_selected[your_interest_vari].value_counts()
    output_dict = {}
    for i in Exp_Areas:
        if i in count_no.index:
            output_dict[i] = int(count_no[i])
        else:
            output_dict[i] = np.NaN # 0也可以
    output_df = pd.DataFrame(pd.Series(output_dict), columns = [colname])
    return output_df




# 只是进行了一个包装 TODO # Exp_Areas尚未确认

def Fon_one_agg(json_dataframe, colname = 'count'):   
    Exp_Areas = ['NPCHair', 'NPCForehead', 'NPCBrowLeft', 'NPCBrowRight', 'NPCEyeLeft', 'NPCEyeRight', 'NPCEye', 'NPCEarLeft', 'NPCEarRight', 'NPCNose', 'NPCMouth', 'NPCCheekLeft', 'NPCCheekRight', 'NPCFace', 'NPCBody', 'NPCLeftUpperarm', 'NPCRightUpperarm', 'NPCLeftForearm', 'NPCRightForearm', 'NPCLeftHand', 'NPCRightHand', 'NPCHip', 'NPCLeftThigh', 'NPCRightThigh', 'NPCLeftShank', 'NPCRightShank', 'NPCLeftFeet', 'NPCRightFeet', 'LeftHand', 'RightHand']
    Head = ['NPCHair', 'NPCForehead', 'NPCBrowLeft', 'NPCBrowRight', 'NPCEyeLeft', 'NPCEyeRight', 'NPCEye', 'NPCEarLeft', 'NPCEarRight', 'NPCNose', 'NPCMouth', 'NPCCheekLeft', 'NPCCheekRight', 'NPCFace']
    Body = ['NPCBody', 'NPCLeftUpperarm', 'NPCRightUpperarm', 'NPCLeftForearm', 'NPCRightForearm', 'NPCLeftHand', 'NPCRightHand', 'NPCHip', 'NPCLeftThigh', 'NPCRightThigh', 'NPCLeftShank', 'NPCRightShank', 'NPCLeftFeet', 'NPCRightFeet']
    Others = ['LeftHand', 'RightHand']
    your_interest_vari = 'fon'

    dataframe_head = json_dataframe.loc[json_dataframe[your_interest_vari].isin(Head)] # 排除无关数据
    dataframe_body = json_dataframe.loc[json_dataframe[your_interest_vari].isin(Body)] # 排除无关数据
    dataframe_others = json_dataframe.loc[json_dataframe[your_interest_vari].isin(Others)] # 排除无关数据
    count_no_head = dataframe_head[your_interest_vari].value_counts()
    count_no_body = dataframe_body[your_interest_vari].value_counts()
    count_no_others = dataframe_others[your_interest_vari].value_counts()
    
    output_dict = {}
    
    count = 0
    for i in Head:
        if i in count_no_head.index:
            count += int(count_no_head[i])
    if count != 0:
        output_dict['NPCHead'] = count
    else:
        output_dict['NPCHead'] = np.NaN

    count = 0
    for i in Body:
        if i in count_no_body.index:
            count += int(count_no_body[i])
        if count != 0:
            output_dict['NPCBody'] = count
        else:
            output_dict['NPCBody'] = np.NaN

    for i in Others:
        if i in count_no_others.index:
            output_dict[i] = int(count_no_others[i])
        else:
            output_dict[i] = np.NaN # 0也可以

    output_df = pd.DataFrame(pd.Series(output_dict), columns = [colname])
    return output_df





# 多个提取眼动数据注视点频次并算NPC注视点数总和 TODO # Exp_Areas尚未确认

def batch_Fon_count(input_path):
    result = pd.DataFrame(index = ['NPCHair', 'NPCForehead', 'NPCBrowLeft', 'NPCBrowRight', 'NPCEyeLeft', 'NPCEyeRight', 'NPCEye', 'NPCEarLeft', 'NPCEarRight', 'NPCNose', 'NPCMouth', 'NPCCheekLeft', 'NPCCheekRight', 'NPCFace', 'NPCBody', 'NPCLeftUpperarm', 'NPCRightUpperarm', 'NPCLeftForearm', 'NPCRightForearm', 'NPCLeftHand', 'NPCRightHand', 'NPCHip', 'NPCLeftThigh', 'NPCRightThigh', 'NPCLeftShank', 'NPCRightShank', 'NPCLeftFeet', 'NPCRightFeet', 'LeftHand', 'RightHand'])
    i = 1
    FileName_List = os.listdir(input_path) # 返回文件名列表
    for FileName in FileName_List: # 对于所有文件名
        InputPaths = os.path.join('%s/%s' % (input_path, FileName)) # 拼接路径
        if os.path.isfile(InputPaths): # 判断是否为文件还是文件夹。
            if InputPaths.endswith('.json'): # 判断文件扩展名是否为指定格式
                result = result.join(Fon_count_one(read_one(InputPaths), colname = i))
                i += 1
    result.loc['NPCsum'] = result.iloc[0:28,:].sum(axis = 0) # 这行后续要改动 TODO
    return result





# 多个提取眼动数据注视点频次，进行了一个包装 TODO # Exp_Areas尚未确认

def batch_Fon_agg(input_path):
    result = pd.DataFrame(index = ['NPCHead', 'NPCBody', 'LeftHand', 'RightHand'])
    i = 0
    FileName_List = os.listdir(input_path) # 返回文件名列表
    for FileName in FileName_List: # 对于所有文件名
        InputPaths = os.path.join('%s/%s' % (input_path, FileName)) # 拼接路径
        if os.path.isfile(InputPaths): # 判断是否为文件还是文件夹。
            if InputPaths.endswith('.json'): # 判断文件扩展名是否为指定格式
                result = result.join(Fon_one_agg(read_one(InputPaths), colname = i))
                i += 1
    return result





# 用于回车数据的预处理，输出带时间戳的矩阵、左通道数据集和右通道数据集

def read_eeg_txt(path, VREF = 4.8, range = [-2.4, 2.4]): # 单文件运算时间4.1s，仍有优化空间，暂时不做了 TODO
    volt_conver_coeff = VREF*1000000/(2**24-1) # 计算电压转换系数

    a = ['[', ']']
    raw = open(path, mode = 'r', encoding = 'utf-8')
    lst = raw.readlines()
    new_dic = {}
    for lines in lst:
        new_lines = lines.strip(u'\n')
        mat = new_lines.split("  ")
        for i in a:
            mat[1] = mat[1].strip(i)
        mat[1] = mat[1].split(', ')
        mat[1] = np.array(mat[1]).astype(int).tolist()
        new_dic[mat[0]] = mat[1]
    temp_df = pd.DataFrame.from_dict(new_dic, orient = 'index').reset_index() # 0-1 包序号，无意义。采样率250Hz（实际有波动，180-270个点均有可能）
    df = pd.DataFrame()
    df['index'] = temp_df['index']
    df['left1'] = temp_df.apply(lambda x: ((x[2]*2**16 + x[3]*2**8 + x[4]*2**0)/volt_conver_coeff-24), axis = 1) # 2-4 左通道采样点1
    df['left2'] = temp_df.apply(lambda x: ((x[8]*2**16 + x[9]*2**8 + x[10]*2**0)/volt_conver_coeff-24), axis = 1) # 8-10 左通道采样点2
    df['left3'] = temp_df.apply(lambda x: ((x[14]*2**16 + x[15]*2**8 + x[16]*2**0)/volt_conver_coeff-24), axis = 1) # 14-16 左通道采样点3
    df['right1'] = temp_df.apply(lambda x: ((x[5]*2**16 + x[6]*2**8 + x[7]*2**0)/volt_conver_coeff-24), axis = 1) # 5-7 右通道采样点1
    df['right2'] = temp_df.apply(lambda x: ((x[11]*2**16 + x[12]*2**8 + x[13]*2**0)/volt_conver_coeff-24), axis = 1) # 11-13 右通道采样点2
    df['right3'] = temp_df.apply(lambda x: ((x[17]*2**16 + x[18]*2**8 + x[19]*2**0)/volt_conver_coeff-24), axis = 1) # 17-19 右通道采样点3
    left, right = [], []
    for i in df.values:
        for j in [1, 2, 3]:
            left.append(i[j])
        for k in [4, 5, 6]:
            right.append(i[k])
    return df, left, right





















###############################################################
##################### 一些之前函数的存档 #######################
###############################################################

# 以下旧函数（对热图有一定调整）
# def json_2Dhm(json_filepath, background_filepath, z_len = [-3.302,3.395], y_len = [0,3.938], interest_vari = 'fp'): # Your json_filepath must be like: './paired_data/ASD/A1.json', your interest_vari must be like: 'fp'

#     def read_one(InputPaths):
#         content = []
#         raw = open(InputPaths, 'r', encoding='utf-8') # 读取文件
#         for line in raw.readlines():
#             if line.startswith(u'\ufeff'):
#                 line = line.encode('utf8')[3:].decode('utf8')
#             dict = json.loads(line)
#             content.append(dict)
#         df = pd.DataFrame(content)
#         return df

#     def scale(x, srcRange, dstRange):
#         return (x - srcRange[0]) * (dstRange[1] - dstRange[0]) / (srcRange[1] - srcRange[0]) + dstRange[0]

#     raw_json = read_one(json_filepath)[interest_vari]
#     background = mpimg.imread(background_filepath)
#     raw_json = raw_json.values.tolist()
#     z,y,z_f,y_f = [],[],[],[]

#     for dict in raw_json:
#         z.append(dict['Z'])
#         y.append(dict['Y'])
#     for elements in z:
#         elements_trans = z_len[1] - elements + z_len[0]
#         z_f.append(scale(elements_trans, z_len, [0, background.shape[1]]))
#     for elements in y:
#         y_f.append(scale(elements, y_len, [0, background.shape[0]]))

#     zy_df = pd.DataFrame({'z': z_f, 'y': y_f})
#     fig, ax = plt.subplots()
#     ax.imshow(background, extent = [0, background.shape[1], 0, background.shape[0]])
#     plt.scatter(zy_df['z'], zy_df['y'], c='r', marker='.', s = 10, alpha = 0.1)
#     return plt.show()