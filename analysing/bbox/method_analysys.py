import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# global
cls_dict = {0: 'per',
            1: 'car',
            2: 'bus',
            3: 'tru',
            4: 'cyc',
            5: 'mot'}

# 데이터셋의 클래스별 비율
def ratio_by_cls(df, data_set, num_or_ratio = 'Ratio', show= False):
    total_num = len(df)

    num_list = []
    for cls in cls_dict.values():
        temp = df[df['class'] == cls]
        num_list.append(len(temp))

    if(num_or_ratio == 'Ratio'):
        num_list = [round((item/total_num)*100, 2) for item in num_list]

    if(show):
        x_labels = cls_dict.values()  # x축 레이블
        # plt.figure(figsize=(10, 6))
        plt.bar(x_labels, num_list, color='royalblue', edgecolor='royalblue')

        plt.xlabel('Class')
        plt.ylabel('Number' if num_or_ratio != 'Ratio' else 'Ratio (%)', rotation= 0)
        if(num_or_ratio == 'Ratio'):
            title = f'Instance Ratio by Class : {data_set} '
        elif(num_or_ratio == 'Num'):
            title = f'Instance Num by Class : {data_set}'
        plt.title(title)
        # plt.xticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        return num_list
    
    else:
        return num_list
    
# 데이터셋의 사이즈별 히스토그램을 클래스 종류별로 출력
def size_ratio_by_cls(df, data_set, num_or_ratio= 'Ratio', obj_num= 3, show= True):
    if(obj_num == 4):
        section= [0, 1600, 6300, 921600]
    elif(obj_num == 6):
        section= [0, 460, 870, 1600, 6300, 921600]
    elif(obj_num == 3):
        section= [0, 460, 870, 1600]
        
    cls_list = list(cls_dict.values())
    class_df_list = [df]
    size_ratio_list = []

    for cls in cls_list:
        class_df_list.append(df[df['class'] == cls])

    # class_df 별로 실행
    for idx in range(len(class_df_list)):
        class_df = class_df_list[idx]
        total = len(class_df)

        bar_list = []
        for i, item in enumerate(section[:-1]):
            bar = len(class_df[(item <= class_df['size']) & (class_df['size'] < section[i+1])])
            bar_list.append(bar)
        
        # 오류 검사
        if(total != sum(bar_list)):
            print('num err!!!')

        size_ratio = [round((item/total)*100, 2) for item in bar_list]
        if(idx == 0):
            size_ratio.insert(0, '')
        else:
            size_ratio.insert(0, cls_list[idx - 1])

        size_ratio_list.append(size_ratio)

        if(show):
            if(obj_num == 3):
                x = ['small_s', 'small_m', 'small_l']
            elif(obj_num == 4):
                x = ['small', 'medium', 'large']
            elif(obj_num == 6):
                x = ['small_s', 'small_m', 'small_l', 'medium', 'large']
            else:
                pass
            
            y = bar_list
            if(num_or_ratio == 'Ratio'):
                y = [round((item/total)*100, 2) for item in y]
            
            plt.bar(x, y, color='royalblue')
            plt.xlabel('Object size (pixel)')
            plt.ylabel('Num' if num_or_ratio == 'Num' else 'Ratio (%)', rotation= 0)
            
            if(('Train' in data_set) | ('Valid' in data_set) | ('Test' in data_set)):
                plt.title(f'{size_ratio[0]} Histogram by size : {data_set}')
            else:
                plt.title(f'{size_ratio[0]} Histogram by size')

            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.show()

    return size_ratio_list

# 한 df에 대해 정확도, cls 지정 가능
def find_acc_by_class(df, cls_name):
    if(cls_name == 'all'):
        df_class = df
    else:
        df_class = df[df[f'class'] == cls_name]

    num = len(df_class)

    detect = len(df_class[df_class['dt_condition'] == 'Detect'])
    conf_lack = len(df_class[df_class['dt_condition'] == 'conf_lack'])
    detect_clsF = len(df_class[df_class['dt_condition'] == 'Detect_clsF'])
    
    acc1 = round((detect/num)*100, 2)
    acc2 = round((conf_lack/num)*100, 2)
    acc3 = round((detect_clsF/num)*100, 2)

    return [acc1, acc2, acc3]

# 한 df에 대해 사이즈별로 정확도, cls 지정 가능
def find_acc_by_size(df, cls_name, section, dt_condition= '_'):
    if(cls_name == 'all'):
        df_class = df
    else:
        df_class = df[df['class'] == cls_name]

    size_acc_list = [find_acc_by_class(df_class, cls_name)[0]]

    for i, item in enumerate(section[:-1]):
        size_df = df_class[(item <= df['size']) & (df['size'] < section[i+1])]
        size_acc = find_acc_by_class(size_df, cls_name)[0]
        size_acc_list.append(size_acc)

    return size_acc_list

def find_acc_by_cls_and_size(df, dt_condition= '_', obj_num= 3):
    if(obj_num == 4):
        section= [0, 1600, 6300, 921600]
    elif(obj_num == 6):
        section= [0, 460, 870, 1600, 6300, 921600]
    elif(obj_num == 3):
        section= [0, 460, 870, 1600]

    cls_list = list(cls_dict.values())
    class_df_list = []
    size_acc_list = []

    size_acc_list.append(['all', *find_acc_by_size(df, 'all', section= section)])

    # 클래스별로 사이즈별 정확도
    for cls in cls_list:
        class_df_list.append(df[df['class'] == cls])

    for idx in range(len(class_df_list)):
        # 클래스에 대해 사이즈별 정확도
        size_acc = find_acc_by_size(class_df_list[idx], cls_list[idx], section= section)
        # 클래스명 추가
        size_acc.insert(0, cls_list[idx])
        size_acc_list.append(size_acc)

    return size_acc_list

def make_Detect_Acc_by_class(category, exp_name, conf= 0.3, graph_name= '_', show= False):
    csv_path = r'..\..\result\data_result'
    result_df = pd.read_csv(rf'{csv_path}\{category}\{exp_name}_{conf}.csv', index_col= 0)
    # display(result_df)

    per_df = result_df[result_df['class'] == 'per']
    car_df = result_df[result_df['class'] == 'car']
    bus_df = result_df[result_df['class'] == 'bus']
    tru_df = result_df[result_df['class'] == 'tru']
    cyc_df = result_df[result_df['class'] == 'cyc']
    mot_df = result_df[result_df['class'] == 'mot']

    all_acc = find_acc_by_class(result_df, 'all')[0]
    per_acc = find_acc_by_class(result_df, 'per')[0]
    car_acc = find_acc_by_class(result_df, 'car')[0]
    bus_acc = find_acc_by_class(result_df, 'bus')[0]
    tru_acc = find_acc_by_class(result_df, 'tru')[0]
    cyc_acc = find_acc_by_class(result_df, 'cyc')[0]
    mot_acc = find_acc_by_class(result_df, 'mot')[0]

    x = ['all', 'per', 'car', 'bus', 'tru', 'cyc', 'mot']
    y = [all_acc, per_acc, car_acc, bus_acc, tru_acc, cyc_acc, mot_acc]

    if(show == True):
        plt.bar(x, y, color='salmon')
        plt.title(f'Detect_Acc(%) by class [{graph_name}]')
    else:
        pass
    
    # print(y)

    return x, y

def make_size_Acc_by_cls(category, exp_name, conf= 0.5, obj_num= 3, graph_name= '_', show= False):
    csv_path = r'..\..\result\data_result'
    result_df = pd.read_csv(rf'{csv_path}\{category}\{exp_name}_{conf}.csv', index_col= 0)
    
    if(obj_num == 4):
        x = ['whole', 'small', 'medium', 'large']
    elif(obj_num == 3):
        x = ['small_s', 'small_m', 'small_l']
    elif(obj_num == 6):
        x = ['whole', 'small_s', 'small_m', 'small_l', 'medium', 'large']
    else:
        print('obj_num은 3, 4, 6이야 멍청아')
    
    y_list = find_acc_by_cls_and_size(result_df, obj_num= obj_num)
    
    for y in y_list:
        if(show == True):
            plt.bar(x, y[1:], color='salmon')
            plt.title(f'Detect_Acc(%) by class [{y[0]}]')
            plt.show()
        else:
            pass

    return x, y_list