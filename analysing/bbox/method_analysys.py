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
def find_acc_by_size(df, cls_name, dt_condition= 'Detect', section= [460, 870, 1600, 6300, 921600]):
    if(cls_name == 'all'):
        df_class = df
    else:
        df_class = df[df['class'] == cls_name]
    
    num = len(df_class)
    
    # section = [600, 1700, 6500, 921600]

    # size별로 구간 구분
    size0 = df_class[(df['size'] < section[0])]
    size1 = df_class[(section[0] <= df['size']) & (df['size'] < section[1])]
    size2 = df_class[(section[1] <= df['size']) & (df['size'] < section[2])]
    size3 = df_class[(section[2] <= df['size']) & (df['size'] < section[3])]
    size4 = df_class[section[3] <= df['size']]

    size0_acc = find_acc_by_class(size0, cls_name)[0]
    size1_acc = find_acc_by_class(size1, cls_name)[0]
    size2_acc = find_acc_by_class(size2, cls_name)[0]
    size3_acc = find_acc_by_class(size3, cls_name)[0]
    size4_acc = find_acc_by_class(size4, cls_name)[0]

    return [size0_acc, size1_acc, size2_acc, size3_acc, size4_acc]

def find_acc_by_cls_and_size(df, dt_condition= '_', section= [460, 870, 1600, 6300, 921600]):
    cls_list = list(cls_dict.values())
    class_df_list = []
    size_acc_list = []

    # 전체에 대해 사이즈별 정확도
    size_acc_list.append(['all', *find_acc_by_size(df, 'all')])

    # 클래스별로 사이즈별 정확도
    for cls in cls_list:
        class_df_list.append(df[df['class'] == cls])

    for idx in range(len(class_df_list)):
        # 클래스에 대해 사이즈별 정확도
        size_acc = find_acc_by_size(class_df_list[idx], cls_list[idx], dt_condition, section)
        # 클래스명 추가
        size_acc.insert(0, cls_list[idx])
        size_acc_list.append(size_acc)

    return size_acc_list

# df에 대해 클래스 별로 사이즈별 히스토그램 출력
def find_size_ratio_by_cls(df, set_name, num_or_ratio= 'Ratio', show= True, section= [460, 870, 1600, 6300]):
    cls_list = list(cls_dict.values())
    class_df_list = [df]
    size_ratio_list = []

    for cls in cls_list:
        class_df_list.append(df[df['class'] == cls])

    # class_df 별로 실행
    for idx in range(len(class_df_list)):
        class_df = class_df_list[idx]
        total = len(class_df)

        bar1 = len(class_df[class_df['size'] < section[0]])
        bar2 = len(class_df[(section[0] <= class_df['size']) & (class_df['size'] < section[1])])
        bar3 = len(class_df[(section[1] <= class_df['size']) & (class_df['size'] < section[2])])
        bar4 = len(class_df[(section[2] <= class_df['size']) & (class_df['size'] < section[3])])
        bar5 = len(class_df[section[3] <= class_df['size']])
        
        # 오류 검사
        if(total != bar1 + bar2 + bar3 + bar4 + bar5):
            print('num err!!!')

        size_ratio = [bar1/total, bar2/total, bar3/total, bar4/total, bar5/total]
        size_ratio = [round(item*100, 4) for item in size_ratio]

        if(idx == 0):
            size_ratio.insert(0, 'all')
        else:
            size_ratio.insert(0, cls_list[idx - 1])
        size_ratio_list.append(size_ratio)

        if(show):
            x = ['small_s', 'small_m', 'small_l', 'medium', 'large']
            
            y = [bar1, bar2, bar3, bar4, bar5]
            y_ratio = [bar1/total, bar2/total, bar3/total, bar4/total, bar5/total]
            y_ratio = [round(item*100, 4) for item in y_ratio]

            

            if(num_or_ratio == 'Num'):
                plt.bar(x, y, color='royalblue')
                plt.ylabel('Num')

            elif(num_or_ratio == 'Ratio'):
                plt.bar(x, y_ratio, color='royalblue')
                plt.ylabel('Ratio (%)')

            else:
                print('num_or_ratio error!!')
                return
            
            plt.xlabel('Box_size (pixel)')
            
            if(('Train' in set_name) | ('Test' in set_name)):
                plt.title(f'{set_name}_{size_ratio[0]}_BBox size_Histogram')
            else:
                plt.title(f'{size_ratio[0]}_BBox size_Histogram')
            
            # print(y)
            # print(y_ratio)
            plt.show()

    return size_ratio_list

def make_Detect_Acc_by_class(category, exp_name, graph_name= '_', show= False):
    csv_path = r'..\..\result\data_result'
    result_df = pd.read_csv(rf'{csv_path}\{category}\{exp_name}.csv', index_col= 0)
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

def make_size_Acc_by_cls(category, exp_name, graph_name= '_', show= False):
    csv_path = r'..\..\result\data_result'
    result_df = pd.read_csv(rf'{csv_path}\{category}\{exp_name}.csv', index_col= 0)
    # display(result_df)

    # per_df = result_df[result_df['class'] == 'per']
    # car_df = result_df[result_df['class'] == 'car']
    # bus_df = result_df[result_df['class'] == 'bus']
    # tru_df = result_df[result_df['class'] == 'tru']
    # cyc_df = result_df[result_df['class'] == 'cyc']
    # mot_df = result_df[result_df['class'] == 'mot']

    all_acc = find_acc_by_cls_and_size(result_df)
    per_acc = find_acc_by_cls_and_size(result_df)
    car_acc = find_acc_by_cls_and_size(result_df)
    bus_acc = find_acc_by_cls_and_size(result_df)
    tru_acc = find_acc_by_cls_and_size(result_df)
    cyc_acc = find_acc_by_cls_and_size(result_df)
    mot_acc = find_acc_by_cls_and_size(result_df)

    x = ['all', 'per', 'car', 'bus', 'tru', 'cyc', 'mot']
    y = [all_acc, per_acc, car_acc, bus_acc, tru_acc, cyc_acc, mot_acc]

    if(show == True):
        plt.bar(x, y, color='salmon')
        plt.title(f'Detect_Acc(%) by class [{graph_name}]')
    else:
        pass
    
    # print(y)

    return x, y