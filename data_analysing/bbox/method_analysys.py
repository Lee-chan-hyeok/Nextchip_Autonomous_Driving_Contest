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

def find_class_acc(df, cls_name):
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

def find_acc_by_size(df, cls_name, dt_condition, section= [460, 870, 1600, 6300, 921600]):
    df_class = df[df[f'class'] == cls_name]
    num = len(df_class)
    
    # section = [600, 1700, 6500, 921600]

    # size별로 구간 구분
    size0 = df_class[(df['size'] < section[0])]
    size1 = df_class[(section[0] <= df['size']) & (df['size'] < section[1])]
    size2 = df_class[(section[1] <= df['size']) & (df['size'] < section[2])]
    size3 = df_class[(section[2] <= df['size']) & (df['size'] < section[3])]
    size4 = df_class[section[3] <= df['size']]

    size0_acc = find_class_acc(size0, cls_name)[0]
    size1_acc = find_class_acc(size1, cls_name)[0]
    size2_acc = find_class_acc(size2, cls_name)[0]
    size3_acc = find_class_acc(size3, cls_name)[0]
    size4_acc = find_class_acc(size3, cls_name)[0]

    return [size0_acc, size1_acc, size2_acc, size3_acc, size4_acc]

def find_acc_by_cls_and_size(df, dt_condition, section= [600, 1700, 6500, 921600]):
    cls_list = list(cls_dict.values())
    class_df_list = []
    size_acc_list = []

    for cls in cls_list:
        class_df_list.append(df[df['class'] == cls])

    for idx in range(len(class_df_list)):
        size_acc = find_acc_by_size(class_df_list[idx], cls_list[idx], dt_condition, section)
        size_acc.insert(0, cls_list[idx])
        size_acc_list.append(size_acc)

    return size_acc_list

def find_ratio_by_cls_and_size(df, set_name, num_or_ratio= 'ratio', show= True, section= [460, 870, 1600, 6300]):
    cls_list = list(cls_dict.values())
    class_df_list = []
    size_ratio_list = []

    for cls in cls_list:
        class_df_list.append(df[df['class'] == cls])

    # class_df 별로 실행
    for idx in range(len(class_df_list)):
        class_df = class_df_list[idx]
        total = len(class_df)

        bar1 = len(class_df[df['size'] < section[0]])
        bar2 = len(class_df[(section[0] <= class_df['size']) & (class_df['size'] < section[1])])
        bar3 = len(class_df[(section[1] <= class_df['size']) & (class_df['size'] < section[2])])
        bar4 = len(class_df[(section[2] <= class_df['size']) & (class_df['size'] < section[3])])
        bar5 = len(class_df[section[3] <= class_df['size']])
        
        # 오류 검사
        if(total != bar1 + bar2 + bar3 + bar4 + bar5):
            print('num err!!!')

        size_ratio = [bar1/total, bar2/total, bar3/total, bar4/total, bar5/total]
        size_ratio = [round(item*100, 4) for item in size_ratio]

        size_ratio.insert(0, cls_list[idx])
        size_ratio_list.append(size_ratio)

        if(show):
            x = ['small_s', 'small_m', 'small_l', 'medium', 'large']
            
            y = [bar1, bar2, bar3, bar4, bar5]
            y_ratio = [bar1/total, bar2/total, bar3/total, bar4/total, bar5/total]
            y_ratio = [round(item*100, 4) for item in y_ratio]

            

            if(num_or_ratio == 'Num'):
                plt.bar(x, y, color='skyblue')
                plt.ylabel('Num')

            elif(num_or_ratio == 'Ratio'):
                plt.bar(x, y_ratio, color='skyblue')
                plt.ylabel('Ratio (%)')

            else:
                print('num_or_ratio error!!')
                return
            
            plt.xlabel('Box_size (pixel)')
            
            if(('Train' in set_name) | ('Test' in set_name)):
                plt.title(f'{set_name}_{cls_list[idx]}_BBox size_Histogram')
            else:
                plt.title(f'{cls_list[idx]}_BBox size_Histogram')
            
            # print(y)
            # print(y_ratio)
            plt.show()

    return size_ratio_list

def make_Detect_Acc(file_name, graph_name, show= True):
    csv_path = r'C:\Users\Ino\Desktop\NextChip\Minions_git\result\data_result'
    result_df = pd.read_csv(rf'{csv_path}\{file_name}.csv', index_col= 0)
    # display(result_df)

    per_df = result_df[result_df['class'] == 'per']
    car_df = result_df[result_df['class'] == 'car']
    bus_df = result_df[result_df['class'] == 'bus']
    tru_df = result_df[result_df['class'] == 'tru']
    cyc_df = result_df[result_df['class'] == 'cyc']
    mot_df = result_df[result_df['class'] == 'mot']

    all_acc = find_class_acc(result_df, 'all')[0]
    per_acc = find_class_acc(result_df, 'per')[0]
    car_acc = find_class_acc(result_df, 'car')[0]
    bus_acc = find_class_acc(result_df, 'bus')[0]
    tru_acc = find_class_acc(result_df, 'tru')[0]
    cyc_acc = find_class_acc(result_df, 'cyc')[0]
    mot_acc = find_class_acc(result_df, 'mot')[0]

    x = ['all', 'per', 'car', 'bus', 'tru', 'cyc', 'mot']
    y = [all_acc, per_acc, car_acc, bus_acc, tru_acc, cyc_acc, mot_acc]

    if(show == True):
        plt.bar(x, y)
        plt.title(f'Detect_Acc(%) by class [{graph_name}]')
    else:
        pass
    
    print(y)

    return x, y

def make_compare_graph(x, y1, y2, x_title, y_title, Title):
    labels = x

    # 각 막대 위치 조정
    x = np.arange(len(labels))
    width = 0.4

    # 그래프 그리기
    plt.bar(x - width/2, y1, width, label='v8s_P2G', color='skyblue')
    plt.bar(x + width/2, y2, width, label='v8s_org', color='salmon')

    # 라벨 및 제목 추가
    # plt.xlabel('Class')
    # plt.ylabel('Acc')
    # plt.title('Detect_Acc(%) by class')

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(Title)
    plt.xticks(x, labels)  # 새로운 x축 레이블 설정
    plt.legend()  # 범례 추가

    plt.show()