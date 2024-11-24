import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import method_dataframe
import method_analysys

def make_histo_list(gt_label_path):
    file_list = os.listdir(gt_label_path)
    area_list = []
    
    for name in file_list:
        with open(f'{gt_label_path}/{name}', 'r') as txt_file:
            for line in txt_file:
                a, b, c, w, h = line.split(' ')
                w = float(w)
                h = float(h)
                area = 1280*720*(w*h)
                area_list.append(area)

    print(len(area_list))
    return area_list

def rename_files(folder_path):
    check_name = '0000.txt'

    # folder내의 파일 목록 불러오기
    file_list = os.listdir(folder_path)

    # 0000.txt가 없으면 rename 과정 실행
    if check_name not in file_list:
        print(f'{check_name} is not in the folder, excute renaming')
        for name in file_list:
            num = int(name[:-4]) # .txt 제거
            src = os.path.join(folder_path, name)
            new_name = '{0:04d}.txt'.format(num - 1)
            dst = os.path.join(folder_path, new_name)

            os.rename(src, dst)
    else:
        print(f'{check_name} is in the folder, do not excute renaming')

# 클래스별 정확도, csv path들을 넣어주면 한꺼번에 비교
def compare_graph(x_ticks, y_data, labels, x_title= 'Class', y_title= 'Acc (%)', title= 'Acc by class'):    
    n = len(y_data)  # 데이터 세트의 개수
    num_classes = len(x_ticks)  # x축 레이블의 개수
    x_pos = np.arange(num_classes)  # x축 위치
    width = 0.8 / n  # 막대 너비 (막대 간 여유 공간 확보)

    # 그래프 그리기
    for i, y in enumerate(y_data):
        plt.bar(x_pos + (i - (n - 1) / 2) * width, y, width, label=labels[i])

    # 라벨 및 제목 추가
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.xticks(x_pos, x_ticks)  # x축 레이블 설정
    plt.legend()  # 범례 추가

    plt.show()

def compare_Acc_by_class(csv_path_list, x_title= 'Class', y_title= 'Acc (%)', title= 'Acc by class'):
    y_data = []
    labels = []

    for csv in csv_path_list:
        x, y = method_analysys.make_Detect_Acc_by_class(csv.split('\\')[0], csv.split('\\')[1], show= False)
        y_data.append(y)
        labels.append(csv.split('\\')[1])
    
    n = len(y_data)  # 데이터 세트의 개수
    num_classes = len(x)  # x축 레이블의 개수
    x_pos = np.arange(num_classes)  # x축 위치
    width = 0.8 / n  # 막대 너비 (막대 간 여유 공간 확보)

    # 그래프 그리기
    for i, y in enumerate(y_data):
        plt.bar(x_pos + (i - (n - 1) / 2) * width, y, width, label=labels[i])

    # 라벨 및 제목 추가
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.xticks(x_pos, x)  # x축 레이블 설정
    plt.legend()  # 범례 추가

    plt.show()

def acc_graph_by_csv_list(csv_list, conf= 0.3):
    y_list = []
    name_list = []
    
    for csv in csv_list:
        cat, name = csv.split('/')[-2:]
        name_list.append(name)

        x, y = method_analysys.make_Detect_Acc_by_class(cat, name, conf)
        y_list.append(y)

    compare_graph(x, y_list, name_list)

def size_acc_graph_by_csv_list(csv_list, conf= 0.3):
    for i in range(7):
        y_list = []
        name_list = []

        for csv in csv_list:
            cat, name = csv.split('/')[-2:]
            name_list.append(name)

            x, y = method_analysys.make_size_Acc_by_cls(cat, name, conf)
            y_list.append(y[i][1:])

        compare_graph(x, y_list, name_list, x_title= 'Box_size', title= f'Acc of {y[i][0]}_by_size')