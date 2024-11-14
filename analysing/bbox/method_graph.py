import os
import numpy as np
import pandas as pd
import method_dataframe

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