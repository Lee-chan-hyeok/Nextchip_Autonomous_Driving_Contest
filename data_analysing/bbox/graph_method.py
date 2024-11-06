import os
import numpy as np
import pandas as pd
import dataframe_method

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

def get_meta_df(true_label_path, pred_label_path, iou_th):
    area1 = []
    area2 = []
    area3 = []
    area4 = []

    true_txt_list = os.listdir(true_label_path)
    pred_txt_list = os.listdir(pred_label_path)

    # pred 파일 이름 변환
    rename_files(pred_label_path)
    pred_txt_list = os.listdir(pred_label_path)
    
    # check files
    if(len(true_txt_list) == len(pred_txt_list)):
        print('num of files is same')
        for idx in range(len(true_txt_list)):
            if(true_txt_list[idx] == pred_txt_list[idx]): # pred파일은 1부터 시작
                pass
            else:
                print(f'{idx+1}th file is different, true is {true_txt_list[idx]}, pred is {pred_txt_list[idx]}')
                return
    else:
        print(f'num of files error, true : {len(true_txt_list)}, pred : {len(pred_txt_list)}')
        return
    
    # box size 구간 구분
    size_th1 = 1000
    size_th2 = 10000
    size_th3 = 50000
    size_th4 = 100000
    
    column_name = ['file_name', 'class', 'detect_tf', 'size', 'iou_tf','class_tf',  'iou', 'conf']
    result_df = pd.DataFrame(columns= column_name)

    for name in true_txt_list:
        out_list = dataframe_method.compare_file_to_file(f'{true_label_path}/{name}', f'{pred_label_path}/{name}', iou_th)
        for out in out_list:
            out.insert(0, name)
            result_df.loc[len(result_df)] = out

    return result_df