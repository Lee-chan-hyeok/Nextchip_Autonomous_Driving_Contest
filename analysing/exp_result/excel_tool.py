import sys
sys.path.append(r'C:\Users\ihman\Desktop\NextChip\ino\edited_YOLO') # ult 경로 설정

import os
import logging
import pandas as pd

from pathlib import Path
from ultralytics import YOLO
from torchinfo import summary

# PATH
PT_PATH = r"../../files/weights_files" # pt 경로
CFG_PATH = r"../../colab/cfg" # nextchip.yaml 경로
VAL_PATH = r'../../result/val_result'
CSV_PATH = r'../../documents/exp_list.csv'

def model_summary(model_path):
    # YOLO 모델 초기화
    model = YOLO(model_path)
    
    # 모델 파라미터 요약 정보 출력
    model_summary_info = summary(model.model, input_size=(1, 3, 640, 640), verbose=0)
    return model_summary_info

# pt파일로 validation 자동화
def val_model(category, model_name, data_path= CFG_PATH, csv_path= CSV_PATH, exist= True, split= 'test', first= False, model_path= PT_PATH, re_exp= False):
    if(first == False):
        # 중복 검사
        df = pd.read_csv(csv_path, index_col= 0)
        for idx in range(len(df)):
            if((model_name == df['Model'][idx]) & (re_exp == False)):
                print(f'{model_name}은 이미 잇성, 끝나버려, 히히히')
                
                return # 종료해버려
    
    train_set = model_name.split('_')[-1]
    
    # try val
    m = YOLO(rf'{model_path}/{category}/{model_name}.pt')
    val_result = m.val(
        data = rf'{data_path}/nextchip_{train_set}.yaml', 
        project = rf'{VAL_PATH}/{category}', 
        name = rf'{model_name}', 
        iou = 0.5, 
        task = 'detect', 
        exist_ok = exist, 
        split = split)
    
    # layers, params, GFLOPs 수집
    layers, params, _, GFLOPs = list(m.info())
    params = round(params/1000000, 2)
    GFLOPs = round(GFLOPs, 2)

    # mAP50 all과 클래스별로 수집
    G_map50 = round(val_result.mean_results()[-2] * 100, 2)
    
    map50_list = []
    for idx in range(6):
        map50_list.append(round(val_result.class_result(idx)[-2] * 100, 2))

    row = [model_name,
           params,
           0, 
           GFLOPs, 
           0, 
           0, 
           G_map50, 
           *map50_list, # per ~ mot
           train_set,
           ]

    if re_exp:
        df = pd.read_csv(csv_path, index_col= 0)

        for idx in range(len(df)):
            if(model_name == df['Model'][idx]):
                df.iloc[idx] = row
            
            return df

    if first:
        # 열 종류
        col_name = ['Model',
                    'params',
                    'FPS',
                    'GFLOPs',
                    'N_mAP / G_mAP (%)',
                    'N_mAP',
                    'G_mAP',
                    'per',
                    'car',
                    'bus',
                    'tru',
                    'cyc',
                    'mot',
                    'data_set',
                    ]
        df = pd.DataFrame(columns= col_name)

        # 행행행
        df.loc[len(df)] = row
        df.to_csv(csv_path) # save
    else:
        df = pd.read_csv(csv_path, index_col= 0)
        
        # 행추가
        df.loc[len(df)] = row
        df.to_csv(csv_path) # save

    return df

def val_all_by_dir(dir_name):
    name_list = os.listdir(f'{PT_PATH}/{dir_name}')

    for exp_name in name_list:
        if(exp_name[-3:] == '.pt'):
            val_model(dir_name, exp_name[:-3])

def val_allll():
    folder_list = os.listdir(PT_PATH)

    for folder_name in folder_list:
        if(folder_name == 'undefined'):
            continue
        else:
            val_all_by_dir(folder_name)

def extract_map50(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    lines = content.strip().split('\n')

    if len(lines) >= 3:  # 줄 수 확인
        # line3의 세 번째 값
        return float(lines[2].split('|')[2])  # 0-based index로 세 번째 값 선택
    return None  # 줄이 부족하면 None 반환

def edit_NmAP():
    df = pd.read_csv('../../documents/exp_list.csv', index_col= 0)

    