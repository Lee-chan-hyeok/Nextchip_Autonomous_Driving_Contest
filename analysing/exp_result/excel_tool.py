import sys
sys.path.append('C:/Users/ihman/Desktop/NextChip/sechan/JARVIS/ult') # ult 경로 설정
import os
import logging
import pandas as pd

from pathlib import Path
from ult import ultralytics
from ultralytics import YOLO
from torchinfo import summary

# PATH
PROJECT_DIR_PT = r"../../files/weights_files" # pt 경로
DATA_PATH = r"../../colab/cfg" # nextchip.yaml 경로
# PROJECT_DIR = r"C:\Users\ihman\Desktop\NextChip\sechan\JARVIS\result"  # pt, val, csv 상위 경로
VAL_PATH = r'../../result/val_result'

# LOG_DIR = os.path.join(PROJECT_DIR, "val") # val 경로
CSV_PATH = '../../documents/exp_list.csv'
# os.path.join(PROJECT_DIR, "csv_excel_file", "model_summary.csv") # csv 경로
# EXCEL_PATH = os.path.join(PROJECT_DIR, "csv_excel_file", "model_summary.xlsx") # excel 경로

def model_summary(model_path):
    # YOLO 모델 초기화
    model = YOLO(model_path)
    
    # 모델 파라미터 요약 정보 출력
    model_summary_info = summary(model.model, input_size=(1, 3, 640, 640), verbose=0)
    return model_summary_info

# pt파일로 validation 자동화
def val_model(category, model_name, model_path, data_path, csv_path, exist= True, split= 'test', first= False):
    train_set = model_name[:-4].split('_')[-1]
    
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

    # mAP50 all과 클래스별로 수집
    G_map50 = val_result.mean_results()[-2]
    
    map50_list = []
    for idx in range(6):
        map50_list.append(val_result.class_result(idx)[-2])

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
        df.loc[len(df)] = [model_name, 
                           params, 
                           GFLOPs, 
                           0, 
                           0, 
                           G_map50, 
                           *map50_list, # per ~ mot
                           train_set,
                           ]
        df.to_csv(CSV_PATH) # save
    else:
        df = pd.read_csv(CSV_PATH)
        
        # 행추가
        df.loc[len(df)] = [model_name, 
                           params, 
                           0,
                           GFLOPs, 
                           0, 
                           0, 
                           G_map50, 
                           *map50_list, # per ~ mot
                           train_set,
                           ]
        df.to_csv(CSV_PATH) # save

    return df