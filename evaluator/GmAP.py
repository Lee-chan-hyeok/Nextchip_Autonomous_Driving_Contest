import sys
sys.path.append(r'C:\Users\ihman\Desktop\NextChip\ino\edited_YOLO')

import os
import tarfile
import shutil
import ultralytics
from ultralytics import YOLO

# PATH
PT_PATH = r"../files/weights_files" # pt 경로
RESULT_PATH = r'../result/pred_result'
src = r'../../dataset/images/test'
output_path = r'../../output'

# CFG_PATH = r"../colab/cfg" # nextchip.yaml 경로
# VAL_PATH = r'../result/val_result'
# CSV_PATH = r'../documents/exp_list.csv'

def sample(cat, name):
    model = YOLO(f'{PT_PATH}/{cat}/{name}.pt')
    tar_path = rf'{RESULT_PATH}/{cat}'

    results = model.predict(source= src, iou= 0.5, save_txt= True, save_conf= True, project= output_path, name= name, conf= 0.001)

def make_pred(cat, name, re_exp= False):
    model = YOLO(f'{PT_PATH}/{cat}/{name}.pt')
    tar_path = rf'{RESULT_PATH}/{cat}'

    # 중복 검사
    if((re_exp == False) & (os.path.exists(f'{tar_path}/{name}.tar'))):
        print(f'{cat}에 {name}은 이미 잇성 히히')
        
        return
    
    elif((re_exp == True) & (os.path.exists(f'{tar_path}/{name}.tar'))):
        print(f'{cat}에 {name}은 이미 있는데 교체될거양 히히')

    results = model.predict(source= src, iou= 0.5, save_txt= True, save_conf= True, project= output_path, name= name, conf= 0.001, verbose= False)

    target_folder = f'{output_path}/{name}/labels'

    if not os.path.isdir(target_folder):
        raise ValueError(f"'{target_folder}'는 유효한 디렉터리가 아닙니다.")
    
    os.makedirs(tar_path, exist_ok= True)
    with tarfile.open(f'{tar_path}/{name}.tar', "w") as tar:
        tar.add(target_folder, arcname=name)
    print(f"'{output_path}/{name}'가 압축되었습니다.")

    shutil.rmtree(f'{output_path}/{name}')

def make_pred_by_dir(dir_name, re_exp= False):
    name_list = os.listdir(f'{PT_PATH}/{dir_name}')

    for exp_name in name_list:
        if(exp_name[-3:] == '.pt'):
            make_pred(dir_name, exp_name[:-3], re_exp= re_exp)

def pred_alllll(re_exp = False):
    folder_list = os.listdir(PT_PATH)

    for folder_name in folder_list:
        if(folder_name == 'undefined'):
            continue
        else:
            make_pred_by_dir(folder_name, re_exp= re_exp)