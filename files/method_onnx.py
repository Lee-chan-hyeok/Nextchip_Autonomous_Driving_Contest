from ultralytics import YOLO
import shutil
import os

PT_PATH = r"../files/weights_files" # pt 경로
CFG_PATH = r"../colab/cfg" # nextchip.yaml 경로

def make_onnx(category, model_name, pt_path= PT_PATH, re_exp= False):
    source = rf'{pt_path}/{category}/{model_name}.onnx'
    dst = source.replace('weights_files', 'onnx_files')
    
    # 중복 검사
    if((re_exp == False) & (os.path.exists(dst))):
        print(f'{category}에 {model_name}은 이미 잇성 히히')
        
        return
    
    elif((re_exp == True) & (os.path.exists(dst))):
        print(f'{category}에 {model_name}은 이미 있는데 교체될거양 히히')
    
    # 사용 데이터셋
    train_set = model_name.split('_')[-1]

    # 모델 선언
    model = YOLO(f'{pt_path}/{category}/{model_name}.pt')

    # 익쓰뽀뜨
    model.export(format= "onnx", 
                 nms= True, 
                 opset = 17,
                 data= rf'{CFG_PATH}/nextchip_{train_set}.yaml', 
                 imgsz= (384, 640)
                 )

    shutil.copy(source, dst)
    os.remove(source)
    



def onnx_all_by_dir(dir_name):
    name_list = os.listdir(f'{PT_PATH}/{dir_name}')

    for exp_name in name_list:
        if(exp_name[-3:] == '.pt'):
            make_onnx(dir_name, exp_name[:-3])

def onnx_allll():
    folder_list = os.listdir(PT_PATH)

    for folder_name in folder_list:
        if(folder_name == 'undefined'):
            continue
        else:
            onnx_all_by_dir(folder_name)