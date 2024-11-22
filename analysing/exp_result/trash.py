def train_model(model_path, data_path, project_dir, model_name, epochs, batch_size):
    model = YOLO(model_path)
    model.train(data=data_path, epochs=epochs, project=project_dir, name=model_name, verbose=True, exist_ok=True, batch=batch_size)

    def validate_model(model_path, data_path, project_dir, model_name, split= 'test'):
    # split_test 폴더 경로 생성
    log_dir = os.path.join(project_dir, f"split_{split}")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = Path(log_dir) / f"{model_name}_test_log.txt"
    
    # logger 생성 및 설정
    logger = logging.getLogger(f"{model_name}_test")
    logger.setLevel(logging.INFO)
    
    # 핸들러 초기화 및 추가 (덮어쓰기 모드로 설정)
    handler = logging.FileHandler(log_filename, mode='w')
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    # 동일 파일 핸들러 중복 추가 방지
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_filename) for h in logger.handlers):
        logger.addHandler(handler)
    
    # 모델 초기화 및 유효성 검사 (split='test')
    model = YOLO(model_path)
    metrics = model.val(
        data=data_path,
        project=project_dir,
        name=model_name,
        exist_ok=True,
        split='test'
    )
    
    # 유효성 검사 결과 로그 저장
    logger.info("Validation results (test split): %s", metrics)
    
    # 모델 파라미터 수 추출 및 로그 저장
    model_summary_info = summary(model.model, verbose=0)
    total_params = model_summary_info.total_params
    logger.info("Model total parameters: %s", total_params)
    
    # 전체 mAP50 값을 로그에 저장
    overall_map50 = metrics.box.mean_results()[2]
    logger.info("all mAP50 = %.4f", overall_map50)
    
    # 클래스별 mAP50 값을 로그에 저장
    class_names = metrics.names.values()
    for i, name in enumerate(class_names):
        map50 = metrics.box.class_result(i)[2]
        logger.info("%s: mAP50 = %.4f", name, map50)

    print(f"Test validation log saved in {log_filename}")
    return log_filename

### valid 코드 ###
## split=test -> test 데이터로 valid 한 결과 ##
## split=valid -> valid 데이터로 valid 한 결과 ##
## 이거 돌리면 val/split_test, val/split_val 폴더 생성되고 각각 모델의 valid 로그 txt 파일로 저장 ##

from torchinfo import summary
from ultralytics import YOLO

import os



def validate_model_val(model_path, data_path, project_dir, model_name):
    # split_val 폴더 경로 생성
    log_dir = os.path.join(project_dir, "split_val")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = Path(log_dir) / f"{model_name}_val_log.txt"
    
    # logger 생성 및 설정
    logger = logging.getLogger(f"{model_name}_val")
    logger.setLevel(logging.INFO)
    
    # 핸들러 초기화 및 추가 (덮어쓰기 모드로 설정)
    handler = logging.FileHandler(log_filename, mode='w')
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    # 동일 파일 핸들러 중복 추가 방지
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_filename) for h in logger.handlers):
        logger.addHandler(handler)
    
    # 모델 초기화 및 유효성 검사 (split='val')
    model = YOLO(model_path)
    metrics = model.val(
        data=data_path,
        project=project_dir,
        name=model_name,
        exist_ok=True,
        split='val'
    )
    
    # 유효성 검사 결과 로그 저장
    logger.info("Validation results (val split): %s", metrics)
    
    # 모델 파라미터 수 추출 및 로그 저장
    model_summary_info = summary(model.model, verbose=0)
    total_params = model_summary_info.total_params
    logger.info("Model total parameters: %s", total_params)
    
    # 전체 mAP50 값을 로그에 저장
    overall_map50 = metrics.box.mean_results()[2]
    logger.info("all mAP50 = %.4f", overall_map50)
    
    # 클래스별 mAP50 값을 로그에 저장
    class_names = metrics.names.values()
    for i, name in enumerate(class_names):
        map50 = metrics.box.class_result(i)[2]
        logger.info("%s: mAP50 = %.4f", name, map50)

    print(f"Val validation log saved in {log_filename}")
    return log_filename

### graph 코드 인데 수정 해야됨 ###

import matplotlib.pyplot as plt
# Jupyter Notebook에서 그래프를 표시하도록 설정
%matplotlib inline

def plot_results_from_csv(data):
    plt.figure(figsize=(12, 8))
    unique_models = data['Model Name'].unique()
    scatter_points = []

    for model_name in unique_models:
        model_data = data[data['Model Name'] == model_name]
        fps_values = model_data['FPS']
        map50_test = model_data['Total mAP@0.5']
        map50_val = model_data['Val mAP@0.5']
        params_values = model_data['Parameters (M)']
        
        scatter = plt.scatter(fps_values, map50_test, map50_val, s=params_values*250, alpha=0.6, label=model_name)
        scatter_points.append((scatter, model_name))
        
        for fps, map50, params in zip(fps_values, map50_test, map50_val, params_values):
            print(f"Model: {model_name}, FPS: {fps}, Total mAP@0.5: {map50}, Parameters (M): {params}")
    
    plt.xlabel("FPS")
    plt.ylabel("Total mAP@0.5")
    plt.title("Model Comparison - FPS vs. Total mAP@0.5 (Point Size by Parameters)")
    plt.grid()
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.get_facecolor()[0], markersize=10) for scatter, _ in scatter_points]
    labels = [label for _, label in scatter_points]
    plt.legend(handles, labels, title="Models")
    
    plt.xlim(50, 300)
    plt.ylim(0, 1)
    
    plt.show()

### txt 로그 파일 읽어서 csv, excel 파일 생성 ###


import glob
import re
import pandas as pd
from pathlib import Path

def extract_logs(log_dir, csv_path, excel_path):
    # split_test와 split_val 폴더의 모든 *_log.txt 파일 검색
    test_log_files = glob.glob(f"{log_dir}/split_test/*_test_log.txt")
    val_log_files = glob.glob(f"{log_dir}/split_val/*_val_log.txt")
    data = []

    # split_test 로그 파일 처리
    for log_file in test_log_files:
        with open(log_file, 'r') as file:
            log_content = file.read()

            model_name = Path(log_file).stem.replace("_test_log", "")
            params_match = re.search(r"Model total parameters: (\d+)", log_content)
            total_params = round(int(params_match.group(1)) / 1e6, 4) if params_match else None

            class_map50 = {}
            class_map50_matches = re.findall(r"(\w+): mAP50 = ([\d.]+)", log_content)
            for match in class_map50_matches:
                class_name, map50_value = match
                class_map50[class_name] = round(float(map50_value), 4)

            total_map50_match = re.search(r"all mAP50 = ([\d.]+)", log_content)
            test_map50 = round(float(total_map50_match.group(1)), 4) if total_map50_match else None

            data.append({
                'Model Name': model_name, 
                'Parameters (M)': total_params, 
                'Total mAP@0.5': test_map50, 
                'Val mAP@0.5': None, 
                **class_map50, 
                'FPS': FPS
            })

    # split_val 로그 파일 처리 (total mAP50 값만 저장)
    for log_file in val_log_files:
        with open(log_file, 'r') as file:
            log_content = file.read()

            model_name = Path(log_file).stem.replace("_val_log", "")
            total_map50_match = re.search(r"all mAP50 = ([\d.]+)", log_content)
            val_map50 = round(float(total_map50_match.group(1)), 4) if total_map50_match else None

            # 해당 모델이 이미 data에 존재하는지 확인하여 업데이트 또는 추가
            for entry in data:
                if entry['Model Name'] == model_name:
                    entry['Val mAP@0.5'] = val_map50  # Val split에서의 total mAP50 값만 저장
                    break
            else:
                # 해당 모델이 data에 없으면 새 항목 추가 (Val split의 total mAP50 값만 저장)
                data.append({ 
                    'Val mAP@0.5': val_map50, 
                })

    # DataFrame 생성 및 저장
    df = pd.DataFrame(data).drop_duplicates(subset=['Model Name'], keep='last')
    df.to_csv(csv_path, index=False, float_format="%.4f")
    df.to_excel(excel_path, index=False, float_format="%.4f")
    print(f"CSV와 Excel 파일이 {csv_path} 및 {excel_path}에 저장되었습니다.")
    return df

df = extract_logs(LOG_DIR, CSV_PATH, EXCEL_PATH)