import os
import shutil
import argparse
import ultralytics
from ultralytics import YOLO

# 결과 저장 함수
def result_save_copy(result_dir, name):
    save_path = '/content'  # 압축할 경로
    zip_file = f'{save_path}/{name}'  # .zip 확장자는 자동으로 붙을 것임
    shutil.make_archive(zip_file, 'zip', result_dir)

    # 내 드라이브에 복사할 경로
    drive_path = f'/content/drive/MyDrive/Nextchip_result/{name.split(".")[0]}.zip'
    shutil.copy(f'{zip_file}.zip', drive_path)
    print(f'압축 파일이 Google Drive에 저장되었습니다: {drive_path}')

# 학습 및 저장 함수
def train_and_save(name, ep, save_period, batch, result_dir, exist_ok, data):
    model_path = f'/content/drive/MyDrive/Nextchip_cfg/{name}'
    if not ('.yaml' in name or '.pt' in name):
        print('너 바보, .yaml .pt 둘 다 아님')
        return

    model = YOLO(model_path)
    model.train(data=data, exist_ok=exist_ok, epochs=ep, save_period=save_period, batch=batch, project=result_dir, name=name.split('.')[0])

    # 결과를 압축할 폴더 경로 지정
    result_folder = f'{result_dir}/{name.split(".")[0]}'
    result_save_copy(result_folder, name)

# 데이터 압축 해제 함수
def unzip_data(file_name_list, output_dir_list):
    for dir in output_dir_list:
        os.makedirs(dir, exist_ok=True)
    for idx in range(len(file_name_list)):
        os.system("unzip " + file_name_list[idx] + " -d " + output_dir_list[idx])
        

file_name_list = [# nextchip dataset
                  # images
                  '/content/drive/MyDrive/Nextchip_dataset/images_train.zip',
                  '/content/drive/MyDrive/Nextchip_dataset/images_valid.zip',
                  '/content/drive/MyDrive/Nextchip_dataset/images_test.zip',
                  '/content/drive/MyDrive/Nextchip_dataset/images_add.zip',

                  # labels
                  '/content/drive/MyDrive/Nextchip_dataset/labels_train.zip',
                  '/content/drive/MyDrive/Nextchip_dataset/labels_valid.zip',
                  '/content/drive/MyDrive/Nextchip_dataset/labels_test.zip',
                  '/content/drive/MyDrive/Nextchip_dataset/labels_add.zip',

                  '/content/drive/MyDrive/Nextchip_dataset/path_txt.zip',]


output_dir_list = [# nextchip dataset
                   '/content/Nextchip_dataset/images/train',
                   '/content/Nextchip_dataset/images/valid',
                   '/content/Nextchip_dataset/images/test',
                   '/content/Nextchip_dataset/images/add',

                   '/content/Nextchip_dataset/labels/train',
                   '/content/Nextchip_dataset/labels/valid',
                   '/content/Nextchip_dataset/labels/test',
                   '/content/Nextchip_dataset/labels/add',


                   '/content/Nextchip_dataset',]

# Main 함수
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model with specific configurations.")
    parser.add_argument('--name', type=str, required=True, help='The name of the model file (.yaml or .pt)')
    parser.add_argument('--ep', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save_period', type=int, default=5, help='Interval at which models are saved')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--result_dir', type=str, default='/content/results', help='Directory to save results')
    parser.add_argument('--exist_ok', action='store_true', help='Allow existing directories')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file for training')

    args = parser.parse_args()
    
    train_and_save(
        name=args.name,
        ep=args.ep,
        save_period=args.save_period,
        batch=args.batch,
        result_dir=args.result_dir,
        exist_ok=args.exist_ok,
        data=args.data
    )