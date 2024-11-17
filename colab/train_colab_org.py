import os
import shutil
import ultralytics
from ultralytics import YOLO

# 결과 저장 함수

def result_save_copy(result_dir, name):
    save_path = '/content'  # 압축할 경로

    # 압축 (이때 .zip 확장자가 자동으로 붙으므로 확장자 중복 방지)
    zip_file = f'{save_path}/{name}'  # .zip 확장자는 자동으로 붙을 것임
    shutil.make_archive(zip_file, 'zip', result_dir)

    # 내 드라이브에 복사할 경로
    drive_path = f'/content/drive/MyDrive/Nextchip_result/{name.split(".")[0]}.zip'

    # Google Drive로 복사
    shutil.copy(f'{zip_file}.zip', drive_path)

    print(f'압축 파일이 Google Drive에 저장되었습니다: {drive_path}')

def train_and_save(name, ep, save_period, batch, result_dir, exist_ok):

    if('.yaml' in name):
        model = YOLO(f'/content/drive/MyDrive/Nextchip_cfg/{name}')
    elif('.pt' in name):
        model = YOLO(f'/content/drive/MyDrive/Nextchip_cfg/{name}')
    else:
        print('너 바보, .yaml .pt 둘 다 아님')


    model.train(data='/content/drive/MyDrive/Nextchip_cfg/nextchip_colab_add.yaml', exist_ok= exist_ok, epochs= ep, save_period= save_period, batch= batch, project= result_dir, name= name.split('.')[0])

    # 결과를 압축할 폴더 경로 지정
    result_folder = f'{result_dir}/{name.split(".")[0]}'  # 훈련된 모델 결과 폴더

    result_save_copy(result_folder, name)

def unzip_data(file_name_list, output_dir_list):
  for dir in output_dir_list:
    os.makedirs(dir, exist_ok = True)
  for idx in range(len(file_name_list)):
    os.system("unzip " + file_name_list[idx] + " -d" + output_dir_list[idx])

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

if __name__ == "__main__":