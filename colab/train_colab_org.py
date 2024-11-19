import os
import shutil
import argparse
import ultralytics
from ultralytics import YOLO

# 결과 저장 함수
def result_save_copy(result_dir, name):
    save_path = '/content'  # 압축할 경로
    zip_file = f'{save_path}/{name}'  # .zip 확장자는 자동으로 붙을 것임
    shutil.make_archive(zip_file[:-4], 'zip', result_dir)

    # best.pt를 name과 일치하게 변경
    src = os.path.join(f'{result_dir}/weights/best.pt')
    dst = os.path.join(f'{result_dir}/weights/{name}.pt')
    os.rename(src, dst)

    # 변경한 파일을 git레포에 복사
    shutil.copy(f'{result_dir}/weights/{name}.pt', f'Minions/files/weights_files/{name}.pt')

    # 내 드라이브에 복사할 경로
    drive_path = f'/content/drive/MyDrive/Nextchip_result/{name.split(".")[0]}.zip'
    shutil.copy(f'{zip_file}s', drive_path)
    print(f'압축 파일이 Google Drive에 저장되었습니다: {drive_path}')


# 학습 및 저장 함수
def train_and_save(name, ep, save_period, batch, result_dir, exist_ok, data):
    model_path = f'/content/Minions/files/yaml_files/{name}'
    if not ('.yaml' in name or '.pt' in name):
        print('너 바보, .yaml .pt 둘 다 아님')
        return

    # git에서 txt들 복사
    shutil.copy(f'Minions/colab/cfg/train_sample.txt', f'/content/Nextchip_dataset/train_sample.txt')
    shutil.copy(f'Minions/colab/cfg/train_t.txt', f'/content/Nextchip_dataset/train_t.txt')
    shutil.copy(f'Minions/colab/cfg/train_ta.txt', f'/content/Nextchip_dataset/train_ta.txt')
    shutil.copy(f'Minions/colab/cfg/train_tv.txt', f'/content/Nextchip_dataset/train_tv.txt')
    shutil.copy(f'Minions/colab/cfg/valid.txt', f'/content/Nextchip_dataset/valid.txt')
    shutil.copy(f'Minions/colab/cfg/test.txt', f'/content/Nextchip_dataset/test.txt')

    model = YOLO(model_path)
    model.train(data=rf'/content/Minions/colab/cfg/colab_{data}.yaml', exist_ok=exist_ok, epochs=ep, save_period=save_period, batch=batch, project=result_dir, name=name.split('.')[0])

    # 결과를 압축할 폴더 경로 지정
    result_folder = f'{result_dir}/{name.split(".")[0]}'
    result_save_copy(result_folder, name)



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