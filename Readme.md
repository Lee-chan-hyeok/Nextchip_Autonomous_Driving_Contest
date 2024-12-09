# AI반도체 기술인재 선발대회
주최 : 과학기술정보통신부 <br> 주관: 한국정보통신진흥협회(KAIT) <br> 참가 분야 : 모바일/엣지 <br> 참여 기업 : Nextchip <br> 기간: 2024-09-10 ~ 2024-12-03 <br> **`주제 : YOLO 모델 최적화 및 NPU 모델 탑재`**



## 모델 학습 프로세스
### 1. 데이터셋
Nextchip사에서 제공받은 train, valid, test set 사용

<img src="./images/dataset.png" alt="데이터셋 샘플">

### 2. 선정 모델
- YOLOv8s
    > YOLO 모델 아키텍처는 크게 Backbone, Neck, Head 3가지로 나뉜다.
    > - Backbone : 입력 이미지로부터 특성(Feature)을 추출하는 역할
    > - Neck : 이미지로부터 추출된 여러 크기의 특성(Feature)들을 결합하는 역할
    > - Head : Feature들을 바탕으로 Object의 위치를 찾는 역할

- 모델 네트워크 아키텍처
    <img src="./images/model_network.png" alt="데이터셋 샘플">

### 3. 최종 모델 학습
- 학습 방법
    - [Ultralytics](https://github.com/ultralytics/ultralytics)의 YOLOv8으로 모델 학습 <br>
    -> (모델 구조가 선언되어 있는 yaml 파일을 수정해가며 학습을 진행하기 위해 github에서 Ultralytics를 clone 받아서 사용) <br>

- Hyperparameter
    ```
    epochs=150 <br>
    save_period=5 <br>
    batch=32 <br>
    patience=8 <br>
    optimizer=auto(SGD) <br>
    lr0=0.01 <br>
    momentum=0.9 <br>
    ```

- 학습 코드
    ```python
    import sys
    sys.path.append('/github/Ultralytics/path')
    from ult import ultralytics
    from ultralytics import YOLO

    model_path = "/model_architecture/yaml_file/path"
    
    model = YOLO(model_path)
    model.train(
        data="/dataset/yaml_file/path",
        exist_ok=exist_ok,
        epochs=ep,
        save_period=save_period,
        batch=batch,
        project=result_dir,
        name=name.split('.')[0],
        patience=patience
    )
    ```

### 4. 학습 결과
- loss graph
    <img src="./images/results.png">

- PR Curve
    <img src="./images/PR_curve.png">


## 모델 Architecture 분석
