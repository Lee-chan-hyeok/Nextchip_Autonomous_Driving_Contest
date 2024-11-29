import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score
# from shapely.geometry import Polygon

def calculate_iou(bbox1, bbox2, img_width=1280, img_height=720):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Bboxes should be in the format (x_min, y_min, x_max, y_max), and normalized.
    """
    # 비정규화 (원본 이미지 크기 기준으로 계산)
    x1_min, y1_min, x1_max, y1_max = bbox1[0] * img_width, bbox1[1] * img_height, bbox1[2] * img_width, bbox1[3] * img_height
    x2_min, y2_min, x2_max, y2_max = bbox2[0] * img_width, bbox2[1] * img_height, bbox2[2] * img_width, bbox2[3] * img_height
    
    # Calculate intersection area
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    # Check for no overlap
    if xi_min >= xi_max or yi_min >= yi_max:
        return 0.0
    
    # Compute the area of intersection
    intersection_area = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Compute the area of both bounding boxes
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Compute the area of union
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Compute IoU
    return intersection_area / union_area

# mAP 계산 함수
def compute_map_by_size(pred_dir, gt_dir, num_classes=6, iou_threshold=0.5):
    size_aps = defaultdict(list)

    # 예측 및 Ground Truth 파일을 순차적으로 처리
    for i in range(6883):  # 파일 개수에 맞게 조정하세요
        pred_file = os.path.join(pred_dir, f"{i}.txt")
        gt_file = os.path.join(gt_dir, f"{i}.txt")

        # 예측 파일과 ground truth 파일 불러오기
        predictions = load_predictions(pred_file)
        ground_truths = load_ground_truth(gt_file)

        # 각 크기별 AP 계산
        for size in ['small', 'medium', 'large']:
            gts = [gt for gt in ground_truths if categorize_bbox_size(gt[1:5]) == size]
            preds = [pred for pred in predictions if categorize_bbox_size(pred[1:5]) == size]
            if len(gts) > 0:
                ap = compute_ap(preds, gts, num_classes, iou_threshold)
                if ap:
                    size_aps[size].append(np.mean(ap))
                else:
                    size_aps[size].append(0)

    # 크기별 평균 mAP 계산 (빈 값은 0으로 처리)
    mAP_by_size = {size: np.mean(aps) if len(aps) > 0 else 0 for size, aps in size_aps.items()}
    return mAP_by_size

# Ground truth 파일 로딩 함수 (예시, 각 텍스트 파일에서 bbox 정보를 읽어옵니다)
def load_ground_truth(gt_file):
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    ground_truths = []
    for line in lines:
        items = line.strip().split()
        class_id = int(items[0])
        bbox = list(map(float, items[1:]))
        ground_truths.append([class_id] + bbox)
    return ground_truths

# 예측 파일 로딩 함수 (prediction의 경우 형식이 동일하다고 가정)
def load_predictions(pred_file):
    with open(pred_file, 'r') as f:
        lines = f.readlines()
    predictions = []
    for line in lines:
        items = line.strip().split()
        class_id = int(items[0])
        bbox = list(map(float, items[1:5]))
        score = float(items[4])  # confidence score
        predictions.append([class_id] + bbox + [score])  # 클래스, bbox, confidence
    return predictions

# bbox 크기별로 분류하는 함수 (예시)
def categorize_bbox_size(bbox):
    img_width = 1280
    img_height = 720
    
    # bbox 좌표 (x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = bbox
    
    # bbox의 너비와 높이 계산
    width = (x_max - x_min)*1280
    height = (y_max - y_min)*720
    area = width * height
    
    # 이미지 기준으로 넓이에 따라 크기 분류
    if area < 1600:  # small
        return 'small'
    elif area < 6300:  # medium
        return 'medium'
    else:  # large
        return 'large'

# 평균 정확도(AP) 계산 함수
def compute_ap(preds, gts, num_classes, iou_threshold):
    ap_per_class = []

    for c in range(num_classes):
        y_true = []  # Ground truth for class 'c'
        y_score = []  # Prediction scores for class 'c'

        for gt in gts:
            if gt[0] == c:
                y_true.append(1)
            else:
                y_true.append(0)

        for pred in preds:
            if pred[0] == c:
                # Bounding box들 간의 IoU 계산
                iou = calculate_iou(gt[1:5], pred[1:5])
                if iou >= iou_threshold:
                    y_score.append(pred[4])  # Prediction confidence score (5th index)

        # Ensure both y_true and y_score have the same length
        if len(y_true) == 0 or len(y_score) == 0:
            continue  # Skip if no ground truths or predictions for this class
        
        try:
            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            # Compute Average Precision (AP) from precision-recall curve
            ap = average_precision_score(y_true, y_score)
            ap_per_class.append(ap)
        except ValueError:
            ap_per_class.append(0.0)  # If error occurs (e.g., no positive predictions or ground truths), return 0.0

    return ap_per_class