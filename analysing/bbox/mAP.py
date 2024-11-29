import os
import numpy as np
from sklearn.metrics import average_precision_score

# 1. Define IoU calculation function
def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for two bounding boxes.
    box format: [x_min, y_min, x_max, y_max]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# 2. Categorize BBox size
def categorize_bbox_size(bbox, size_thresholds=(32, 96)):
    """
    Categorize bounding boxes by size (Small, Medium, Large).
    bbox: [x_min, y_min, x_max, y_max]
    size_thresholds: Tuple of area thresholds for categorization.
    """
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # Compute area
    if area < size_thresholds[0] ** 2:
        return "small"
    elif area < size_thresholds[1] ** 2:
        return "medium"
    else:
        return "large"

# 3. Evaluate predictions against ground truths
def evaluate_predictions(pred_file, gt_file, iou_threshold=0.5):
    """
    Evaluate mAP50 for each size category for a single image.
    pred_file: Path to prediction file.
    gt_file: Path to ground truth file.
    """
    predictions = np.loadtxt(pred_file, ndmin=2)  # Load predictions
    ground_truths = np.loadtxt(gt_file, ndmin=2)  # Load ground truths

    size_categories = {"small": [], "medium": [], "large": []}
    for gt in ground_truths:
        gt_size = categorize_bbox_size(gt[1:5])  # Extract BBox and categorize
        size_categories[gt_size].append(gt)
    
    category_ap = {}
    for size, gts in size_categories.items():
        preds = [p for p in predictions if categorize_bbox_size(p[1:5]) == size]
        ap = compute_ap(preds, gts, iou_threshold)
        category_ap[size] = ap
    return category_ap

# 4. Compute Average Precision (AP)
def compute_ap(predictions, ground_truths, iou_threshold):
    """
    Compute Average Precision for a specific category.
    """
    if not ground_truths:
        return 0.0
    
    preds_sorted = sorted(predictions, key=lambda x: x[5], reverse=True)
    matched = set()
    tp = []
    fp = []
    
    for pred in preds_sorted:
        max_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(ground_truths):
            if i in matched:
                continue
            iou = compute_iou(pred[1:5], gt[1:5])
            if iou > max_iou:
                max_iou = iou
                best_gt_idx = i
        
        if max_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched.add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
    recall = tp_cumsum / len(ground_truths)
    
    return average_precision_score(recall, precision)

# 5. Process all files
def process_all_files(pred_dir, gt_dir, iou_threshold=0.5):
    """
    Process all prediction and ground truth files.
    pred_dir: Directory containing prediction TXT files.
    gt_dir: Directory containing ground truth TXT files.
    """
    all_files = os.listdir(gt_dir)
    size_aps = {"small": [1600], "medium": [6300], "large": [921600]}

    for file in all_files:
        pred_file = os.path.join(pred_dir, file)
        gt_file = os.path.join(gt_dir, file)
        if not os.path.exists(pred_file):
            continue

        aps = evaluate_predictions(pred_file, gt_file, iou_threshold)
        for size in size_aps.keys():
            size_aps[size].append(aps[size])
    
    # Calculate average AP for each size category
    avg_aps = {size: np.mean(size_aps[size]) for size in size_aps.keys()}
    return avg_aps
