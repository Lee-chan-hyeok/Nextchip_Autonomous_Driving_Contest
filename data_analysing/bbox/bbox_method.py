import os
import cv2
import glob

def get_info_from_txt(file_path):
    with open(file_path, 'r') as f:
        line_list = []
        for line in f:  
            line_list.append(line[:-1].split(' '))

        return line_list
    
def coordinate_conveter(cx, cy, w, h, img_w, img_h):
    # center -> lower left, upper right
    x1 = img_w*(cx - w / 2)
    y1 = img_h*(cy - h / 2)
    x2 = img_w*(cx + w / 2)
    y2 = img_h*(cy + h / 2)
    return [x1, y1, x2, y2]

def get_IoU(true_box_center, pred_box_center):
    # 입력 영상
    w = 1280
    h = 720
    
    # center 좌표를 변환
    true_box_center.extend([w, h])
    pred_box_center.extend([w, h])
    true_bbox = coordinate_conveter(*true_box_center)
    pred_bbox = coordinate_conveter(*pred_box_center)

    inter_x1 = max(true_bbox[0], pred_bbox[0])
    inter_y1 = max(true_bbox[1], pred_bbox[1])
    inter_x2 = min(true_bbox[2], pred_bbox[2])
    inter_y2 = min(true_bbox[3], pred_bbox[3])

    true_bbox_area = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
    pred_bbox_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    inter_area = max(0, inter_x2 -inter_x1) * max(0, inter_y2 - inter_y1)

    iou = inter_area / (true_bbox_area + pred_bbox_area - inter_area)

    # print('true_bbox_area is *****************\n', true_bbox_area)
    return iou, int(true_bbox_area)
    
def compare_true_pred(trues, preds, th):
    trues = [float(item) for item in trues]
    preds = [float(item) for item in preds]
    class_tf = False    # class 일치 여부
    iou_tf = False      # box 검출 여부
    iou = 0     # box 겹치는 정도
    section = 0 # 면적별 구간

    # class 일치 확인
    if(trues[0] == preds[0]):
        class_tf = True
    else:
        class_tf = False
        #print('class is not correct!')

    # iou 체크
    iou, true_bbox_area = get_IoU(trues[1:], preds[1:-1])
    #print('iou is ', iou, 'true_bbox is', true_bbox_area)
    if iou > th:
        iou_tf = True

    # conf는 pred.txt의 마지막 값
    conf = preds[-1]

    return [class_tf, iou_tf, true_bbox_area, iou, conf]

def compare_file_to_file(true_file_path, pred_file_path):
    output = []
    # line_true는 true.txt 파일의 한 줄(객체 하나)
    for line_true in get_info_from_txt(true_file_path):
        cls_dict = {'0': 'per',
                    '1': 'car',
                    '2': 'bus',
                    '3': 'tru',
                    '4': 'cyc',
                    '5': 'mot'}
        line_ture = [float(item) for item in line_true]
        class_tf = False
        box_tf = False
        size = 0
        conf = 0
        iou = 0

        # line_pred는 pred.txt 파일의 객체 하나
        for line_pred in get_info_from_txt(pred_file_path):
            line_pred = [float(item) for item in line_pred]
            # class_tf, box_tf, size
            temp1, temp2, temp3, iou, conf = compare_true_pred(line_true, line_pred, 0.5)
            if(temp1 == True & temp2 == True):
                class_tf, box_tf, size = temp1, temp2, temp3
                # 만족하는 것둘중에서 iou 높은것으로 고르게 수정
                break
            else:
                size = temp3

        # 검출 된 경우 안된 경우
        if(class_tf == True & box_tf == True):
            output.append([cls_dict[line_true[0]], True, size, iou, conf])
        else:
            output.append([cls_dict[line_true[0]], False, size, 0, 0])

    return output   # 각 line에 대한 결과값 리스트(2차원)

def check_result(gt_path, pred_path):
    gt_line = []
    pred_line = []
    gt_img_path = gt_path.replace('labels', 'images')[:-3] + 'jpg'
    gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)

    with open(gt_path, 'r') as gt:
        for line in gt:
            gt_line.append(line)
            cls, center_x, center_y, width, height = line.split(' ')
            center_x = float(center_x)
            center_y = float(center_y)
            width = float(width)
            height = float(height)
            #print(center_x, center_y, width, height)
            pt1 = (int(1280*(center_x - 0.5*width)), int(720*(center_y - 0.5*height)))
            pt2 = (int(1280*(center_x + 0.5*width)), int(720*(center_y + 0.5*height)))
            #print(pt1, pt2)
            cv2.rectangle(gt_img, pt1, pt2, (0, 0, 255), 4)

    pred_img_path = pred_path.replace('labels', 'images')[:-3] + 'jpg'
    with open(pred_path, 'r') as gt:
        for line in gt:
            pred_line.append(line)
            cls, center_x, center_y, width, height = line.split(' ')[:-1]
            center_x = float(center_x)
            center_y = float(center_y)
            width = float(width)
            height = float(height)
            #print(center_x, center_y, width, height)
            pt1 = (int(1280*(center_x - 0.5*width)), int(720*(center_y - 0.5*height)))
            pt2 = (int(1280*(center_x + 0.5*width)), int(720*(center_y + 0.5*height)))
            #print(pt1, pt2)
            cv2.rectangle(gt_img, pt1, pt2, (255, 0, 0), 1)

    # 10000size box sample, black line
    #cv2.rectangle(gt_img, (400, 500), (500, 600), (0, 0, 0), 2)
    
    cv2.imshow('gt_img', gt_img)
    cv2.waitKey()
    cv2.destroyAllWindows()