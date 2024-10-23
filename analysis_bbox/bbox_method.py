import os
import cv2
import glob
import numpy as np

def get_info_from_txt(file_path):
    with open(file_path, 'r') as f:
        line_list = []
        for line in f:  
            line_list.append(line[:-1].split(' '))

        return line_list
    
def coordinate_conveter(cx, cy, w, h):
    # center -> upper left, lower right
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def get_IoU(true_box_center, pred_box_center):
    # center 좌표를 변환
    true_bbox = coordinate_conveter(true_box_center)
    pred_bbox = coordinate_conveter(pred_box_center)

    inter_x1 = max(true_bbox[0], pred_bbox[0])
    inter_y1 = max(true_bbox[1], pred_bbox[1])
    inter_x2 = min(true_bbox[2], pred_bbox[2])
    inter_y2 = min(true_bbox[3], pred_bbox[3])

    true_bbox_area = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
    pred_bbox_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    inter_area = max(0, inter_x2 -inter_x1) * max(0, inter_y2 - inter_y1)

    iou = inter_area / (true_bbox_area + pred_bbox_area - inter_area)
    return iou, true_bbox_area
    
def compare_true_pred(trues, preds, th):
    class_tf = False    # class 일치 여부
    box_tf = False      # box 검출 여부
    iou = 0     # box 겹치는 정도
    section = 0 # 면적별 구간

    # class 일치 확인
    if(trues[0] == preds[0]):
        class_tf = True
    else:
        class_tf = False
        #print('class is not correct!')

    # iou 체크
    iou, true_bbox_area = get_IoU(trues[1:], preds[1:])
    if iou > th:
        box_tf = True

    return [class_tf, box_tf, true_bbox_area]

def get_acc(true_label_path, pred_label_path):
    section1 = []
    section2 = []
    section3 = []
    section4 = []

    true_txt_list = glob.glob(true_label_path + '/*.txt')
    pred_txt_list = glob.glob(pred_label_path + '/*.txt')

    # check files
    if(len(true_txt_list) == len(pred_label_path)):
        print('num of files is same')
        for idx in range(len(true_txt_list)):
            if(true_txt_list[idx] == pred_txt_list[idx]):
                pass
            else:
                print(f'{idx+1}th file is different, {true_txt_list[idx].split(' ')[-1]}, {pred_txt_list[idx].split(' ')[-1]}')
                return
    else:
        print('num of files is not same')
        return
    
    area_th1 = 10
    area_th2 = 100
    area_th3 = 1000
    area_th4 = 10000
    
    for idx in range(len(true_txt_list)):
        for line_true in get_info_from_txt(true_txt_list[idx]):
            for line_pred in get_info_from_txt(pred_txt_list[idx]):
                class_tf, box_tf, area = compare_true_pred(line_true, line_pred, 0.5)
                
                # box 크기별
                if(class_tf == True & box_tf == True):
                    if area <= area_th1:
                        pass
                    elif area_th1 < area <= area_th2:
                        pass
                    elif area_th2 < area <= area_th3:
                        pass
                    elif area_th3 < area:
                        pass