import os
import cv2
import glob
import numpy as np
import pandas as pd

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
    # center 좌표를 변환
    true_bbox = coordinate_conveter(*true_box_center, 1280, 720)
    pred_bbox = coordinate_conveter(*pred_box_center, 1280, 720)

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
    trues = [float(item) for item in trues]
    preds = [float(item) for item in preds]
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
    #print('iou is ', iou, 'true_bbox is', true_bbox_area)
    if iou > th:
        box_tf = True

    return [class_tf, box_tf, true_bbox_area]

def compare_file_to_file(true_file_path, pred_file_path):
    output = []
    for line_true in get_info_from_txt(true_file_path):
        line_ture = [float(item) for item in line_true]
        class_tf = False
        box_tf = False
        size = 0

        for line_pred in get_info_from_txt(pred_file_path):
            line_pred = [float(item) for item in line_pred]
            temp1, temp2, temp3 = compare_true_pred(line_true, line_pred, 0.5)
            if(temp1 == True & temp2 == True):
                class_tf, box_tf, size = temp1, temp2, temp3
                break

        if(class_tf == True & box_tf == True):
            output.append([line_true[0], True, size])
        else:
            output.append([line_true[0], False, 0])

    return output   # 각 line에 대한 결과값 리스트(2차원)


def get_acc(true_label_path, pred_label_path):
    area1 = []
    area2 = []
    area3 = []
    area4 = []

    true_txt_list = os.listdir(true_label_path)
    pred_txt_list = os.listdir(pred_label_path)

    # check files
    if(len(true_txt_list) == len(pred_txt_list)):
        print('num of files is same')
        for idx in range(len(true_txt_list)):
            if(true_txt_list[idx] == pred_txt_list[idx]):
                pass
            else:
                print(f'{idx+1}th file is different, true is {true_txt_list[idx]}, pred is {pred_txt_list[idx]}')
                return
    else:
        print('num of files is not same')
        return
    
    # box size 구간 구분
    size_th1 = 1000
    size_th2 = 10000
    size_th3 = 50000
    size_th4 = 100000
    
    column_name = ['file_name', 'class', 'detect_tf', 'box_size']
    result_df = pd.DataFrame(columns= column_name)

    for name in true_txt_list:
        out_list = compare_file_to_file(f'{true_label_path}/{name}', f'{pred_label_path}/{name}')
        for out in out_list:
            out.insert(0, name)
            result_df.loc[len(result_df)] = out

    return result_df