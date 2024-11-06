import os
import cv2
import glob

# global
cls_dict = {0: 'per',
            1: 'car',
            2: 'bus',
            3: 'tru',
            4: 'cyc',
            5: 'mot'}

# txt파일 경로를 받아 정보 받아오기
def get_info_from_txt(file_path):
    with open(file_path, 'r') as f:
        line_list = []
        for line in f:
            line = line[:-1].split(' ')
            line = [float(item) for item in line]
            line[0] = int(line[0])
            line_list.append(line)

        return line_list
    
# 박스를 center 기준 좌표 -> 좌상단, 우하단 기준 좌표
def coordinate_conveter(cx, cy, w, h, img_w, img_h):
    # center -> lower left, upper right
    x1 = int(img_w*(cx - w / 2))
    y1 = int(img_h*(cy - h / 2))
    x2 = int(img_w*(cx + w / 2))
    y2 = int(img_h*(cy + h / 2))
    return [x1, y1, x2, y2]

# iou와 gt_bbox_area를 계산
def get_IoU(true_box_center, pred_box_center):
    # 입력 영상
    size = (1280, 720)
    
    # center 좌표를 변환
    true_box_center.extend(size)
    pred_box_center.extend(size)
    true_bbox = coordinate_conveter(*true_box_center)
    pred_bbox = coordinate_conveter(*pred_box_center)

    # intesection 좌표
    inter_x1 = max(true_bbox[0], pred_bbox[0])
    inter_y1 = max(true_bbox[1], pred_bbox[1])
    inter_x2 = min(true_bbox[2], pred_bbox[2])
    inter_y2 = min(true_bbox[3], pred_bbox[3])

    true_bbox_area = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
    pred_bbox_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    inter_area = max(0, inter_x2 -inter_x1) * max(0, inter_y2 - inter_y1)

    iou = inter_area / (true_bbox_area + pred_bbox_area - inter_area)

    return iou, int(true_bbox_area)
    

# true.txt의 line과 pred.txt의 line들을 비교
def compare_line_to_line(line_true, line_pred, th):
    class_tf = False    # class 일치 여부
    iou_tf = False      # box 검출 여부
    iou = 0     # box 겹치는 정도
    section = 0 # 면적별 구간

    # class 일치 확인
    if(line_true[0] == line_pred[0]):
        class_tf = True
    else:
        class_tf = False
        #print('class is not correct!')

    # iou 체크
    iou, true_bbox_area = get_IoU(line_true[1:], line_pred[1:-1])
    #print('iou is ', iou, 'true_bbox is', true_bbox_area)
    
    # iou 기준으로 필터링
    if iou > 0:
        if iou > th:
            iou_tf = 'True'
        else:
            iou_tf = 'positive'

    else:
        iou_tf = 'not_exist'

    # conf는 pred.txt의 마지막 값
    conf = line_pred[-1]

    return [true_bbox_area, iou_tf, class_tf, iou, conf]

def ftf_by_true(true_file_path, pred_file_path, iou_th= 0.5, conf_th= 0.3):
    output = [-1, -1, -1, -1, -1]
    output_list = []

    # line_true는 true.txt 파일의 한 줄(객체 하나)
    for line_true in get_info_from_txt(true_file_path):
        #line_true = [float(item) for item in line_true]
        class_tf = False
        box_tf = False
        size = 0
        conf = 0
        iou = 0

        # pred.txt에서 iou > th를 만족하는 line의 리스트 생성
        filtered_line_list = []

        # line_pred는 pred.txt 파일의 객체 하나
        for line_pred in get_info_from_txt(pred_file_path):
            # str -> float
            #line_pred = [float(item) for item in line_pred]

            # 조건 만족하는 line 수집
            filtered_line_list.append(compare_line_to_line(line_true, line_pred, iou_th))

        # filtered_line_list에서 conf가 가장 높은것을 output으로
        best_conf = 0

        for line in filtered_line_list:
            if ((line[1] == 'True') & (line[2] == True)):
                conf_temp = line[-1]
                if (best_conf < conf_temp):
                    output = line # [size, iou_tf, class_tf, iou, conf]

        # 검출 된 경우 안된 경우
        # output_list.append(class, detect_tf, size, iou_tf, class_tf, iou, conf)
        if((output[1] == 'True') & (output[2] == True) & (conf_th < output[-1])):
        # iou 만족, class 만족, conf 만족
            output_list.append([cls_dict[line_true[0]], 'True', *output])

        elif((output[1] == 'positive') & (output[2] == True)):
        # iou 만족, class 만족, conf 미달
            output_list.append([cls_dict[line_true[0]], 'conf_lack', *output])
        
        elif((output[1] == 'positive') & (output[2] == True) & (conf_th < output[-1])):
        # iou 미달, class 만족, conf 만족
            output_list.append([cls_dict[line_true[0]], 'iou_lack', *output])
            # _, size = get_IoU(line_true[1:], line_true[1:])
            # output_list.append([cls_dict[line_true[0]], False, size, *output[1:]])

        elif((output[1] == 'positive') & (output[2] == True) & (conf_th > output[-1])):
        # iou 미달, class 만족, conf 미달
            output_list.append([cls_dict[line_true[0]], 'both_lack', *output])
            # _, size = get_IoU(line_true[1:], line_true[1:])
            # output_list.append([cls_dict[line_true[0]], False, size, *output[1:]])
        
        else:
        # d_tf = False -> iou, class 둘 중 하나라도 불만족
            output_list.append([cls_dict[line_true[0]], 'False', *output])
            # _, size = get_IoU(line_true[1:], line_true[1:])
            # output_list.append([cls_dict[line_true[0]], False, size, *output[1:]])

    return output_list

def ftf_by_pred(true_file_path, pred_file_path, iou_th= 0.5, conf_th= 0.3):
    output = [-1, -1, -1, -1, -1]
    output_list = []

    # line_pred는 pred.txt 파일의 한 줄(객체 하나)
    for line_pred in get_info_from_txt(pred_file_path):
        class_tf = False
        box_tf = False
        size = 0
        conf = 0
        iou = 0

        # pred.txt에서 검출된 line들의 list
        result_list = []

        # line_true는 true.txt 파일의 객체 하나
        for line_true in get_info_from_txt(true_file_path):
            result_list.append(compare_line_to_line(line_true, line_pred, iou_th))

        # filtered_line_list에서 conf가 가장 높은것을 output으로
        best_conf = 0

        for result in result_list:
            if ((result[1] == 'True') & (result[2] == True)):
                conf_temp = result[-1]
                if (best_conf < conf_temp):
                    output = result # [size, iou_tf, class_tf, iou, conf]

        # 검출 된 경우 안된 경우
        # output_list.append(class, detect_tf, size, iou_tf, class_tf, iou, conf)
        if((output[1] == 'True') & (output[2] == True) & (conf_th < output[-1])):
        # iou 만족, class 만족, conf 만족
            output_list.append([cls_dict[line_pred[0]], 'True', *output])

        elif((output[1] == 'positive') & (output[2] == True) & (conf_th < output[-1])):
        # iou 만족, class 만족, conf 미달
            output_list.append([cls_dict[line_pred[0]], 'conf_lack', *output])
        
        elif((output[1] == 'positive') & (output[2] == True) & (conf_th < output[-1])):
        # iou 미달, class 만족, conf 만족
            output_list.append([cls_dict[line_pred[0]], 'iou_lack', *output])
            # _, size = get_IoU(line_true[1:], line_true[1:])
            # output_list.append([cls_dict[line_true[0]], False, size, *output[1:]])

        elif((output[1] == 'positive') & (output[2] == True) & (conf_th > output[-1])):
        # iou 미달, class 만족, conf 미달
            output_list.append([cls_dict[line_pred[0]], 'both_lack', *output])
            # _, size = get_IoU(line_true[1:], line_true[1:])
            # output_list.append([cls_dict[line_true[0]], False, size, *output[1:]])
        
        else:
        # d_tf = False -> iou, class 둘 중 하나라도 불만족
            output_list.append([cls_dict[line_pred[0]], 'False', *output])
            # _, size = get_IoU(line_true[1:], line_true[1:])
            # output_list.append([cls_dict[line_true[0]], False, size, *output[1:]])

    return output_list