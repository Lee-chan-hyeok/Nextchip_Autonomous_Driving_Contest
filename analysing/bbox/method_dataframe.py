import os
import cv2
import glob
import shutil
import tarfile
import pandas as pd
import method_graph

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
def coordinate_conveter(cx, cy, w, h, size):
    img_w = size[0]
    img_h = size[1]
    
    # center -> lower left, upper right
    x1 = int(img_w*(cx - w / 2))
    y1 = int(img_h*(cy - h / 2))
    x2 = int(img_w*(cx + w / 2))
    y2 = int(img_h*(cy + h / 2))

    return [x1, y1, x2, y2]

def get_size(center_x, center_y, w, h, img_size):    
    size = (img_size[0]*w) * (img_size[1]*h)

    return size

# iou와 gt_bbox_area를 계산
def get_IoU(true_box_center, pred_box_center, size= (1280, 720)):
    # center 좌표를 변환
    #true_box_center.extend(size)
    #pred_box_center.extend(size)
    true_bbox = coordinate_conveter(*true_box_center, size)
    pred_bbox = coordinate_conveter(*pred_box_center, size)

    # intesection 좌표
    inter_x1 = max(true_bbox[0], pred_bbox[0])
    inter_y1 = max(true_bbox[1], pred_bbox[1])
    inter_x2 = min(true_bbox[2], pred_bbox[2])
    inter_y2 = min(true_bbox[3], pred_bbox[3])

    true_bbox_area = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
    pred_bbox_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    inter_area = max(0, inter_x2 -inter_x1) * max(0, inter_y2 - inter_y1)

    iou = inter_area / (true_bbox_area + pred_bbox_area - inter_area)

    return iou
    #return iou, int(true_bbox_area)
    

# true.txt의 line과 pred.txt의 line을 비교
def compare_line_to_line(line_true, line_pred, iou_th, size= (1280, 720)):
    class_tf = False    # class 일치 여부
    iou_tf = False      # box 검출 여부
    iou = 0     # box 겹치는 정도
    section = 0 # 면적별 구간

    # class 일치 확인
    if(line_true[0] == line_pred[0]):
        class_tf = True
    else:
        class_tf = False

    # iou 체크
    iou = get_IoU(line_true[1:], line_pred[1:-1])
    true_bbox_size = get_size(*line_true[1:], size)
    
    # iou 기준으로 필터링
    if iou > 0:
        if iou > iou_th:
            iou_tf = 'True'
        else:
            iou_tf = 'positive'

    else:
        iou_tf = 'not_exist'

    # conf는 pred.txt의 마지막 값
    conf = line_pred[-1]

    return [int(true_bbox_size), iou_tf, class_tf, cls_dict[line_pred[0]], iou, conf]

def ftf_by_true(true_file_path, pred_file_path, iou_th= 0.5, conf_th= 0.3):
    detect_condition_list = []

    # line_true는 true.txt 파일의 한 줄(객체 하나)
    for line_true in get_info_from_txt(true_file_path):
        class_tf = False
        box_tf = False
        size = 0
        conf = 0
        iou = 0

        # pred.txt에서 iou > th를 만족하는 line의 리스트 생성

        compare_list = []
        # line_pred는 pred.txt 파일의 객체 하나
        for line_pred in get_info_from_txt(pred_file_path):
            # 조건 만족하는 line 수집
            compare_list.append(compare_line_to_line(line_true, line_pred, iou_th))

        check_list = []
        iou_tf_list = [item[1] for item in compare_list]

        detect_condition = []

        if(('True' not in iou_tf_list) & ('positive' not in iou_tf_list)):
            size = get_size(*line_true[1:], (1280, 720))
            detect_condition = [cls_dict[line_true[0]], 'False', size, 'not_exist', 0, 0, -1, -1]

        elif(('True' not in iou_tf_list) & ('positive' in iou_tf_list)):
            # best_conf = 0            
            # 체크할 인덱스 수집
            for idx in range(len(compare_list)):
                item = compare_list[idx]
                if(item[1] == 'positive'):
                    check_list.append(int(idx))
            
            cls_tf_list = []
            for idx in check_list:
                item = compare_list[idx]
                cls_tf_list.append(item[2])

            if(True in cls_tf_list):
                best_conf = 0
                for idx in check_list:
                    line = compare_list[idx]
                    if((line[2] == True) & (best_conf < line[-1])):
                        best_conf = line[-1]
                        detect_condition = [cls_dict[line_true[0]], 'pos_clsT', *line]
            else:
                for idx in check_list:
                    best_conf = 0
                    line = compare_list[idx]
                    if(best_conf < line[-1]):
                        best_conf = line[-1]
                        detect_condition = [cls_dict[line_true[0]], 'pos_clsF', *line]

        elif('True' in iou_tf_list):
            # best_conf = 0

            # 체크할 인덱스 수집
            for idx in range(len(compare_list)):
                item = compare_list[idx]
                if(item[1] == 'True'):
                    check_list.append(int(idx))
            
            cls_tf_list = []
            for idx in check_list:
                item = compare_list[idx]
                cls_tf_list.append(item[2])

            if(True in cls_tf_list):
                best_conf = 0
                for idx in check_list:
                    line = compare_list[idx]
                    if((line[2] == True) & (best_conf < line[-1])):
                        best_conf = line[-1]
                        temp_line = line

                if(conf_th < best_conf):
                    detect_condition = [cls_dict[line_true[0]], 'Detect', *temp_line]
                else:
                    detect_condition = [cls_dict[line_true[0]], 'conf_lack', *temp_line]

            else:
                best_conf = 0
                for idx in check_list:
                    line = compare_list[idx]
                    if(best_conf < line[-1]):
                        best_conf = line[-1]
                        detect_condition = [cls_dict[line_true[0]], 'Detect_clsF', *line]

        else:
                print('check check check compare_list Uhaha\n')
                print(compare_list)

        detect_condition_list.append(detect_condition)

    return detect_condition_list

def get_ratio(txt_dir):
    filename_list = os.listdir(txt_dir)
    img_dir = txt_dir.replace('labels', 'images')

    column_name = ['file_name', 'class', 'size']
    return_df = pd.DataFrame(columns = column_name)
    
    process_count = 0
    for file_name in filename_list:
        img_path = img_dir + '\\' + file_name.replace('.txt', '.jpg')

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_size = (img.shape[1], img.shape[0])

        file_path = os.path.join(txt_dir, file_name)
        line_list = get_info_from_txt(file_path)
        for line in line_list:
            if(len(line) != 5):
                print('err ->', file_name)
                continue
            else:
                size = int(get_size(*line[1:], (1280, 720)))
                return_df.loc[len(return_df)] = [file_name, cls_dict[line[0]], size]

        # print progress
        process_count = process_count + 1
        if(process_count % 500 == 0):
            print(f'progress is {round((process_count / len(filename_list) * 100), 2)}%')
    
    print(f'progress is {round((process_count / len(filename_list) * 100), 2)}%')
    
    return return_df

def get_meta_df(true_label_path, pred_label_path, iou_th= 0.5, conf_th= 0.1):
    area1 = []
    area2 = []
    area3 = []
    area4 = []

    #true_txt_list = os.listdir(true_label_path)
    #pred_txt_list = os.listdir(pred_label_path)
    true_txt_list = range(0, 6883, 1)
    pred_txt_list = [item + 1 for item in true_txt_list]

    true_txt_list = [str(name) + '.txt' for name in true_txt_list]
    pred_txt_list = [str(name) + '.txt' for name in pred_txt_list]


    # pred 파일 이름 변환
    #method_graph.rename_files(pred_label_path)
    #pred_txt_list = os.listdir(pred_label_path)
    
    # check files
    if(len(true_txt_list) == len(pred_txt_list)):
        print('num of files is same')
        # for idx in range(len(true_txt_list)):
            # if(true_txt_list[idx] == pred_txt_list[idx]): # pred파일은 1부터 시작
                # pass
            # else:
                # print(f'{idx+1}th file is different, true is {true_txt_list[idx]}, pred is {pred_txt_list[idx]}')
                # return
    else:
        print(f'num of files error, true : {len(true_txt_list)}, pred : {len(pred_txt_list)}')
        return
    
    # box size 구간 구분
    size_th1 = 1000
    size_th2 = 10000
    size_th3 = 50000
    size_th4 = 100000
    
    column_name = ['file_name', 'class', 'dt_condition', 'size', 'iou_tf','class_tf', 'pred_cls',  'iou', 'conf']
    result_df = pd.DataFrame(columns= column_name)

    process_count = 0

    # dir_to_dir
    for idx in range(len(true_txt_list)):
        name_true = true_txt_list[idx]
        name_pred = pred_txt_list[idx]
        out_list = ftf_by_true(f'{true_label_path}/{name_true}', f'{pred_label_path}/{name_pred}', iou_th, conf_th)
        #print(f'out_list num of {name} is', len(out_list))
        #print(out_list)
        #return
        for out in out_list:
            out.insert(0, name_true)
            # print(out)
            result_df.loc[len(result_df)] = out
        
        process_count = process_count + 1
        if(process_count % 1000 == 0):
            print(f'progress is {round((process_count / len(true_txt_list) * 100), 2)}%')

    print(f'progress is {round((process_count / len(true_txt_list) * 100), 2)}%')

    return result_df


# 인컴 기준
ground_true_path = rf'C:\Users\ihman\Desktop\NextChip\dataset\labels\test'
# 깃 기준
teratum_result_path = r'..\..\result\teratum_result'
data_result_path = r'..\..\result\data_result'

def make_csv(category, exp_name, re_exp= False):
    # 저장 목표
    dst = rf'{data_result_path}\{category}\{exp_name}.csv'

    # 중복 검사
    if((re_exp == False) & (os.path.exists(dst))):
        print(f'{dst} is already exist, 스킵!!!')
        
        return
    elif((re_exp == True) & (os.path.exists(dst))):
        print(f'{dst} is already exist, but 다시!!!')

    # 압축파일의 경로, 압축 해제할 경로
    tar_file_path = rf'{teratum_result_path}\{category}\{exp_name}.tar'
    extract_path = rf'{teratum_result_path}\{category}'
    
    tar = tarfile.open(tar_file_path)
    tar.extractall(path= extract_path)
    tar.close()

    v8s_org_train = get_meta_df(ground_true_path, f'{extract_path}\{exp_name}', 0.5, 0.3)
    os.makedirs(rf'{data_result_path}\{category}', exist_ok= True)
    v8s_org_train.to_csv(dst)
    shutil.rmtree(rf'{extract_path}\{exp_name}')

def make_csv_by_dir(dir_name, re_exp= False):
    name_list = os.listdir(f'{teratum_result_path}/{dir_name}')

    for exp_name in name_list:
        if(exp_name[-4:] == '.tar'):
            print(dir_name, exp_name[:-4])
            make_csv(dir_name, exp_name[:-4], re_exp= re_exp)

def make_csv_allll(re_exp= False):
    category_list = os.listdir(teratum_result_path)

    for category in category_list:
        if(category == 'undefined'):
            continue
        elif(category == '.gitkeep'):
            continue
        else:
            make_csv_by_dir(category, re_exp= re_exp)