def ftf_by_pred(true_file_path, pred_file_path, iou_th= 0.5, conf_th= 0.1):
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