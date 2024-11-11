import cv2
import method_dataframe

def check_result(gt_path, pred_path):
    gt_line = []
    pred_line = []
    gt_img_path = gt_path.replace('test_rename', 'test')
    gt_img_path = gt_img_path.replace('labels', 'images')[:-3] + 'jpg'
    print(gt_img_path)
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
    # cv2.rectangle(gt_img, (400, 500), (420, 520), (0, 0, 0), 2)
    
    cv2.imshow('gt_img', gt_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def draw_box(gt_txt_path, idx_list):
    # img path
    gt_img_path = gt_txt_path.replace('labels', 'images')
    gt_img_path = gt_img_path.replace('txt', 'jpg')
    gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
    img_size = (gt_img.shape[1], gt_img.shape[0])

    # get info
    info = method_dataframe.get_info_from_txt(gt_txt_path)

    if(idx_list == 'all'):
        idx_list = range(0, len(info), 1)

    for idx in idx_list:
        line = [float(item) for item in info[idx]]
        center_cord = line[1:]
        center_cord.append(img_size)
        print(center_cord)
        box_cord = method_dataframe.coordinate_conveter(*center_cord)
        cv2.rectangle(gt_img, tuple(box_cord[:2]), tuple(box_cord[2:]), (0, 255, 0), 2)

    # draw
    gt_img_resize = cv2.resize(gt_img, (1280, 720))
    cv2.imshow('gt_img', gt_img_resize)
    cv2.waitKey()
    cv2.destroyAllWindows()