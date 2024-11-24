import os
import torch
import numpy as np
import argparse

from torchvision import ops
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.metrics import box_iou, DetMetrics
from ultralytics.engine.validator import BaseValidator


parser = argparse.ArgumentParser(description="Nextchip Challenge Evaluator")

parser.add_argument('--w', default=640, type=int, metavar='WIDTH', help='width')
parser.add_argument('--h', default=384, type=int, metavar='HEIGHT', help='height')
parser.add_argument('--pred', default='', type=str, metavar='PRED_PATH',
                    help='path to predictions')
parser.add_argument('--target', default='ground_truth', type=str, metavar='TARGET_PATH',
                    help='path to ground truth')


def print_table(myDict, model_name, colList=None):
   """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
   If column names (colList) aren't specified, they will show in random order.
   Author: Thierry Husson - Use it as you want but don't blame me.
   """
   myDict = [myDict]
   if not colList: colList = list(myDict[0].keys() if myDict else [])
   myList = [colList] # 1st row = header
   for item in myDict: myList.append([str(item[col] if item[col] is not None else '') for col in colList])
   colSize = [max(map(len,col)) for col in zip(*myList)]
   formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
   myList.insert(1, ['-' * i for i in colSize]) # Seperating line
   
   txt_path = 'C:/Users/Ino/Desktop/NextChip/Minions_git/result/eval_result' + model_name + '.txt'
   with open (txt_path, 'w') as f:
    for item in myList:
       print(formatStr.format(*item))
       f.write(formatStr.format(*item) + '\n')


class Evaluator(BaseValidator):
    def __init__(self, width, height):
        self.use_nms = False
        self.w = width
        self.h = height

        self.device = "cpu"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        names = ("person", "car", "bus", "truck", "cycle", "motorcycle")
        names = {i : n for i, n in enumerate(names)}
        
        self.nc = len(names)
        self.metrics = DetMetrics(names=names, plot=False)
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def register(self, pred_path, target_path):
        preds = []
        targets = dict(cls = [], bboxes=[])

        pred_name_list = os.listdir(pred_path)
        pred_name_list.sort(key=lambda x: int(x[:-4])) #.sort(key=lambda x: int(x[22:-4]))
        
        target_name_list = os.listdir(target_path)
        target_name_list.sort(key=lambda x: int(x[:-4]))


        for (p_name, t_name) in zip(pred_name_list, target_name_list):
            pred = np.loadtxt(os.path.join(pred_path, p_name))
            target = np.loadtxt(os.path.join(target_path, t_name))  
            
            pred = np.zeros((1, 6)) if pred.shape[-1] == 0 else pred
            pred = pred[np.newaxis, :] if len(pred.shape) == 1 else pred
            
            pred = pred[pred[:, -1].argsort()[::-1], :] # sorted by confidence
            new_pred = np.zeros_like(pred)
            new_pred[:, :4] = xywh2xyxy(pred[:, 1:5] * np.array((self.w, self.h))[[0, 1, 0, 1]])
            new_pred[:, 4] = pred[:, -1]
            new_pred[:, -1] = pred[:, 0]
            
            del_row = np.where(new_pred[:, :4] < 0)[0]
            if len(del_row) > 0:
                new_pred = np.delete(new_pred, del_row, axis=0)
            
            new_pred = torch.from_numpy(new_pred).round()
            keep = ops.nms(new_pred[:, :4], new_pred[:, 4], 0.6)
            preds.append(new_pred[keep])

            target = target[np.newaxis, :] if len(target.shape) == 1 else target
            target = torch.from_numpy(target)
            targets['cls'].append(target[:, 0])
            targets['bboxes'].append(xywh2xyxy(target[:, 1:] * torch.tensor((self.w, self.h))[[0, 1, 0, 1]]))
        
        return preds, targets
    
    def update_metrics(self, preds, targets):
        for si, pred in enumerate(preds):  
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            ) 
            cls, bbox = targets["cls"][si], targets["bboxes"][si]
            
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()

            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                continue
                
            # Predictions
            pred[:, :4] = pred[:, :4]
            
            stat["conf"] = pred[:, 4]
            stat["pred_cls"] = pred[:, 5]
            if nl:
                stat["tp"] = self._process_batch(pred, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict


if __name__ == "__main__":
    args = parser.parse_args()

    w = args.w
    h = args.h
    model_name = args.pred
    target_path = args.target

    teratum_result = '../result/teratum_result'

    

    solver = Evaluator(w, h)
    preds, targets = solver.register(pred_path='C:/Users/Ino/Desktop/NextChip/eval_result/' + model_name, target_path=target_path)
    solver.update_metrics(preds, targets)
    result_dict = solver.get_stats()
    print(result_dict)
    print_table(result_dict, model_name)
