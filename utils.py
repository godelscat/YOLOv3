import numpy as np
import torch

"""
Transform (x, y, w, h) into (x1, y1, x2, y2)
x->vertical; y->horizontal
"""
def whToxy(box):
    x, y, w, h = box
    x1 = x - h/2.
    y1 = y - w/2.
    x2 = x + h/2.
    y2 = y + w/2.
    box_ = (x1, y1, x2, y2)
    return box_

"""
calculate iou of two boxes
box1: (x1, y1, x2, y2)
box2: (x1, y1, x2, y2)
"""
def iou(box1, box2, eps=1e-8):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2
    device = box1_x1.device

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.max(box1_x2, box2_x2)
    y2 = torch.max(box1_y2, box2_y2) 

    zero_tensor = torch.zeros((1), device=device)
    ins_area = torch.max((x1 - x2) * (y1 - y2), zero_tensor)
    uni_area = (
        (box1_x1 - box1_x2) * (box1_y1 - box1_y2) + 
        (box2_x1 - box2_x2) * (box2_y1 - box2_y2)
        - ins_area
    )

    iou_scores = ins_area / (uni_area + eps)
    return iou_scores

"""
Read the classes names
"""
def get_classes(file_path):
    classes = []
    with open(file_path, 'r')  as f:
        for line in f.readlines():
            line = line.rstrip()
            classes.append(line)
    return classes

"""
Find detector in YOLO where ground truth box should appear.
Parameters
----------
labels : array
    List of ground truth boxes in form of relative x, y, w, h, class.
    Relative coordinates are in the range [0, 1] indicating a percentage
    of the original image dimensions.
anchors : array
    List of anchors in unit of initial image_size in the range [0, 1]
Returns
-------
matching_true_boxes: tensor, [B, H, W, A, num_cls + 5]
-----------
Ref: https://github.com/allanzelener/YAD2K
"""
def preprocess_true_boxes(labels, anchors, grid_size, device, num_cls=80):
    W, H = grid_size
    A = len(anchors)
    B = len(labels) 
    matching_true_boxes = np.zeros((B, H, W, A, 5+num_cls), dtype=np.float32)
    labels_ = labels.copy()
    labels_[..., 0:2] = labels_[..., 0:2] * np.array([H, W]).reshape(1, 2)
    for b in range(B):
        for box in labels_[b]:
            i = int(box[0])
            j = int(box[1])
            box_wh = box[2:4].reshape(1, 2)
            min_wh = np.minimum(box_wh, anchors)
            ins_area = min_wh[..., 0] * min_wh[..., 1]
            uni_area = (
                box_wh[..., 0] * box_wh[..., 1] 
                + anchors[..., 0] * anchors[..., 1]
                - ins_area
            )
            iou_scores = ins_area / (uni_area + 1e-8)
            idx = np.argmax(iou_scores, axis=-1)

            matching_true_boxes[b, i, j, idx, 0] = box[0] / H
            matching_true_boxes[b, i, j, idx, 1] = box[1] / W
            matching_true_boxes[b, i, j, idx, 2] = box[2]
            matching_true_boxes[b, i, j, idx, 3] = box[3]
            matching_true_boxes[b, i, j, idx, 4] = 1
            assert len(box) >= 4
            # for multi-label
            for k in box[4:]:
                matching_true_boxes[b, i, j, idx, int(k)+5] = 1

    matching_true_boxes = torch.from_numpy(matching_true_boxes).to(device)
    return matching_true_boxes