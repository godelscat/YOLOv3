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
TO DO: transform the labels of shape [B, T, 5] into format [B, W, H, 5]
        which means, put the true box into corresponding grid
T : number of true boxes in a single image
labels: array
        shape->[B, T, 5]
        coord type->(x, y, w, h, c)
        coord value are as unit of image_size
image_size: (width, height)
"""
def img_grid(labels, image_size, device, downsample=32, num_cls=80):
    im_W, im_H = image_size
    assert im_W % downsample == 0
    assert im_H % downsample == 0
    W, H = im_W // downsample, im_H // downsample # num of grids
    B = len(labels) 
    targets = torch.zeros((B, H, W, 4 + num_cls), dtype=torch.float)
    labels[..., 0:2] = labels[..., 0:2] * np.array([H, W]).reshape(1, 2)
    for b in B:
        for box in labels[b]:
            i = int(box[0])
            j = int(box[1])
            targets[b, i, j, 0] = box[0] / H
            targets[b, i, j, 1] = box[1] / W
            targets[b, i, j, 2] = box[2]
            targets[b, i, j, 3] = box[3]
            targets[b, i, j, 4+box[4]] = 1 # only one-label
            """
            # support multilabel
            # transform into one-hot label
            for c in box[4:]:
                targets[b, i, j, 4+c] = 1 
            """
    targets.to(device)
    return targets