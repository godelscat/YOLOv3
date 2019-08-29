"""
compute yolo loss
"""
import torch
from utils import whToxy, iou, preprocess_true_boxes
from decode import decode
import numpy as np

"""
Compute yolo one scale loss
We do not consider the first 12800 iters
feats: one scale output of yolo model
    shape->[B, A*(num_cls+5), H, W]
matching_true_boxes: tensor, 
    outputs of preprocess_true_boxes
    shape->[B, H, W, A, num_cls+5]
"""
def one_scale_loss(
    feats,
    matching_true_boxes,
    anchors,
    device,
    num_cls=80,
    iou_threshold=0.6
):
    """default loss params"""
    lambda_obj = 5
    lambda_noobj = 1
    lambda_class = 1
    lambda_coord = 1

    B, H, W, A, _ = matching_true_boxes.size()

    #p_box, p_c, p_cls = decode(feats, anchors, device, num_cls)
    out = decode(feats, anchors, device, num_cls)
    p_box = out[..., 0:4]
    p_c = out[..., 4:5]
    p_cls = out[..., 5:]

    detector_mask = matching_true_boxes[...,4:5]
    t_box = matching_true_boxes[...,0:4]

    p_box = whToxy(p_box)
    t_box = whToxy(t_box)

    ious = iou(p_box, t_box) 
    best_ious, _ = torch.max(ious, dim=3, keepdim=True)

    obj_mask = (best_ious > iou_threshold).to(torch.float)
    obj_mask = obj_mask.to(device)
    obj_mask = obj_mask.view(B, H, W, 1, 1)

    t_box = matching_true_boxes[..., 0:4]
    p_box = out[..., 0:4]

    """non object loss"""
    noobj_loss =  torch.sum(
        lambda_noobj * (1-obj_mask) * (1-detector_mask) * (-p_c)**2
    )

    """object loss"""
    obj_loss = torch.sum(
        lambda_obj * obj_mask * detector_mask * (1-p_c)**2
    )

    """coord loss"""
    coord_loss = torch.sum(
        lambda_coord * detector_mask * (p_box - t_box)**2
    )

    """classification loss"""
    t_cls = matching_true_boxes[..., 5:]
    cls_loss_fn= torch.nn.BCELoss(reduction='sum')
    class_loss = lambda_class * cls_loss_fn(p_cls * detector_mask, t_cls)

    loss_ =  (noobj_loss + obj_loss + coord_loss + class_loss) / B
    return loss_

"""
compute the whole yolo loss
labels: array; shape->[B, num_of_boxes, 5+]
"""
def yolo_loss(
    feats,
    labels,
    anchors,
    anchor_mask,
    device,
    image_size,
    num_cls=80,
    iou_threshold=0.6
):
    im_W, im_H = image_size
    loss = 0.
    init_downsample = 32
    for l in range(3):
        feat = feats[l]
        anchor = anchors[anchor_mask[l]]
        downsample = init_downsample // 2**l
        grid_size = im_W // downsample, im_H // downsample
        matching_true_boxes = preprocess_true_boxes(
            labels, anchor, grid_size, device, num_cls 
        )
        loss_ = one_scale_loss(
            feat, matching_true_boxes, anchor, device, num_cls, iou_threshold
        )
        loss += loss_
    return loss