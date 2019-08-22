"""
compute yolo loss
"""
import torch
from utils import whToxy, iou, preprocess_true_boxes
from decode import decode
import numpy as np

"""
TO DO:    find the anchor which is responsible for the true box 
targets:   output tensor of img_grid;
           shape->[B, H, W, 4+num_cls]
           coordinates type-> (x, y, w, h, c)
           coord value are as unit of image_size
anchors:   numpy array
"""
def create_mask(targets, anchors, eps=1e-8):
    B, H, W, L = targets.size()
    device = targets.device
    A = len(anchors)
    """anchors unit of grid"""
    anchors_ = torch.tensor(anchors, dtype=torch.float, device=device)
    anchors_[:, 0] /= W
    anchors_[:, 1] /= H
    anchors_ = anchors_.view(1, 1, 1, A, 2)

    """targets unit of image"""
    targets_ = targets.view(B, H, W, 1, L)
    targets_wh = targets_[..., 2:4]
    min_wh = torch.min(targets_wh, anchors_)
    ins_area = min_wh[..., 0] * min_wh[..., 1]
    uni_area = (
        targets_wh[..., 0] * targets_wh[..., 1] 
        + anchors_[..., 0] * anchors_[..., 1]
        - ins_area
    )
    iou_scores = ins_area / (uni_area + eps)
    assert iou_scores.size() == (B, H, W, A)
    best_iou, _ = torch.max(iou_scores, dim=3, keepdim=True)
    mask = (iou_scores == best_iou).to(torch.float)
    if mask.device != device:
        mask = mask.to(device)
    return mask

"""
Compute yolo one scale loss
We do not consider the first 12800 iters
feats: one scale output of yolo model
    shape->[B, A*(num_cls+5), H, W]
targets: tensor; outputs of img_grid
    shape->[B, H, W, num_cls+4]
"""
def one_scale_loss(feats, targets, anchors, image_size, downsample, num_classes=80, iou_threshold=0.6):
    """default loss params"""
    lambda_obj = 5
    lambda_noobj = 1
    lambda_class = 1
    lambda_coord = 1

    A = len(anchors)
    B = feats.size()[0]
    im_W, im_H = image_size
    W, H = im_W // downsample, im_H // downsample
    device = feats.device

    p_box, p_c, p_cls = decode(feats, anchors, num_classes, device)
    p_x, p_y, p_w, p_h = p_box

    detector_mask, matching_true_boxes = preprocess_true_boxes(
        targets, anchors, image_size, device, downsample, num_classes
    )    
    detector_mask = torch.from_numpy(detector_mask).to(device)
    matching_true_boxes = torch.from_numpy(matching_true_boxes).to(device)

    t_x = matching_true_boxes[..., 0:1] 
    t_y = matching_true_boxes[..., 1:2]
    t_w = matching_true_boxes[..., 3:4]
    t_h = matching_true_boxes[..., 4:5]

    t_box = (t_x, t_y, t_w, t_h)

    p_box = whToxy(p_box)
    t_box = whToxy(t_box)

    ious = iou(p_box, t_box) 
    ious = torch.squeeze(ious, dim=-1)
    best_ious, _ = torch.max(ious, dim=3, keepdim=True)

    obj_mask = (best_ious > iou_threshold).to(torch.float)
    obj_mask = obj_mask.to(device)
    obj_mask = obj_mask.view(B, H, W, 1, 1)

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
        lambda_coord * obj_mask * detector_mask * (
            (p_x - t_x)**2 + (p_y - t_y)**2 \
            + (p_w - t_w)**2 + (p_h - t_h)**2
        )
    )

    """classification loss"""
    t_cls = matching_true_boxes[..., 4:]
    p_cls = p_cls 
    loss_fn = torch.nn.BCELoss(reduction='none')
    class_loss = torch.sum(lambda_class * obj_mask * detector_mask * loss_fn(p_cls, t_cls))
    """
    class_loss = torch.sum(
        lambda_class * detector_mask * (p_cls - t_cls)**2
    )
    """

    loss_ =  (noobj_loss + obj_loss + coord_loss + class_loss) / B
    return loss_

"""
compute the whole yolo loss
labels: array; shape->[B, num_of_boxes, 5+]
"""
def yolo_loss(feats, labels, anchors, image_size, num_cls=80, iou_threshold=0.6):
    anchor_mask = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    loss = 0.
    init_downsample = 32
    for l in range(3):
        feat = feats[l]
        device = feat.device
        anchor = anchors[anchor_mask[l]]
        downsample = init_downsample // 2**l
#        target = img_grid(labels, image_size, device, downsample, num_cls)
#        loss_ = one_scale_loss(feat, target, anchor, num_cls, iou_threshold) 
        loss_ = one_scale_loss(feat, labels, anchor, image_size, downsample, num_cls)
        loss += loss_
    return loss

if __name__ == "__main__":
    import numpy as np
    feat = torch.rand(1, 30, 3, 3)
    target = torch.empty(1, 3, 3, 9).random_(5)
    anchors = np.random.rand(3,2)
    loss_ = one_scale_loss(feat, target, anchors, 5)
    print(loss_)