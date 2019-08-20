"""
compute yolo loss
"""
import torch
from utils import whToxy, iou, img_grid
from decode import decode

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
        mask.to(device)
    return mask

"""
Compute yolo one scale loss
We do not consider the first 12800 iters
"""
def one_scale_loss(feats, targets, anchors, num_classes=80, iou_threshold=0.6):
    """default loss params"""
    lambda_obj = 5
    lambda_noobj = 1
    lambda_class = 1
    lambda_coord = 1

    A = len(anchors)
    device = feats.device

    B, H, W, L = targets.size()
    p_box, p_c, p_cls = decode(feats, anchors, num_classes, device)
    p_x, p_y, p_w, p_h = p_box
    targets_ = targets.view(B, H, W, 1, L)
    t_x = targets_[..., 0:1] 
    t_y = targets_[..., 1:2]
    t_w = targets_[..., 3:4]
    t_h = targets_[..., 4:5]
    t_box = (t_x, t_y, t_w, t_h)

    p_box = whToxy(p_box)
    t_box = whToxy(t_box)

    ious = iou(p_box, t_box) 
    ious = torch.squeeze(ious, dim=-1)
    assert ious.size() == (B, H, W, A)
    best_ious, _ = torch.max(ious, dim=3, keepdim=True)

    obj_mask = (best_ious > iou_threshold).to(torch.float)
    obj_mask.to(device)
    detector_mask = create_mask(targets, anchors)
    obj_mask = obj_mask.view(B, H, W, 1, 1)
    detector_mask = detector_mask.view(B, H, W, A, 1)

    """non object loss"""
    noobj_loss =  torch.mean(
        lambda_noobj * (1-obj_mask) * (1-detector_mask) * (-p_c)**2
    )

    """object loss"""
    obj_loss = torch.mean(
        lambda_obj * obj_mask * detector_mask * (1-p_c)**2
    )

    """coord loss"""
    coord_loss = torch.mean(
        lambda_coord * obj_mask * detector_mask * (
            (p_x - t_x)**2 + (p_y - t_y)**2 \
            + (p_w - t_w)**2 + (p_h - t_h)**2
        )
    )

    """classification loss"""
    t_cls = targets_[..., 4:]
    p_cls = p_cls * obj_mask * detector_mask
    t_cls = t_cls * obj_mask * detector_mask
    loss_fn = torch.nn.BCELoss()
    class_loss = lambda_class * loss_fn(p_cls, t_cls)

    loss_ = 0.5 * (noobj_loss + obj_loss + coord_loss + class_loss)
    return loss_


if __name__ == "__main__":
    import numpy as np
    feat = torch.rand(1, 30, 3, 3)
    target = torch.empty(1, 3, 3, 9).random_(5)
    anchors = np.random.rand(3,2)
    loss_ = one_scale_loss(feat, target, anchors, 5)
    print(loss_)