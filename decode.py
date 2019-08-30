"""
decode output of yolo model
(tx, ty, tw, th) -> (bx, by, bw, bh)
"""
import torch
import numpy as np
from utils import nms, whToxy

"""
anchors: numpy array of (W, H)
feats: tensor one scale output of yolo
    shape (B, num_anchors * (num_classes + 5), H, W)
predictions: reshape feats into (B, H, W, num_anchors, num_classes+5)
"""

def decode(feats, anchors, device, num_cls=80):
    num_anchors = len(anchors)
    B, C, H, W = feats.size()
    sig_fn = torch.nn.Sigmoid()
    predictions = (
        feats.view(B, num_anchors, num_cls + 5, H, W)
        .permute(0, 3, 4, 1, 2)
        .contiguous()
    )
    tx = predictions[..., 0:1]
    ty = predictions[..., 1:2]
    tw = predictions[..., 2:3]
    th = predictions[..., 3:4]
    box_conf = sig_fn(predictions[..., 4:5])
    box_cls = sig_fn(predictions[..., 5:])

    """
    cal bx, by = sigmoid(tx, ty) + cx, cy
    rescale by H, W
    """
    cy, cx = torch.meshgrid(torch.arange(H), torch.arange(W))
    cx = cx.view(1, H, W, 1, 1).float().to(device)
    cy = cy.view(1, H, W, 1, 1).float().to(device)
    bx = (sig_fn(tx) + cx) / W
    by = (sig_fn(ty) + cy) / H

    """
    cal bh, bw and rescale by H, W
    """
    anchors = torch.tensor(anchors, dtype=torch.float, device=device)
    anchors = anchors.view(1, 1, 1, num_anchors, 2)
    pw = anchors[..., 0:1]
    ph = anchors[..., 1:2]
    bw = pw * torch.exp(tw)
    bh = ph * torch.exp(th)
    out = torch.cat(
        (bx, by, bw, bh, box_conf, box_cls), dim=-1
    )
    return out

"""decode all results"""
def full_decode(feats, anchors, anchor_mask, device, num_cls=80):
    out = []
    assert len(feats) == 3
    for l in range(3):
        feat = feats[l]
        device = feat.device
        anchor = anchors[anchor_mask[l]]
        out_ = decode(feat, anchor, device, num_cls)
        out.append(out_)
    return out


"""
filter yolo outputs by confidence threshold and nms
feats: tensor
        raw-outputs of yolo
"""
def filter(
    feats,
    anchors,
    image_size,
    device,
    num_cls=80,
    threshold=0.6,
    iou_threshold=0.5,
    max_output_size=None
):
    anchor_mask = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    outputs = full_decode(feats, anchors, anchor_mask, device, num_cls)
    im_W, im_H = image_size
    scale = torch.FloatTensor([im_H, im_W, im_H, im_W]).to(device).view(1,4)
    boxes = []
    scores = []
    classes = []
    """let's assume one image per test, B=1"""
    for l in range(3):
        out = outputs[l]
        box = whToxy(out[...,0:4], reversed=True)
        b_ = box.view(-1, 4) * scale # rescale boxes [H*W*A, 4]
        s_ = (out[..., 4:5] * out[..., 5:]).view(-1, num_cls) # [H*W*A, num_cls]
        c_ = torch.argmax(out[..., 5:], dim=-1).view(-1) # [H*W*A]
        boxes.append(b_)
        scores.append(s_)
        classes.append(c_)
    boxes = torch.cat(boxes, dim=0)
    scores = torch.cat(scores, dim=0)
    classes = torch.cat(classes, dim=0)

    """
    step 1: filter out predict scores lower than threshold
    """
    best_scores, _ = torch.max(scores, dim=-1) 
    mask = best_scores >= threshold
    boxes = boxes[mask]
    scores = best_scores[mask]
    classes = classes[mask]

    """
    step 2: non max suppression
    """
    selected_indices = nms(boxes, scores, device)
    boxes_ = boxes[selected_indices]
    scores_ = scores[selected_indices]
    classes_ = classes[selected_indices]
    """

    # nms by each classes
    boxes_ = []
    scores_ = []
    classes_ = []
    unique_classes = classes.unique()
    for c in unique_classes:
        cls_mask = (classes == c)
        selected_indices = nms(
            boxes[cls_mask],
            scores[cls_mask],
            device,
            iou_threshold,
            max_output_size
        )
        boxes_.append(boxes[selected_indices])
        scores_.append(scores[selected_indices])
        classes_.append(classes[selected_indices])
    boxes_ = torch.cat(boxes_, dim=0)
    scores_ = torch.cat(scores_, dim=0)
    classes_ = torch.cat(classes_, dim=0)
    """

    return boxes_, scores_, classes_