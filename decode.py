"""
decode output of yolo model
(tx, ty, tw, th) -> (bx, by, bw, bh)
"""
import torch
import torch.nn.functional as F

"""
anchors: numpy array of (W, H)
feats: tensor one scale output of yolo
    shape (B, num_anchors * (num_classes + 5), H, W)
predictions: reshape feats into (B, H, W, num_anchors, num_classes+5)
"""

def decode(feats, anchors, num_classes, device):
    num_anchors = len(anchors)
    B, C, H, W = feats.size()
    predictions = (
        feats.view(B, num_anchors, num_classes + 5, H, W)
        .permute(0, 3, 4, 1, 2)
        .contiguous()
    )
    tx = predictions[..., 0:1]
    ty = predictions[..., 1:2]
    th = predictions[..., 2:3]
    tw = predictions[..., 3:4]
    box_conf = F.sigmoid(predictions[..., 4:5])
    box_cls = F.sigmoid(predictions[..., 5:])

    """
    cal bx, by = sigmoid(tx, ty) + cx, cy
    rescale by H, W
    """
    cx, cy = torch.meshgrid(torch.arange(H), torch.arange(W))
    cx = cx.view(1, H, W, 1, 1).to(torch.float).to(device)
    cy = cy.view(1, H, W, 1, 1).to(torch.float).to(device)
    bx = (F.sigmoid(tx) + cx) / H
    by = (F.sigmoid(ty) + cy) / W

    """
    cal bh, bw and rescale by H, W
    """
    anchors = torch.tensor(anchors, dtype=torch.float, device=device)
    anchors = anchors.view(1, 1, 1, num_anchors, 2)
    pw = anchors[..., 0:1]
    ph = anchors[..., 1:2]
    bw = pw * torch.exp(tw) / W
    bh = ph * torch.exp(th) / H
    boxes = torch.cat(
        (bx, by, bh, by), dim=-1
    )
    assert boxes.size() == (B, H, W, num_anchors, 4)
    return boxes, box_conf, box_cls