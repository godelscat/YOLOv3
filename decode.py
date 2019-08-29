"""
decode output of yolo model
(tx, ty, tw, th) -> (bx, by, bw, bh)
"""
import torch

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
    th = predictions[..., 2:3]
    tw = predictions[..., 3:4]
    box_conf = sig_fn(predictions[..., 4:5])
    box_cls = sig_fn(predictions[..., 5:])

    """
    cal bx, by = sigmoid(tx, ty) + cx, cy
    rescale by H, W
    """
    cx, cy = torch.meshgrid(torch.arange(H), torch.arange(W))
    cx = cx.view(1, H, W, 1, 1).to(torch.float).to(device)
    cy = cy.view(1, H, W, 1, 1).to(torch.float).to(device)
    bx = (sig_fn(tx) + cx) / H
    by = (sig_fn(ty) + cy) / W

    """
    cal bh, bw and rescale by H, W
    """
    anchors = torch.tensor(anchors, dtype=torch.float, device=device)
    anchors = anchors.view(1, 1, 1, num_anchors, 2)
    pw = anchors[..., 0:1]
    ph = anchors[..., 1:2]
    bw = pw * torch.exp(tw)
    bh = ph * torch.exp(th)
#    boxes = (bx, by, bw, bh)
#    return boxes, box_conf, box_cls
    out = torch.cat(
        (bx, by, bw, bh, box_conf, box_cls), dim=-1
    )
    assert out.size() == (B, H, W, num_anchors, 85)
    return out

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