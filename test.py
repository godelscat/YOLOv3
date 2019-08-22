import torch
from model import YOLO, load_weights
from decode import decode
import numpy as np

# set random seed
torch.manual_seed(7)
device = torch.device('cpu')

anchors = np.array([
    [10, 13], [16, 30], [33, 23], [30, 61], [62, 45], 
    [59, 119], [116, 90], [156, 198], [373, 326]
], dtype=np.float)

anchors = anchors / 416

anchor_mask = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

# input
x = torch.rand(1, 3, 416, 416)

weightsfile = './weights/yolov3.weights'

# load model and its weights
net = YOLO()
load_weights(net, weightsfile)
net.eval()

out = net(x)


"""
outputs = []
for l in range(3):
    feat = out[l]
    anchor = anchors[anchor_mask[l]]
    out_ = decode(feat, anchor, 80, device)
    outputs.append(out_)

one_scale_boxes = outputs[0][0]
osx = one_scale_boxes[3]
osx = osx.view(1, -1)
print(osx[:, :20])
"""

out_size = [x.size() for x in out]
print(out_size)

first = out[0]
print(first[0, 0, ...])