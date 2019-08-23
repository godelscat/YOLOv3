import torch
from model import YOLO, load_weights
from decode import full_decode
from loss import yolo_loss
import numpy as np

# set random seed
torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

anchors = np.array([
    [10, 13], [16, 30], [33, 23], [30, 61], [62, 45], 
    [59, 119], [116, 90], [156, 198], [373, 326]
], dtype=np.float)
anchors = anchors / 416
anchor_mask = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
image_size = (416, 416)

# input
x = torch.rand(1, 3, 416, 416)

tars = np.random.randint(0, 80, size=(1, 5, 5))
tars = tars.astype(float)
for j in range(4):
    tars[..., j] = tars[...,j] / 80 

#y = torch.from_numpy(tars)

weightsfile = './weights/yolov3.weights'

# load model and its weights
net = YOLO()
load_weights(net, weightsfile)
net.eval()

out = net(x)

"""
outputs = full_decode(out, anchors, anchor_mask, device)
print(len(outputs))
one_scale_boxes = outputs[0][0]
osx = one_scale_boxes[0]
osx = osx.permute(0, 3, 2, 1, 4).contiguous().view(1,-1)
print(osx[0,:20])
"""

# check loss
loss = yolo_loss(out, tars, anchors, anchor_mask, device, image_size)
print(loss)