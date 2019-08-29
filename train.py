"""
Train the YOLOv3 model from scratch.
But it's only a quite small fake dataset, because of my limited GPU resource
Default params:
    image_shape-> [3, 256, 256], channel-first
    num_classes-> 80
    num_anchors-> 3 * 3
"""
import numpy as np
import torch
from loss import yolo_loss
from model import YOLO

def create_fake_data(image_shape, num_classes=80, num_of_batch=10):
    width, height = image_shape
    batch_size = 8
    for i in range(num_of_batch):
        imgs = torch.rand((batch_size, 3, width, height), dtype=torch.float)
        tars = np.random.randint(0, num_classes, size=(batch_size, 5, 5))
        tars = tars.astype(float)
        for j in range(4):
            tars[..., j] = tars[...,j] / num_classes
        yield imgs, tars

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""params"""
# anchors as unit of pixel
anchors = np.array([
    [10, 13], [16, 30], [33, 23], [30, 61], [62, 45], 
    [59, 119], [116, 90], [156, 198], [373, 326]
])
anchor_mask = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
image_size = (416, 416) # default input image size
# transform into unit of image
anchors = anchors / np.asarray(image_size).reshape(1,2)
num_classes = 80

"""train"""
net = YOLO()
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
loss_fn = yolo_loss

for epoch in range(10):
    loss = 0.
    num_of_batch = 20
    for imgs, tars in create_fake_data(image_size, num_of_batch=num_of_batch):
        imgs = imgs.to(device)
        feats = net(imgs)
        loss_ = loss_fn(feats, tars, anchors, anchor_mask, device, image_size)
        loss +=  loss_.item()
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
    print("epoch {0}, loss {1}".format(epoch, loss / num_of_batch))