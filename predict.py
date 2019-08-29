import torch
import utils
import numpy as np
from model import YOLO, load_weights

"""params"""
image_path  = './data/car.jpg'
input_shape = (416, 416)
im, image, image_size = utils.preprocess_image(image_path, input_shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

anchors = np.array([
    [10, 13], [16, 30], [33, 23], [30, 61], [62, 45], 
    [59, 119], [116, 90], [156, 198], [373, 326]
], dtype=np.float32)
anchors = anchors / input_shape[0] # as unit of image

class_path = "./data/coco_classes.txt"
class_names = utils.get_classes(class_path)
num_cls = len(class_names)

"""load model"""
weightsfile = './weights/yolov3.weights'
net = YOLO()
net.to(device)
load_weights(net, weightsfile)
net.eval()

with torch.no_grad():
    image = torch.from_numpy(image).to(device)
    image = image.permute(0, 3, 1, 2).contiguous()
    feats = net(image)
    boxes_, scores_, classes_ = utils.filter(
        feats,
        anchors,
        image_size,
        device,
        num_cls,
        threshold=0.6
    )

boxes = boxes_.cpu().numpy()
scores = scores_.cpu().numpy()
classes = classes_.cpu().numpy()
colors = utils.generate_colors(class_names)
utils.draw_boxes(im, scores, boxes, classes, class_names, colors)
im.save("./data/out.jpg")