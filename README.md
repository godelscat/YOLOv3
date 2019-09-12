An exercise to implement YOLOv3 in pyTorch. 

## Yolo Arch

Below is the overall architecture of yolov3. 

![model](https://github.com/godelscat/YOLOv3/blob/master/arch/yolov3.jpg)

Every conv block except the final yolo output, including a conv layer, batchnormalization layer and a leaky relu layer.


[1] In darknet cfg settings, when upsampling conv layer, we need to upsample the layer with route = -4, which is an intermediate layer in the conv block. Specifically, 

![upsampling](https://github.com/godelscat/YOLOv3/blob/master/arch/upsampling.jpg)