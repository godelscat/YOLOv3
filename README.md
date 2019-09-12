An exercise to implement YOLOv3 in pyTorch. 

## Yolo Arch

Below is the overall architecture of yolov3. 

![model]()

Every conv block except the final yolo output, including a conv layer, batchnormalization layer and a leaky relu layer. The ResBlock is: 

![res]()

In darknet cfg settings, when upsampling conv layer, we need to upsample the layer with route = -4, which it's an intermediate layer in the block. Specifically, 

![upsampling]()