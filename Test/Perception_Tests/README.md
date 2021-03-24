# CARLA perception test
# The version of CARLA is 0.9.5.
## Detection_Carla.py for testing object detection and lane detection


### Lane detection examples
lines,size_im= lane_detection**example**(RGB_Camera_im) , example= v1, v2,v3 are 3 different implementations for test

### Object detection examples
lines,size_im=object_detection_SSD(RGB_Camera_im)  : SSD based object detection in Carla
lines, size_im = object_detection_Yolo(RGB_Camera_im) : Yolo based object detection in Carla
lines, size_im = object_detection_mask(RGB_Camera_im) : Semantic Segmentation based object detection in Carla



# Inside Project shared folder
/Phase3/Code/Models  , download the models to be used in Objecte detection
     
