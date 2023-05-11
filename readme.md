# Optic disc detection with key_point_detection
As the project [SAM](https://github.com/facebookresearch/segment-anything) is released, I hold that there is no need to perform the detection tasks which generates a bound box for optic disc. Instead, we can detect the center of the optic disc and using SAM to generate a mask for the optic disc(if you need). In fact, key point detection and object detection is resemble in the model structure and some other fasets.

The model is [HRNetv2](https://github.com/HRNet/HRNet-Facial-Landmark-Detection), thanks for their open source code. Even through it may not be the best model for the optic detect task because I straight crop the model from a facial landmark detect porject, the visualization also shown a acceptable performance.

If you have any question, do not hesitate to push an issue in ths repository. Actually, I am not confident about my code style. At least, it can sucessfully run in my computer.
# Datasets Condition
## DRIONS-DB
There are train: 100 , val: 5 ,test:5
## GY
There are train: 140 , val: 40 ,test:20
## HRF
There are train: 31 , val: 9 ,test:5
## ODVOC
There are train: 81 , val: 23 ,test:12

DRIONS-DB, HRF, ODVOC is publical avaible dataset you can downloaded them from there official dataset.

GY is the private dataset for my [main project](https://github.com/defensetongxue/ROP_diagnose), and labeled by myself with another open source project [label-studio](https://labelstud.io/)

