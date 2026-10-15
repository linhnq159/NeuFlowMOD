# ILNet: Low-level Matters for Salient Infrared Small Target Detection


## Requirements
     Python 3.7.10
     torch 1.10.1

## Training

``python train.py  --batch_size 32 --epochs 600 --warm_up_epochs 10 --learning_rate 0.001 --dataset custom --mode 'LW_V1' --amp True --device 'cuda:0' --transfer "/mnt/data/danhnt/Seg/work_seg/checkpoint/results_custom/LW_V1_CUSTOM_2025-07-24-13-48-17_bs32_lr0.001/checkpoint/Epoch- 42_IoU-0.7803_nIoU-0.7762_Fa:0.00011824_Pd:0.95012.pth" ``

exp : python train.py  --batch_size 32 --epochs 600 --warm_up_epochs 10 --learning_rate 0.001 --dataset custom --mode 'LW_V1' --amp True --device 'cuda:0'


In default, the _'.pth'_ will be saved at `` ./results_sirst/ `` or `` ./results_IRSTD-1k/ ``.

## Distillation Knowledge
``python train.py --batch_size 8 --epochs 600 --warm_up_epochs 10 --learning_rate 0.001 --dataset sirst --mode 'L' --amp True --device 'cuda:1' --teacher ' .pth' --alpha 0.5 ``


## Valuating

``python val.py --dataset 'IRSTD-1k' --batch_size 1 --mode 'S' --checkpoint ' .pth' ``


## Demo
``python demo.py --img_path ../work_seg/Datasets/data/ --mode LW_V1 --checkpoint ../work_seg/checkpoint/weights/LW_V1_Epoch-432_IoU-0.9031_nIoU-0.4694_Fa\:0.00000643_Pd\:0.97158.pth --imshow``

## Demo Video
``python demo_vid.py --vid_path ' .mp4/avi'  --mode 'L' --checkpoint ' .pth'``

## Convert onnx
`` python export.py --model LW_V1 --weight ILNet_LW_V1.pth --output LW_Net_V1.onnx

## Datasets
Dataset folder should be like:

https://github.com/RuiZhang97/ISNet
~~~
IRSTD-1k
└───imges
│       │   XDU0.png
│       │   XDU1.png
│       │  ...
└───masks
│       │   XDU0.png
│       │   XDU1.png
│       │  ...
└───trainval.txt
└───test.txt
~~~
https://github.com/YimianDai/sirst
~~~
SIRST
└───idx_320
│       │   trainval.txt
│       │   test.txt
└───idx_427
│       │   trainval.txt
│       │   test.txt
└───imges
│       │   Misc_1.png
│       │   Misc_2.png
│       │  ...
└───masks
│       │   Misc_1_pixels0.png
│       │   Misc_2_pixels0.png
│       │  ...
~~~


## Best Results
SIRST
Mode      | Best IoU(%) | Best nIoU(%)  | Best Pd(%)  | Best Fa(1e-6)
---       | ---         | ---           | ---         | ---
ILNet-S   | 78.12       |  76.42        |  99.07      |  5.50
ILNet-M   | 79.57       |  77.19        |  98.15      |  3.02
ILNet-L   | 80.31       |  78.22        |  100        |  1.33
---       | ---         | ---           | ---         | ---
IRSTD-1k
Mode      | Best IoU(%) | Best nIoU(%)  | Best Pd(%)  | Best Fa(1e-6)
---       | ---         | ---           | ---         | ---
ILNet-S   | 66.01       |  64.78        |  93.27      |  5.26
ILNet-M   | 67.86       |  68.40        |  94.61      |  5.09
ILNet-L   | 70.15       |  68.91        |  95.29      |  3.23


## Thanks:
     Part of the code draws on the work of the following authors:
        https://github.com/Tianfang-Zhang/acm-pytorch
        https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_segmentation/u2net
     
     Datasets:
        https://github.com/YimianDai/sirst
        https://github.com/RuiZhang97/ISNet

     Metrics:
        https://github.com/YimianDai/sirst
        https://github.com/Lliu666/DNANet_BatchFormer

