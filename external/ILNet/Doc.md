# train knowledge distillation with custom dataset  
```bash

`python train.py --img_size 512 --batch_size 16 --epochs 600 --warm_up_epochs 10 --learning_rate 0.001 --dataset custom --mode 'S' --amp True  --teacher /mnt/data/danhnt/Seg/ILNet-master/results_IRSTD-1k/2024-01-31-13-49-17_bs10_lr0.001/checkpoint/Epoch-501_IoU-0.7857_nIoU-0.7585_Fa:0.00000828_Pd:0.95178.pth --alpha 0.5

`
```

folder: results_custom/2024-02-01-08-49-14_bs16_lr0.001


# check model L 31/1
# check model S 31/1
# check model S-KD 1/2 9-18 data 1k1
