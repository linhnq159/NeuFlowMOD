import numpy as np
import cv2
from PIL import Image

class Detect:
    def __init__(self, isDanger=True):
        self.isDanger = isDanger
        self.true_target = 0
        self.total_target = 0
        self.miss_detect = 0
        self.false_detect = 0
        self.total_pixel = 0
        self.min_area = 999
        self.max_area = 0
        

    def update(self,pred,labels,IoU = 0.0):
        correct, total, miss_detect, false_detect = self.matching_pred(pred,labels, IoU= IoU)
        # print(correct, total)
        self.true_target += correct
        self.total_target += total
        self.miss_detect += miss_detect
        self.false_detect += false_detect

    def get_max_min(self):
        print("min area: ", self.min_area, " max area: ", self.max_area)
        
    def get(self):
        assert self.total_target > 0
        return self.true_target/self.total_target, self.total_pixel, self.miss_detect/self.total_target , self.false_detect/self.total_target

    def sort_countour(self,x,h):
        x,y,w,h = cv2.boundingRect(x)
        return (x + y*h)

    def matching_pred(self, pred, labels, IoU):
        detect_true = 0
        pred = pred.detach().numpy()
        labels = labels.detach().numpy()
        predict = (pred > 0).astype(np.uint8)*255
        self.total_pixel += np.sum(predict)
        # print("max: ", np.max(labels))
        # print("min: ", np.min(labels))
        labels = (labels > 0).astype(np.uint8)*255
        # print(labels.shape)
        predict = predict[0,0,:,:]
        labels = labels[0,0,:,:]
        # mask = predict
        # mask = np.dstack((mask, mask, mask)).astype('uint8')
        # print(mask.shape)
        # # mask = Image.fromarray(mask.astype('uint8')).convert('RGB')

        # labels = np.dstack((labels, labels, labels)).astype('uint8')
        # cv2.imwrite("pred.png", mask)
        # cv2.imwrite("labels.png", labels)

        # import sys 
        # sys.exit(0)
        # print(predict.shape)
        # print(labels.shape)
        

        contours, _ = cv2.findContours(predict, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        labels_contours, _ = cv2.findContours(labels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        check = predict*labels
        out_check,_ = cv2.findContours(check, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        miss_detect = max(len(labels_contours) - len(contours), 0)
        false_detect = max(len(contours) - len(labels_contours), 0)


        # contours = sorted(contours,key= lambda ctr: self.sort_countour(ctr,predict.shape[1]))
        # labels_contours = sorted(labels_contours,key=lambda ctr: self.sort_countour(ctr,labels.shape[1]))

        # temp = min(len(contours), len(labels_contours))
       
        # for i in range(temp):
        #     x1,y1,w1,h1 = cv2.boundingRect(contours[i])
        #     x2,y2,w2,h2 = cv2.boundingRect(labels_contours[i])
            
        #     x_overlap = max(0,min(x1 + w1, x2 + w2) - max(x1,x2))
        #     y_overlap = max(0,min(y1 + h1, y2 + h2) - max (y1,y2))

        #     if ((x_overlap*y_overlap)/(w2*h2)) > IoU:
        #         detect_true +=1
        #         self.min_area = min(w2 * h2, self.min_area)
        #         self.max_area = max(h2 * w2, self.max_area)
        
        # print(len(contours), len(labels_contours))

        for i in range(len(labels_contours)):
            x1,y1,w1,h1 = cv2.boundingRect(labels_contours[i])
            for j in range(len(contours)):
                x2,y2,w2,h2 = cv2.boundingRect(contours[j])
                
                x_overlap = max(0,min(x1 + w1, x2 + w2) - max(x1,x2))
                y_overlap = max(0,min(y1 + h1, y2 + h2) - max (y1,y2))

                if ((x_overlap*y_overlap)/(w2*h2)) > IoU:
                    detect_true +=1
                    self.min_area = min(w2 * h2, self.min_area)
                    self.max_area = max(h2 * w2, self.max_area)
                    break

                

        if self.isDanger:
            return detect_true , max(len(labels_contours),len(contours))
            # print("isDanger")
            # return detect_true , len(contours)
        else:
            # return detect_true , len(labels_contours), miss_detect, false_detect
            return len(out_check) , len(contours), miss_detect, false_detect
    
    def reset(self):
        self.true_target = 0
        self.total_target = 0
        self.miss_detect = 0
        self.false_detect = 0


# import time
# print("Start")
# start = time.time()    
# mask1_path = '/mnt/data/danhnt/Seg/ILNet-master/data_custom/SIRST/masks/Misc_422_pixels0.png'
# mask2_path = '/mnt/data/danhnt/Seg/ILNet-master/data_custom/SIRST/masks/Misc_422_pixels0.png'
# pred = cv2.imread(mask1_path)
# target = cv2.imread(mask2_path)

# pred = cv2.resize(pred,(240,320))
# target = cv2.resize(target,(240,320))

# pred = cv2.cvtColor(pred,cv2.COLOR_BGR2GRAY)/255
# target = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)/255

# # pred.astype('int64')
# # target.astype('int64')

# Check = Detect()
# Check.update(pred,target, IoU=0.0)
# print(Check.get())
# end = time.time()
# print("End")
# print("Time elapsed: ", end - start )