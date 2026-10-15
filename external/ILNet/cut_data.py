import cv2 
import os
list_folder = ["33"]
base_dir = "/mnt/data/linhnq11/vee/linhnq11/Data/Data_Drone/Data_redetect/Raw/image/"
out_put = "data/"

for folder in list_folder:
    dir_path = base_dir + folder
    list_img = os.listdir(dir_path)
    step = int(len(list_img) / 15)
    print(dir_path)

    for i in range(0, len(list_img), step):
        img = cv2.imread(dir_path + "/" + list_img[i])
        img = img[:,320:1600]
        img = cv2.resize(img,[640,512])

        cv2.imwrite(out_put + folder + "_" + str(i) + ".png", img)