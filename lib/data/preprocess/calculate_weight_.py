import cv2
import numpy as np
import os
from tqdm import tqdm


LABELLIST = [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127], [0, 63, 191],
                [0, 63, 255], [0, 127, 63], [0, 127,127], [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127], [0, 127, 191],
                [0, 127, 255], [0, 100, 155]]
weighLst = [0 for _ in range(16)]
weighLst_ = np.array([706241071, 1457856, 168874, 1504746, 4707883, 414640, 3414441, 77810, 4536251, 5255900, 12890, 1409871, 2756070, 5855356, 3115302, 3264441])

def main():
    root = '/home/ubuntu/lijiahao/Datasets/ISAID'
    list_path = '/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/lib/data/isaid_list/test_.lst'
    img_ids = [i_id.strip().split() for i_id in open(list_path)]
    labelList = list()
    for item in tqdm(img_ids):
        labelListTemp = list()
        label = cv2.imread(os.path.join(root, item[-1]))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        h, w, _ = label.shape
        flagTemp = False
        for i in range(0, h):
            for j in range(0, w):
                if (label[i][j] == np.array([0, 0, 0])).all():
                    weighLst[0] += 1
                elif (label[i][j] == np.array([0, 0, 63])).all():
                    weighLst[1] += 1
                elif (label[i][j] == np.array([0, 63, 63])).all():
                    weighLst[2] += 1
                elif (label[i][j] == np.array([0, 63, 0])).all():
                    weighLst[3] += 1
                elif (label[i][j] == np.array([0, 63, 127])).all():
                    weighLst[4] += 1
                elif (label[i][j] == np.array([0, 63, 191])).all():
                    weighLst[5] += 1
                elif (label[i][j] == np.array([0, 63, 255])).all():
                    weighLst[6] += 1
                elif (label[i][j] == np.array([0, 127, 63])).all():
                    weighLst[7] += 1
                elif (label[i][j] == np.array([0, 127,127])).all():
                    weighLst[8] += 1
                elif (label[i][j] == np.array([0, 0, 127])).all():
                    weighLst[9] += 1
                elif (label[i][j] == np.array([0, 0, 191])).all():
                    weighLst[10] += 1
                    if flagTemp == False:
                        print(item[-1])
                        flagTemp = True
                elif (label[i][j] == np.array([0, 0, 255])).all():
                    weighLst[11] += 1
                elif (label[i][j] == np.array([0, 191, 127])).all():
                    weighLst[12] += 1
                elif (label[i][j] == np.array([0, 127, 191])).all():
                    weighLst[13] += 1
                elif (label[i][j] == np.array([0, 127, 255])).all():
                    weighLst[14] += 1
                elif (label[i][j] == np.array([0, 100, 155])).all():
                    weighLst[15] += 1   
    print(weighLst)


if __name__ == '__main__':
    main()
    # print(weighLst_/np.sum(weighLst_))  # [0.95, 0.02, 0.0003, 0.003, 0.006, 0.0006, 0.005, 0.0001, 0.006, 0.007, 0.00002, 0.002, 0.004, 0.008, 0.004, 0.004]