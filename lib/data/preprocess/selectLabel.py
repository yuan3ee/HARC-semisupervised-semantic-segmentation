import cv2
import numpy as np
import os
from tqdm import tqdm


def main():
    root = '/home/ubuntu/lijiahao/Datasets/ISAID'
    list_path = '/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/lib/data/isaid_list/test_.lst'
    img_ids = [i_id.strip().split() for i_id in open(list_path)]
    labelList = list()
    for item in tqdm(img_ids):
        labelListTemp = list()
        label = cv2.imread(os.path.join(root, item[-1]))
        h, w, _ = label.shape
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                labelListTemp.append(label[i][j])
        for m in range(np.unique(np.array(labelListTemp), axis=0).shape[0]):
            labelList.append(np.unique(np.array(labelListTemp), axis=0)[m])
        # print(labelList)
        # exit(1)
    print(np.unique(np.array(labelList), axis=0))


if __name__ == '__main__':
    main()