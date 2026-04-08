import os
import cv2
import numpy as np
from tqdm import tqdm


def main():
    root = '/home/ubuntu/lijiahao/Datasets/Postdam'
    list_path = '/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/lib/data/postdam_list/val.lst'
    img_ids = [i_id.strip().split() for i_id in open(list_path)]
    # print(img_ids)
    # exit(1)
    for item in img_ids:
        label = cv2.imread(os.path.join(root, item[-1]))
        print(np.amax(label))
        print(np.amin(label))
        # label = label - 1
        # cv2.imwrite(os.path.join(root, item[-1]), label)
        # exit(3)


if __name__ == '__main__':
    main()