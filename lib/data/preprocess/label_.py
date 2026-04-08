import cv2
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


LABELLIST = [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127], [0, 63, 191],
                [0, 63, 255], [0, 127, 63], [0, 127,127], [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127], [0, 127, 191],
                [0, 127, 255], [0, 100, 155]]


def label(unlabelImgDir, labelImgDir):
    unlabelImgList = os.listdir(unlabelImgDir)
    unlabelImgList = sorted(unlabelImgList)
    # print(range(len(unlabelImgList)))
    for i in tqdm(range(len(unlabelImgList))):
        unlabelImgPath = os.path.join(unlabelImgDir, unlabelImgList[i])
        unlabelImg = cv2.imread(unlabelImgPath)
        unlabelImg = cv2.cvtColor(unlabelImg, cv2.COLOR_BGR2RGB)
        changed = np.zeros(unlabelImg.shape[:2])
        # width = unlabelImg.shape[1]
        # height = unlabelImg.shape[0]
        temp = unlabelImg.copy()
        for j in range(len(LABELLIST)):
            changed[np.where(np.all(temp == LABELLIST[j], axis=-1))] = j
        cv2.imwrite(os.path.join(labelImgDir, unlabelImgList[i]), changed)


def testLabelFunc(labelImgPath):
    labelImg = cv2.imread(labelImgPath,0)
    print(labelImg)
    width = labelImg.shape[1]
    height = labelImg.shape[0]
    # labelImg = cv2.cvtColor(labelImg, cv2.COLOR_BGR2RGB)
    temp = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            if labelImg[i, j] == 0:
                temp[i, j] = [0, 0, 0]
            elif labelImg[i, j] == 1:
                temp[i, j] = [0, 0, 63]
            elif labelImg[i, j] == 2:
                temp[i, j] = [0, 63, 63]
            elif labelImg[i, j] == 3:
                temp[i, j] = [0, 63, 0]
            elif labelImg[i, j] == 4:
                temp[i, j] = [0, 63, 127]
            elif labelImg[i, j] == 5:
                temp[i, j] = [0, 63, 191]
            elif labelImg[i, j] == 6:
                temp[i, j] = [0, 63, 255]
            elif labelImg[i, j] == 7:
                temp[i, j] = [0, 127, 63]
            elif labelImg[i, j] == 8:
                temp[i, j] = [0, 127,127]
            elif labelImg[i, j] == 9:
                temp[i, j] = [0, 0, 127]
            elif labelImg[i, j] == 10:
                temp[i, j] = [0, 0, 191]
            elif labelImg[i, j] == 11:
                temp[i, j] = [0, 0, 255]
            elif labelImg[i, j] == 12:
                temp[i, j] = [0, 191, 127]
            elif labelImg[i, j] == 13:
                temp[i, j] = [0, 127, 191]
            elif labelImg[i, j] == 14:
                temp[i, j] = [0, 127, 255]
            elif labelImg[i, j] == 15:
                temp[i, j] = [0, 100, 155]
    cv2.imwrite('testLabelFunc.png', temp[:, :, ::-1])
    # cv2.waitKey(5000)


if __name__ == '__main__':
    # unlabelImgDir = '/home/ubuntu/lijiahao/Datasets/ISAID/train/visualize_gt'
    # labelImgDir = '/home/ubuntu/lijiahao/Datasets/ISAID/train/label'
    # label(unlabelImgDir, labelImgDir)
    # unlabelImgDir = '/home/ubuntu/lijiahao/Datasets/ISAID/valid/visualize_gt'
    # labelImgDir = '/home/ubuntu/lijiahao/Datasets/ISAID/valid/label'
    # label(unlabelImgDir, labelImgDir)
    labelImgPath = '/home/ubuntu/lijiahao/Datasets/ISAID/valid/label/P0003_instance_color_RGB.png'
    testLabelFunc(labelImgPath)