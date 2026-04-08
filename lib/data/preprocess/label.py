import cv2
import os
from PIL import Image
import numpy as np

LABELLIST = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]


def label(unlabelImgDir, labelImgDir):
    unlabelImgList = os.listdir(unlabelImgDir)
    unlabelImgList = sorted(unlabelImgList)
    # print(range(len(unlabelImgList)))
    for i in range(len(unlabelImgList)):
        unlabelImgPath = os.path.join(unlabelImgDir, unlabelImgList[i])
        unlabelImg = cv2.imread(unlabelImgPath)
        unlabelImg = cv2.cvtColor(unlabelImg, cv2.COLOR_BGR2RGB)
        changed = np.zeros(unlabelImg.shape[:2])
        width = unlabelImg.shape[0]
        height = unlabelImg.shape[1]
        temp = unlabelImg.copy()
        for j in range(len(LABELLIST)):
            changed[np.where(np.all(temp==LABELLIST[j],axis=-1))] = j
        cv2.imwrite(os.path.join(labelImgDir, unlabelImgList[i]), changed)


def testLabelFunc(labelImgPath):
    labelImg = cv2.imread(labelImgPath,0)
    print(labelImg.shape)
    # labelImg = cv2.cvtColor(labelImg, cv2.COLOR_BGR2RGB)
    temp = np.zeros((512,512,3))
    width = labelImg.shape[0]
    height = labelImg.shape[1]
    # exit(2)
    # for i in range
    for i in range(width):
        for j in range(height):
            if labelImg[i, j] == 0:
                temp[i, j] = [255, 255, 255]
            elif labelImg[i, j] == 1:
                temp[i, j] = [0, 0, 255]
            elif labelImg[i, j] == 2:
                temp[i, j] = [0, 255, 255]
            elif labelImg[i, j] == 3:
                temp[i, j] = [0, 255, 0]
            elif labelImg[i, j] == 4:
                temp[i, j] = [255, 255, 0]
            elif labelImg[i, j] == 5:
                temp[i, j] = [255, 0, 0]
    # for i in range(len(LABELLIST)):
    #     labelImg[temp == i] = LABELLIST[i]
    cv2.imshow('testLabelFunc', temp[:,:,::-1])
    cv2.waitKey(5000)


if __name__ == '__main__':
    # unlabelImgDir = '/home/ubuntu/lijiahao/datasets/Vaihingen/train/visualize_gt'
    # labelImgDir = '/home/ubuntu/lijiahao/datasets/Vaihingen/train/label'
    # label(unlabelImgDir, labelImgDir)
    labelImgPath = '/home/ubuntu/lijiahao/datasets/Vaihingen/train/label/1_00000.png'
    testLabelFunc(labelImgPath)