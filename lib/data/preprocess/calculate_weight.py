import numpy as np
import os
import cv2
from tqdm import tqdm


NUM_CLASSES = 8


def calculate_weight(label_path):
    label_name = sorted(os.listdir(label_path))
    pixel_list = list()
    for i in tqdm(range(len(label_name))):
        # if i >= 90000:
        #     break
        pixel_list_per_img = list()
        label = cv2.imread(os.path.join(label_path, label_name[i]), cv2.IMREAD_GRAYSCALE)
        for x in range(label.shape[0]):
            for y in range(label.shape[1]):
                pixel_list_per_img.append(label[x, y])
        pixel_list.append(pixel_list_per_img)

    # assert len(pixel_list) == 90000
    assert len(pixel_list) == len(label_name)

    cal_ndarray = np.zeros((NUM_CLASSES,))
    for i in range(len(pixel_list)):
        cal_list_per_img = list(np.bincount(pixel_list[i]))
        if len(cal_list_per_img) < 8:
            for i in range(8 - len(cal_list_per_img)):
                cal_list_per_img.append(0)
        cal_ndarray += np.asarray(cal_list_per_img)
    print(cal_ndarray)

    pixel_sum = 0
    for i in range(NUM_CLASSES):
        pixel_sum += cal_ndarray[i]

    # assert pixel_sum == 256 * 256 * 90000
    assert pixel_sum == 256 * 256 * 100000

    cal_ndarray = cal_ndarray / pixel_sum

    return cal_ndarray


def main():
    label_path = '/media/fly/4898FC1598FC02EC/lijiahao/AI_data/largeAI/train/label'
    cal_ndarray = calculate_weight(label_path)
    print(cal_ndarray)
    print(1 / cal_ndarray)


if __name__ == '__main__':
    main()