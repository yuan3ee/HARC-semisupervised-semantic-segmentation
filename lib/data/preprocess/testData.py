import os

root = '/home/ubuntu/lijiahao/HRNet-OCR/data'
list_path = '/list/vaihingen/train.lst'
# print([line.strip().split() for line in open(root+list_path)])
img_list = [line.strip().split() for line in open(root+list_path)]
for item in img_list:
    image_path, label_path = item
    name = os.path.splitext(os.path.basename(item[0]))[0]
    print(image_path)
    exit(2)