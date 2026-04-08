import numpy as np
import cv2
import os


def main():
	root = '/home/ubuntu/lijiahao/Datasets/Postdam'
	list_path = '/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/lib/data/postdam_list/train.lst'
	img_ids = [i_id.strip().split() for i_id in open(list_path)]
	# print(img_ids)
	# exit(1)
	for item in img_ids:
		label = cv2.imread(os.path.join(root, item[-1]))
		if np.amin(label) == 0:
			print('correct')
		# print(np.amax(label))
		# print(np.amin(label))

if __name__ == '__main__':
	main()