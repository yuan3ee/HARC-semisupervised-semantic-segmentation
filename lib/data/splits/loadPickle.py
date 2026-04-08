import pickle


def loadPickle(picklePath):
	train_ids = pickle.load(open(picklePath, 'rb'))
	print(train_ids)
	print(len(train_ids))


if __name__ == '__main__':
	picklePath = '/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/splits/vaihingen/vaihigen.pkl'
	loadPickle(picklePath)