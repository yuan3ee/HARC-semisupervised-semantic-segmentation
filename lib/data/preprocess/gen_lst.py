import os
from tqdm import tqdm

# this_dir = os.getcwd()
# print(this_dir)
# root_dir = os.path.abspath(os.path.dirname(this_dir))
# print(root_dir)
list_dir ='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/lib/data/isaid_list'
# label_dir = os.path.join(root_dir, 'water_extract/labels')


def gen_train_lst(root_dir):
    '''生成train.lst'''
    train_dir = os.path.join(root_dir, 'img')
    train_list = os.listdir(train_dir)
    train_list = sorted(train_list)

    train_file = os.path.join(list_dir, 'train.lst')
    with open(train_file, 'w') as f:
        for i in tqdm(range(len(train_list))):
            # if i >= 90000:
            #     break
            train_name = 'train/img' + '/' + train_list[i]
            label_name = 'train/label' + '/' + train_list[i].split('.')[0] + '_instance_color_RGB.png'
            f.write(train_name)
            f.write('\t')
            f.write(label_name)
            if i != len(train_list)-1:
                f.write('\n')


def gen_val_lst(root_dir):
    '''生成val.lst'''
    val_dir = os.path.join(root_dir, 'img')
    val_list = os.listdir(val_dir)
    val_list = sorted(val_list)

    train_file = os.path.join(list_dir, 'val.lst')
    with open(train_file, 'w') as f:
        for i in tqdm(range(len(val_list))):
            # if i < 90000:
            #     continue
            val_name = 'valid/img' + '/' + val_list[i]
            label_name = 'valid/label' + '/' + val_list[i].split('.')[0] + '_instance_color_RGB.png'
            f.write(val_name)
            f.write('\t')
            f.write(label_name)
            if i != len(val_list)-1:
                f.write('\n')


if __name__ == '__main__':
    # print(this_dir)
    # print(root_dir)
    gen_train_lst('/home/ubuntu/lijiahao/Datasets/ISAID/train')
    gen_val_lst('/home/ubuntu/lijiahao/Datasets/ISAID/valid')