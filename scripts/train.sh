# cd /home/ubuntu/lijiahao/Semantic_Segmentation/CCT

# python train.py
# sleep 60s

# cd /home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/scripts

CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/isaid/SEMI_resnet50-FCN8S_1024X1024_train12-5_sgd3e-4_bs2_epoch150_TEMP2.yaml'
# sleep 60s
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/isaid/SEMI_resnet50-FCN8S_1024X1024_train25_sgd3e-4_bs2_epoch150_TEMP2.yaml'
# sleep 60s
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train_supervised.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/isaid/RESNET50-FCN8S_512X512_train100_sgd3e-4_bs2_epoch150_TEMP1.yaml'


echo 'all done'