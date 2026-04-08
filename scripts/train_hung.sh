CUDA_VISIBLE_DEVICES=0,1

echo 'HUNG_resnet50-FCN8S_512X512_train25_sgd3e-4_bs2_epoch150_TEMP1'
python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train_hung.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/postdam/HUNG_resnet50-FCN8S_512X512_train25_sgd3e-4_bs2_epoch150_TEMP1.yaml'

echo 'all done'