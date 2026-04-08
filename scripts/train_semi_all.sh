CUDA_VISIBLE_DEVICES=0,1

echo 'vaihingen/SEMI_resnet50-UNet_512X512_train100_sgd3e-4_bs2_epoch150_TEMP1'
python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train_semi_all.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-UNet_512X512_train100_sgd3e-4_bs2_epoch150_TEMP1.yaml'

echo 'potsdam/SEMI_resnet50-UNet_512X512_train100_sgd3e-4_bs2_epoch150_TEMP1'
python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train_semi_all.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/postdam/SEMI_resnet50-UNet_512X512_train100_sgd3e-4_bs2_epoch150_TEMP1.yaml'

echo 'all done'