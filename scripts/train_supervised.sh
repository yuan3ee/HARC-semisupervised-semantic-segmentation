CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch --nproc_per_node=2 --master_port=25902 ../train_supervised.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/RESNET50-UNet_512X512_train100_sgd3e-4_bs2_epoch150_TEMP1.yaml'