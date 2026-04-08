CUDA_VISIBLE_DEVICES=0,1

# echo 'SEMI_HRNet-OCR_512X512_train100_sgd3e-4_bs2_epoch150_TEMP1'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25902 ../train_semi_all.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_HRNet-OCR_512X512_train100_sgd3e-4_bs2_epoch150_TEMP1.yaml'
# sleep 300s

# echo 'SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP1'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP1.yaml'
# sleep 300s

echo 'S4GAN_HRNet-OCR_512X512_train100_sgd3e-4_bs2_epoch150_TEMP1'
python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train_s4GAN_semi_all.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/S4GAN_HRNet-OCR_512X512_train100_sgd3e-4_bs2_epoch150_TEMP1.yaml'
# sleep 300s

# echo 'SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP3'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP3.yaml'
# sleep 300s

# echo 'SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP4'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP4.yaml'
# sleep 300s

# echo 'SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP5'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP5.yaml'
# sleep 300s

# echo 'SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP6'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP6.yaml'
# sleep 300s

# echo 'SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP7'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP7.yaml'
# sleep 300s

############################################################################################################################################################################################################################################################
# echo 'RESNET50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP1'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25903 ../train_full.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/RESNET50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP1.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP1'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP1.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP2'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP2.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP3'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP3.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP4'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP4.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP5'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP5.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP6'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP6.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP7'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_CCT_TEMP7.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP8'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP8.yaml'
# sleep 5s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP9'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP9.yaml'
# sleep 5s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP10'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP10.yaml'
# sleep 5s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP11'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP11.yaml'
# sleep 5s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP12'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP12.yaml'
# sleep 5s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP13'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP13.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP14'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP14.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP15'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP15.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP16'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP16.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP17'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP17.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP18'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP18.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP19'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP19.yaml'
# sleep 300s

# echo 'SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP20'
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=25901 ../train.py --cfg='/home/ubuntu/lijiahao/Semantic_Segmentation/S4GAN-MLMT/experiments/vaihingen/SEMI_resnet50-FCN8S_512X512_train12-5_sgd3e-4_bs2_epoch150_TEMP20.yaml'
# sleep 300s

echo 'all done'