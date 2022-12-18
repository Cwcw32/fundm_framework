
lr=2e-5
fuse_lr=2e-5
image_output_type=all
optim=adamw
data_type=MVSA-single
train_fuse_model_epoch=20
epoch=30
warmup_step_epoch=2
text_model=bert-base
fuse_type=att
batch=32
acc_grad=4
tran_num_layers=3
image_num_layers=2
CUDA_VISIBLE_DEVICES=0,1,2,3 ssh://bufanxu@10.10.255.119:22/home/bufanxu/anaconda3/envs/CLMLF/bin/python -u /home/bufanxu/.pycharm_helpers/pydev/pydevd.py
--multiproc --qt-support=auto --client 0.0.0.0 --port 39397 -m torch.distributed.launch



 --nproc_per_node=4
 --master_port=22332
--file /home/bufanxu/codes/CLMLF/main.py
-cuda
-gpu_num '0,1,2,3'
-epoch ${epoch}  \
        -add_note ${data_type}-${tran_num_layers}-${fuse_type}-${batch}-${image_num_layers}-${text_model} \
        -data_type ${data_type} -text_model ${text_model} -image_model resnet-50 \
        -batch_size ${batch} -acc_grad ${acc_grad} -fuse_type ${fuse_type} -image_output_type ${image_output_type} -fixed_image_model \
        -data_path_name 10-flod-1 -optim ${optim} -warmup_step_epoch ${warmup_step_epoch} -lr ${lr} -fuse_lr ${fuse_lr} \
        -tran_num_layers ${tran_num_layers} -image_num_layers ${image_num_layers} -train_fuse_model_epoch ${train_fuse_model_epoch}
