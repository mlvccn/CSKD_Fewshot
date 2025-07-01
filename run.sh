python train.py \
    -project ours \
    -dataset mini_imagenet \
    -base_mode distill-clip-RN18-cos \
    -clip_model_path pretrain/clip/mini_imageNet.pth \
    -new_mode avg_cos \
    -gamma 0.1 \
    -lr_base 0.1 \
    -lr_new 0.1 \
    -decay 0.0005 \
    -epochs_base 120 \
    -schedule Milestone \
    -milestones 40 70 100 \
    -gpu 2,3 \
    -temperature 16 \
    -epochs_new 10 \
    -start_session 0 \
    -start_epoch 0\
    
