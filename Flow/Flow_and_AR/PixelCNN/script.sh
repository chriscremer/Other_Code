





python main.py \
       --exp_name='test_learn' \
       --dataset=cifar \
       --data_dir="$HOME/Documents/" \
       --save_dir="$HOME/Documents/pixelcnnpp_test/" \
       --nr_filters=160 \
       --nr_logistic_mix=5 \
       --batch_size=1



# python train_student.py \
#        --exp_name='test_student' \
#        --dataset=cifar \
#        --data_dir="$HOME/Documents/" \
#        --save_dir="$HOME/Documents/pixelcnnpp_test/" \
#        --load_teacher="$HOME/Documents/pixelcnnpp_test/test/params/pcnn_lr:0.00020_nr-resnet5_nr-filters80_62.pth" \
#        --nr_filters=80 \
#        --nr_logistic_mix=5 \
#        --batch_size=2







