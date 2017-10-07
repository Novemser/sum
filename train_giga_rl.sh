mkdir -p checkpoints/fconv_rl
##CUDA_VISIBLE_DEVICES=1 python3 train.py data_giga_bin --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_giga --save-dir checkpoints/fconv
python3 train.py data_giga_bin --lr 0.25 --clip-norm 0.1 --dropout 0.1 --max-tokens 4000  --max-len-b 100 --arch fconv_giga --save-dir checkpoints/fconv_rl -enable_rl --loss_scale 0.99 --cuda-visible-devices 1 --workers 2 --save-interval 500
