mkdir -p checkpoints/fconv
##CUDA_VISIBLE_DEVICES=1 python3 train.py data_giga_bin --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_giga --save-dir checkpoints/fconv
CUDA_VISIBLE_DEVICES=1 python3 train.py data_giga_bin --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_giga --save-dir checkpoints/fconv
