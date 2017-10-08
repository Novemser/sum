mkdir -p checkpoints/fconv
##CUDA_VISIBLE_DEVICES=1 python3 train.py data_giga_bin --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_giga --save-dir checkpoints/fconv
python3 train.py data_giga_bin --lr 0.5 --clip-norm 0.1 --dropout 0.15 --max-tokens 4200  --max-len-b 50 --arch fconv_giga --save-dir checkpoints/fconv --cuda-visible-devices 1 --workers 2 --save-interval 3000
