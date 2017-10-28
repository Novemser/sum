##mkdir -p fairseq_giga11/checkpoints_topic_softmax3
##CUDA_VISIBLE_DEVICES=1 python3 train.py data_giga_bin --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_giga --save-dir checkpoints/fconv
CUDA_VISIBLE_DEVICES=1 python3 train.py data_giga_bin -enable_topic -enable_rl --restore-file checkpoint_last.pt --save-interval 200 --lr 0.0001 --lrshrink 0.9 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --arch fconv_giga --save-dir fairseq_giga11/checkpoints_topic_softmax3 --max-len-b 20 --minlen 4 -hardset_lr --loss_scale 1.0
##CUDA_VISIBLE_DEVICES=1 python3 train.py data_giga_bin_small --lr 0.25 --clip-norm 0.1 --dropout 0.1 --max-tokens 4000 --arch fconv_giga --save-dir checkpoints_small/topic_add
