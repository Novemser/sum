###python3 generate.py data_giga_bin --path checkpoints/fconv/checkpoint_best.pt --batch-size 128 --beam 5

CUDA_VISIBLE_DEVICES=1 python3 generate.py data_giga_bin --path checkpoints/fconv/checkpoint_best.pt --beam 5 --batch-size 128 --lenpen 0.5 --remove-bpe | tee > ./output/result


: <<END
...
| Translated 3003 sentences (95451 tokens) in 81.3s (1174.33 tokens/s)
| Generate test with beam=5: BLEU4 = 40.23, 67.5/46.4/33.8/25.0 (BP=0.997, ratio=1.003, syslen=80963, reflen=81194)
END

# Scoring with score.py:
grep ^H ./output/result | cut -f3- > ./output/gen.out.sys
grep ^T ./output/result | cut -f2- > ./output/gen.out.ref
python3 score.py --sys ./output/gen.out.sys --ref ./output/gen.out.ref
##BLEU4 = 40.23, 67.5/46.4/33.8/25.0 (BP=0.997, ratio=1.003, syslen=80963, reflen=81194)
