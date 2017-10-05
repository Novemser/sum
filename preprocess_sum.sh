export TEXT="$(pwd)/giga_data_small"

python3 preprocess.py --source-lang art --target-lang sum --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data_giga_bin_small
