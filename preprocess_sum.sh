export TEXT="$(pwd)/giga_data2"

python3 preprocess.py --source-lang art --target-lang sum --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data_giga_bin2
