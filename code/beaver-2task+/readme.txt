# pytorch >= 1.0  python 3.6

1. Build vocab

cd tools && python build_vocab $number < $textfile > $vocab_file

2. Start training (for CLS+MT)

python train.py -config run_config/train-example.json

3. Start decoding
python translate.py -config run_config/decode-example.json
