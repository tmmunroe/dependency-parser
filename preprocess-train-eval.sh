bash purge.sh

echo "Extracting vocab..."
python get_vocab.py data/train.conll data/words.vocab data/pos.vocab

echo "Extracting training and dev data..."
python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
python extract_training_data.py data/dev.conll data/input_dev.npy data/target_dev.npy

echo "Training model..."
python train_model.py data/input_train.npy data/target_train.npy data/model.h5

echo "Evaluating dev set..."
python evaluate.py data/model.h5 data/dev.conll

echo "Evaluating test set..."
python evaluate.py data/model.h5 data/test.conll

