python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy

python extract_training_data.py data/dev.conll data/input_dev.npy data/target_dev.npy

python train_model.py data/input_train.npy data/target_train.npy data/model.h5

python evaluate.py data/model.h5 data/dev.conll
