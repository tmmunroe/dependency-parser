
echo "Evaluating dev set..."
python evaluate.py data/model.h5 data/dev.conll

echo "Evaluating test set..."
python evaluate.py data/model.h5 data/test.conll

echo "Evaluating sec0 set..."
python evaluate.py data/model.h5 data/sec0.conll

