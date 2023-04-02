from decoder import Parser
from extract_training_data import FeatureExtractor
from conll_reader import conll_reader
import sys

words = [None, 'Spirit', 'of', 'Perestroika', 'Touches', 'Design', 'World']
pos = [None, 'NN', 'IN', 'FW', 'VBZ', 'NN', 'NN']

if __name__ == "__main__":
    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'
    model_file = 'data/model.h5'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, model_file)

    predict = parser.parse_sentence(words, pos)
    print()
