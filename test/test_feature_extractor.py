from extract_training_data import State, FeatureExtractor, apply_sequence

word_vocab_file = 'data/words.vocab'
pos_vocab_file = 'data/pos.vocab'

with open(word_vocab_file,'r') as word_vocab:
    with open(pos_vocab_file,'r') as pos_vocab:
        extractor = FeatureExtractor(word_vocab, pos_vocab)


def test_get_input_representation():
    words = "a cat ate the silly mouse".split()
    pos = "DT NN V DT ADJ NN".split()
    state = State(words)

    input_representation = extractor.get_input_representation(words, pos, state)
    expected_output = None