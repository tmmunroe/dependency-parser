import copy
import numpy as np

from extract_training_data import State, FeatureExtractor, apply_sequence

word_vocab_file = 'data/words.vocab'
pos_vocab_file = 'data/pos.vocab'

with open(word_vocab_file,'r') as word_vocab:
    with open(pos_vocab_file,'r') as pos_vocab:
        extractor = FeatureExtractor(word_vocab, pos_vocab)


def test_pad_list():
    x = [0]
    p = extractor._pad_list(x, -1, 3)
    assert p == [0,-1,-1]

    x = [0, 1]
    p = extractor._pad_list(x, -1, 3)
    assert p == [0,1,-1]

    x = [0, 1, 2]
    p = extractor._pad_list(x, -1, 3)
    assert p == [0,1,2]

def test_extract_word_and_pos():
    words = [None, "a", "b", "critical", "Capital", "expialadocious", "unknownpos"]
    pos = [None, "NNP", "CD", "NN", "NN", "NN", "ZZZ"]
    
    w, p = extractor._extract_word_and_pos(-1, words, pos)
    assert (w, p) == ("<NULL>", "<NULL>")
    
    w, p = extractor._extract_word_and_pos(0, words, pos)
    assert (w, p) == ("<ROOT>", "<ROOT>")
    
    w, p = extractor._extract_word_and_pos(1, words, pos)
    assert (w, p) == ("<NNP>", "NNP")
    
    w, p = extractor._extract_word_and_pos(2, words, pos)
    assert (w, p) == ("<CD>", "CD")
    
    w, p = extractor._extract_word_and_pos(3, words, pos)
    assert (w, p) == ("critical", "NN")
    
    w, p = extractor._extract_word_and_pos(4, words, pos)
    assert (w, p) == ("capital", "NN")
    
    w, p = extractor._extract_word_and_pos(5, words, pos)
    assert (w, p) == ("<UNK>", "NN")
    
    w, p = extractor._extract_word_and_pos(6, words, pos)
    assert (w, p) == ("<UNK>", "<UNK>")


def test_get_input_representation():
    word_vocab_file = 'test/data/words.vocab'
    pos_vocab_file = 'test/data/pos.vocab'

    with open(word_vocab_file,'r') as word_vocab:
        with open(pos_vocab_file,'r') as pos_vocab:
            loc_extractor = FeatureExtractor(word_vocab, pos_vocab)


    words = [None] + "Cooperman used about 56 words defending the witnesses ' constitutional rights .".split()
    pos = [None] + "NNP VBD RB CD NNS VBG DT NNS POS JJ NNS .".split()

    """
    (0, (None, None))
    (1, ('Cooperman', 'NNP'))
    (2, ('used', 'VBD'))
    (3, ('about', 'RB'))
    (4, ('56', 'CD'))
    (5, ('words', 'NNS'))
    (6, ('defending', 'VBG'))
    (7, ('the', 'DT'))
    (8, ('witnesses', 'NNS'))
    (9, ("'", 'POS'))
    (10, ('constitutional', 'JJ'))
    (11, ('rights', 'NNS'))
    (12, ('.', '.'))
    """
    # case 1: empty stack
    state = State()
    state.stack = []
    state.buffer = [12, 11, 0]

    input_representation = loc_extractor.get_input_representation(words, pos, state)
    expected_output = np.array([4, 4, 4, 3, 11, 22])
    assert (input_representation == expected_output).all()

    # case 2: buffer has 1
    state = State()
    state.stack = [12, 11]
    state.buffer = [0]

    input_representation = loc_extractor.get_input_representation(words, pos, state)
    expected_output = np.array([11, 22, 4, 3, 4, 4])

    assert (input_representation == expected_output).all()

    # case 3: both have 3
    state = State()
    state.stack = [12, 11, 10]
    state.buffer = [0, 1, 2]

    input_representation = loc_extractor.get_input_representation(words, pos, state)
    expected_output = np.array([ 6, 11, 22, 5, 1, 3])
    assert (input_representation == expected_output).all()

    # case 3: both have >3
    state = State()
    state.stack = [12, 11, 10, 9]
    state.buffer = [0, 1, 2, 3, 4]

    input_representation = loc_extractor.get_input_representation(words, pos, state)
    expected_output = np.array([9, 6, 11, 0, 16, 5])
    assert (input_representation == expected_output).all()


def test_get_output_representation():
    output_rep_r = extractor.get_output_representation(("right_arc", "tmod"))
    assert output_rep_r.shape == (91,)
    assert output_rep_r[2] == 1 and (output_rep_r[0:2] == 0).all() and (output_rep_r[3:] == 0).all()

    output_rep_l = extractor.get_output_representation(("left_arc", "det"))
    assert output_rep_l.shape == (91,)
    assert output_rep_l[89] == 1 and (output_rep_l[0:89] == 0).all() and output_rep_l[90] == 0

    output_rep_s = extractor.get_output_representation(("shift", None))
    assert output_rep_s.shape == (91,)
    assert output_rep_s[0] == 1 and (output_rep_s[1:] == 0).all()
