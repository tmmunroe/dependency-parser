from decoder import Parser
from extract_training_data import FeatureExtractor
from train_model import build_model
from extract_training_data import get_training_matrices, get_training_instances
from conll_reader import conll_reader
import sys
import numpy as np

WORD_VOCAB_FILE = 'test/data/words.vocab'
POS_VOCAB_FILE = 'test/data/pos.vocab'
conll_file = 'test/data/test.conll'
model_file = 'test/data/model.h5'


class StubPerfectModel:
    def __init__(self, extractor: FeatureExtractor, conll_filename:str):
        self.input2action = {}
        with open(conll_filename, 'r') as in_file:
            for dtree in conll_reader(in_file):
                words = dtree.words()
                pos = dtree.pos()
                for state, output_pair in get_training_instances(dtree):
                    input_representation = extractor.get_input_representation(words, pos, state)
                    output_representation = extractor.get_output_representation(output_pair)
                    self.input2action[tuple(input_representation.ravel())] = output_representation
                break

    def __call__(self, input_representation: np.ndarray):
        return self.input2action[tuple(input_representation.ravel())]


def generate_model():
    # just builds an empty model -- we don't use this to test the parser
    #  we just use it to satisfy the parser's constructor interface
    # testing the parser depends on the stub model
    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
        # conll_file_f = open(conll_file,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    model.save(model_file)


def compare_parser(target, predict):
    target_unlabeled = set((d.id,d.head) for d in target.deprels.values())
    target_labeled = set((d.id,d.head,d.deprel) for d in target.deprels.values())
    predict_unlabeled = set((d.id,d.head) for d in predict.deprels.values())
    predict_labeled = set((d.id,d.head,d.deprel) for d in predict.deprels.values())

    labeled_correct = len(predict_labeled.intersection(target_labeled))
    unlabeled_correct = len(predict_unlabeled.intersection(target_unlabeled))
    num_words = len(predict_labeled)
    return labeled_correct, unlabeled_correct, num_words 


def test_parser():
    generate_model()

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, model_file)

    
    stub_model = StubPerfectModel(extractor, conll_file)
    parser.model = stub_model

    total_labeled_correct = 0
    total_unlabeled_correct = 0
    total_words = 0

    las_list = []
    uas_list = []

    count = 0 
    with open(conll_file,'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            predict = parser.parse_sentence(words, pos)
            labeled_correct, unlabeled_correct, num_words = compare_parser(dtree, predict)
            las_s = labeled_correct / float(num_words)
            uas_s = unlabeled_correct / float(num_words)
            las_list.append(las_s)
            uas_list.append(uas_s)
            total_labeled_correct += labeled_correct
            total_unlabeled_correct += unlabeled_correct
            total_words += num_words
            count +=1 
    print()

    las_micro = total_labeled_correct / float(total_words)
    uas_micro = total_unlabeled_correct / float(total_words)

    las_macro = sum(las_list) / len(las_list)
    uas_macro = sum(uas_list) / len(uas_list)

    print("{} sentence.\n".format(len(las_list)))
    print("Micro Avg. Labeled Attachment Score: {}".format(las_micro))
    print("Micro Avg. Unlabeled Attachment Score: {}\n".format(uas_micro))
    print("Macro Avg. Labeled Attachment Score: {}".format(las_macro))
    print("Macro Avg. Unlabeled Attachment Score: {}".format(uas_macro))

    # expect all the same given the size of the model
    assert las_micro == las_macro == uas_micro == uas_macro == 1.0
