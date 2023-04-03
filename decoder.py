from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys
from typing import List, Tuple

import numpy as np
import keras
import tensorflow as tf

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor: FeatureExtractor, modelfile:str):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def _apply_transition(self, state: State, action: str, deprel: str) -> None:
        if action[0] == 's':
            state.shift()
        elif action[0] == 'r':
            state.right_arc(deprel)
        elif action[0] == 'l':
            state.left_arc(deprel)
        return None

    def _is_valid_action(self, state: State, action: str) -> bool:
        # precondition: buffer has elements in it
        if action[0] == 's':
            # stack is empty (can always shift if stack is empty)
            # OR, if stack is not empty, we can do this if the
            #   buffer won't be emptied by this shift..
            #   since stack is not empty if we check this condition,
            #   there are elements other than the root remaining
            #   to be parsed.. therefore, we need to avoid emptying the buffer,
            #   which would leave dangling words
            return (
                len(state.stack) == 0
                or len(state.buffer) > 1       
            )
        elif action[0] == 'r':
            # right arc is always possible if there are elements in both stack and buffer..
            #  keep in mind, since the root is the leftmost element, there is no risk of it
            #  being the child in a right arc
            return len(state.stack) > 0
        else: # left arc
            # left arc is possible if there are elements in both stack and buffer
            # AND, the root will not be the child of the arc (i.e. root is not at top of the stack)
            # this can be reduced to checking that the length of the stack is > 1.. 
            # i.e., root + at least one other element
            return len(state.stack) > 1

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer:
            # get features for neural network
            encoded_state = self.extractor.get_input_representation(words, pos, state)

            # get neural network prediction
            transition_probabilities = self.model(encoded_state.reshape(-1, 6))
            transition_probabilities = tf.reshape(transition_probabilities, -1)

            # sort indexes (actions) by probability, note that this will put lowest probability first
            transition_indexes_sorted = np.argsort(transition_probabilities)
            
            # iterate through indexes (actions) until a valid action is found
            action, deprel = None, None
            for transition_id in reversed(transition_indexes_sorted.ravel()):
                action, deprel = self.output_labels[transition_id]
                if self._is_valid_action(state, action):
                    break
            else:
                # sanity check
                raise ValueError('Expected at least one valid transition, but found none!')

            # apply transition to state
            self._apply_transition(state, action, deprel)
            

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
