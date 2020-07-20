from itertools import chain

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from utils import tokenize

class TokenEncoder():
    """ Encode tokens into token ids.
    """
    def __init__(self):
        self.list_of_tokens = []  # map from token_id to token
        self.dict_of_tokens = {}  # map from token to token_id

    def fit(self, docs):
        for sentence in docs:
            for token in sentence:
                if token not in self.dict_of_tokens:
                    self.dict_of_tokens[token] = len(self.list_of_tokens)
                    self.list_of_tokens.append(token)

    def transform(self, docs):
        return [[self.dict_of_tokens[token] for token in sentence] for sentence in docs]

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)

    def inverse_transform(self, encoded_tokens):
        return [[self.list_of_tokens[token_id] for token_id in row] for row in encoded_tokens]

    @property
    def vocab_size(self):
        return len(self.list_of_tokens)

class SimpleWord2Vec():
    def __init__(
        self, 
        dimensions=10, 
        window_size=5, 
        tokenizer=tokenize, 
        token_encoder=TokenEncoder(),
    ):
        assert window_size % 2 == 1, 'Window size needs to be an odd number'
        assert window_size >= 3, 'Window size needs to be greater or equal to 3'
        self.n_dim = dimensions
        self.window_size = window_size
        self.max_j = (window_size - 1) // 2
        self.tokenize = tokenize
        self.token_encoder = token_encoder

    def initialize(self, seed=2020):
        """ Initialize word embedding matrices """
        np.random.seed(seed)
        # Input words weight matrix, this will be the word embedding after training
        self.W_i = np.random.rand(self.vocab_size, self.n_dim)
        # Ouput words weight metrix
        self.W_o = np.random.rand(self.vocab_size, self.n_dim)

    def train(self, docs):
        """ Train Simple Word2Vec on documents
        Parameters
        ----------
        docs: a list of strings (sentences)
        """
        # Tokenize documents
        doc_tokens = self.tokenize(docs)
        # Encode token
        encoded_doc_tokens = self.token_encoder.fit_transform(doc_tokens)
        self.vocab_size = self.token_encoder.vocab_size
        # Split context and center words
        context_words, center_words = SimpleWord2Vec.split_context_center(encoded_doc_tokens, self.max_j)
        # Initialization Word Embedding
        self.initialize()
        self.fit(context_words, center_words)

    def fit(
        self, context_words, center_words, 
        gradient=SimpleWord2Vec.cbow_gradient, loss_function=SimpleWord2Vec.cbow_loss,
        num_iterations=100, learning_rate=0.1, hist=[], verbose=1,
    ):
        W_i = self.W_i
        W_o = self.W_o
        loss = loss_function(context_words, center_words, W_i, W_o)
        print('Inital loss:', loss)
        hist.append(loss)
        for i in range(num_iterations):
            dLdW_i, dLdW_o = gradient(context_words, center_words, W_i, W_o)
            W_i -= learning_rate * dLdW_i
            W_o -= learning_rate * dLdW_o
            loss = loss_function(context_words, center_words, W_i, W_o)
            if verbose > 0 and i % verbose == 0:
                print(f'[{i+1} / {num_iterations}]', loss)
            hist.append(loss)
        print('Final loss:', loss)
        self.W_i = W_i
        self.W_o = W_o

    def transform(self, token_ids):
        """ Predict the word vectors for a list of token ids """
        return self.W_i[token_ids]

    def predict(self, words):
        """ Predict the word vectors for a list of words 
        Parameters
        ----------
        words: list of strings (words)

        Returns
        -------
        2D Numpy array of the shape (len(words), vocab_size)
        """
        return self.transform(self.token_encoder.transform([words])[0])

    @staticmethod
    def split_context_center(doc_tokens, max_j):
        """ Split sentences into context words and center words
        Parameters
        ----------
        doc_tokens: list of list of token
        max_j: int
            the number of context words on one side of the center word

        Returns
        -------
        Tuple of a list of list of context words and a list of centers words
        """
        context_words = [
            sentence_tokens[max(i - max_j, 0): i] + sentence_tokens[i + 1: i + max_j + 1]
            for sentence_tokens in doc_tokens
            for i in range(len(sentence_tokens))        
        ]
        center_words = list(chain.from_iterable(doc_tokens))
        return context_words, center_words

    @staticmethod
    def cbow_gradient(context_words, center_words, W_i, W_o):
        return
    
    @staticmethod
    def cbow_loss(context_words, center_words, W_i, W_o):
        return
