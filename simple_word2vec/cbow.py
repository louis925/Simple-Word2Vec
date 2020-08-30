""" Continuous Bag of Words (CBOW)
"""

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax

# Continuous Bag of Words (CBOW)
def predict_prob(input_tokens, W_i, W_o):
    """
    Predict the probability of each words being the center word given a list of intput
    context words in CBOW
    input_tokens: a list of context word tokens
    """
    return softmax(np.dot(W_o, W_i[input_tokens].mean(axis=0)))


def predict(input_tokens, W_i, W_o):
    """ Predict the most likely center word given a list of input context words in CBOW """
    return np.argmax(np.dot(W_o, W_i[input_tokens].mean(axis=0)))


def prob(input_tokens, output_token, W_i, W_o):
    """ Probability of getting an output word given the input words in CBOW """
    return predict_prob(input_tokens, W_i, W_o)[output_token]


def single_loss(input_tokens, output_token, W_i, W_o):
    """ Loss (-log(probability)) between the input words and a true output word in CBOW """
    return -log_softmax(np.dot(W_o, W_i[input_tokens].mean(axis=0)))[output_token]


def loss(context_words, center_words, W_i, W_o):
    """Loss function of the document in CBOW

    Parameters
    ----------
    context_words: list of list of int
        a list of list of context words corresponding to each center word
    center_words: list of int
        a list of center words
    W_i: 2D np.array of float32
        Input words weight matrix. This is the word embedding.
    W_o: 2D np.array of float32
        Output words weight matrix

    Return
    ------
    loss: float
    """
    return np.sum(
        [
            single_loss(context_tokens, center_token, W_i, W_o)
            for context_tokens, center_token in zip(context_words, center_words)
        ]
    )


def gradient(context_words, center_words, W_i, W_o):
    """Gradiences of the loss function with respect to the word embedding matrices in CBOW

    Parameters
    ----------
    context_words: list of list of int
        a list of list of context words corresponding to each center word
    center_words: list of int
        a list of center words
    W_i: 2D np.array of float32
        Input words weight matrix. This is the word embedding.
    W_o: 2D np.array of float32
        Output words weight matrix

    Returns
    -------
    dLdW_i, dLdW_o: tuple of two 2D np.array of the same shape of W_i and W_o
        Gradient of the loss function with respect to W_i and W_o.
    """
    context_vectors = np.array([W_i[words].mean(axis=0) for words in context_words])
    scalar_products = np.matmul(context_vectors, W_o.T)
    softmax_factors = softmax(scalar_products, axis=1)
    # Compute gradient with respect to W_i
    len_context_words = np.array([len(x) for x in context_words])
    A_tj = (W_o[center_words] - np.matmul(softmax_factors, W_o)) / len_context_words.reshape(-1, 1)
    dLdW_i = np.zeros_like(W_i)
    for words, a_t in zip(context_words, A_tj):
        dLdW_i[words] += a_t
    dLdW_i = -dLdW_i / len(context_words)
    # Compute gradient with respect to W_o
    B_ti = -softmax_factors
    for t, center_word in enumerate(center_words):
        B_ti[t, center_word] += 1
    dLdW_o = -np.matmul(B_ti.T, context_vectors) / len(context_words)
    return dLdW_i, dLdW_o
