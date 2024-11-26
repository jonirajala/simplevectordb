import pytorch 

import torch
import numpy as np

def generate_data_skipgram(corpus, window_size, V):
    maxlen = window_size * 2
    all_in = []
    all_out = []
    for words in corpus:
        L = len(words)
        for index, word in enumerate(words):
            p = index - window_size
            n = index + window_size + 1

            in_words = []
            labels = []
            for i in range(p, n):
                if i != index and 0 <= i < L:
                    # Add the input word
                    all_in.append(word)
                    # Add one-hot of the context words
                    one_hot = np.eye(V)[words[i]]
                    all_out.append(one_hot)

    return (np.array(all_in), np.array(all_out))


