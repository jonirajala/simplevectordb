import numpy as np


class VectorDB:
    def __init__(self, vector_size, init_db=None):
        if init_db:
            self.db = np.array((init_db))
        else:
            self.db = np.zeros_like((1, vector_size))
    
    def get(self, prompt):
        vector = self.vectorize(prompt)
        top_k = self.k_nearest(vector)
        return top_k
    
    def vectorize(self, prompt):
        pass

    def k_nearest(self, vector):
        pass

