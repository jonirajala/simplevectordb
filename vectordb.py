import numpy as np
import torch
from word2vev import Word2Vec
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

def find_closest_words(word, model, word_to_index, top_n=5):
    """
    Find the closest words to the given word using cosine similarity.
    Args:
        word (str): Target word.
        model (Word2Vec): Trained Word2Vec model.
        word_to_index (dict): Mapping of words to indices.
        top_n (int): Number of closest words to return.
    Returns:
        list: List of tuples (word, similarity).
    """
    if word not in word_to_index:
        raise ValueError(f"Word '{word}' not found in the vocabulary.")

    # Get the embedding of the target word
    word_idx = word_to_index[word]
    word_embedding = model.embedding.weight[word_idx].detach().cpu().numpy()

    # Compute cosine similarity with all other embeddings
    embeddings = model.embedding.weight.detach().cpu().numpy()
    similarities = np.dot(embeddings, word_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(word_embedding)
    )

    # Get the top N most similar words (excluding the target word itself)
    most_similar_indices = similarities.argsort()[::-1][1 : top_n + 1]
    closest_words = [(list(word_to_index.keys())[idx], similarities[idx]) for idx in most_similar_indices]

    return closest_words
    

device = "cpu"


checkpoint = torch.load("word2vec_full_skipgram.pt")
vocab_size, embedding_dim = checkpoint["vocab_size"], checkpoint["embedding_dim"]
word2vec = Word2Vec(vocab_size, embedding_dim)
# If you saved the full state (with optimizer):

word2vec.load_state_dict(checkpoint['model_state_dict'])
word_to_index = checkpoint['word_to_index']  # Restore word-to-index mapping
print("Full model and optimizer state restored!")
print(list(word_to_index)[:100])

target_word = "metsä"  # Replace with a Finnish word in your vocabulary
try:
    closest_words = find_closest_words(target_word, word2vec, word_to_index, top_n=5)
    print(f"Closest words to '{target_word}':")
    for word, similarity in closest_words:
        print(f"{word}: {similarity:.4f}")
except ValueError as e:
    print(e)