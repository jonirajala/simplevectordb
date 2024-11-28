from torch import nn
import torch
from torch.optim import AdamW
import numpy as np
import re
from nltk.tokenize import word_tokenize

def generate_data_skipgram(corpus, window_size, vocab_size, word_to_index):
    all_in = []
    all_out = []
    for words in corpus:
        L = len(words)
        for index, word in enumerate(words):
            p = index - window_size
            n = index + window_size + 1
            for i in range(p, n):
                if i != index and 0 <= i < L:
                    # Input word as index
                    all_in.append(word_to_index[word])
                    # Context word as one-hot
                    one_hot = np.zeros(vocab_size)
                    one_hot[word_to_index[words[i]]] = 1
                    all_out.append(one_hot)

    return np.array(all_in), np.array(all_out)

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.out = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        emb = self.embedding(x)
        out = self.out(emb)
        return self.softmax(out)

import wikipediaapi

# Initialize the Wikipedia API for Finnish
user_agent = "YourAppName/1.0 (your.email@example.com)"
wiki = wikipediaapi.Wikipedia(language="fi", user_agent=user_agent)

def get_category_members(category_name, max_pages=10):
    category = wiki.page("Luokka:" + category_name)
    if not category.exists():
        print(f"Category '{category_name}' does not exist.")
        return []

    articles = []
    for member in category.categorymembers.values():
        if len(articles) >= max_pages:
            break
        if member.ns == 0:  # ns=0 indicates articles
            print(f"Fetching: {member.title}")
            articles.append(member.text)
    return articles

def clean_wikipedia_text(text):
    text = re.sub(r"=+.*?=+", "", text)  # Remove headers
    text = re.sub(r"\[\d+\]", "", text)  # Remove references
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    sections = ["See also", "References", "External links", "Further reading", "Katso myös", "Viitteet", "Lähteet"]
    for section in sections:
        text = re.sub(rf"== {section} ==.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text

if __name__ == "__main__":

    # Example: Fetch articles from the "Tekniikka" (Technology) category
    finnish_articles = get_category_members("Historia", max_pages=30)

    # Clean all articles
    corpus = [clean_wikipedia_text(article) for article in finnish_articles]
    print(f"Cleaned {len(corpus)} articles.")

    # Filter short sentences
    corpus = [sentence for sentence in corpus if sentence.count(" ") >= 2]

    # Tokenize corpus into words
    corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

    # Build vocabulary and word-to-index mapping
    flat_corpus = [word for sentence in corpus for word in sentence]
    unique_words = sorted(set(flat_corpus))
    word_to_index = {word: idx for idx, word in enumerate(unique_words)}
    vocab_size = len(unique_words)


    # Generate skip-gram data
    window_size = 2
    X_skip, y_skip = generate_data_skipgram(corpus, window_size, vocab_size, word_to_index)
    print(f"Input shape: {X_skip.shape}, Output shape: {y_skip.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the Word2Vec model
    embedding_dim = 300
    batch_size = 32
    word2vec = Word2Vec(vocab_size, embedding_dim)
    word2vec.to(device)
    optim = AdamW(word2vec.parameters(), lr=3e-3)
    loss_f = nn.CrossEntropyLoss()

    # Convert data to PyTorch tensors
    X_skip = torch.tensor(X_skip, dtype=torch.long, device=device)  # Input indices
    y_skip = torch.tensor(y_skip, dtype=torch.float, device=device)  # One-hot encoded targets

    # Training loop
    idx = np.random.choice(len(X_skip), size=batch_size, replace=True)
    for i in range(40):  # Number of training steps
        # Randomly sample a batch of data
        
        batch_x, batch_y = X_skip[idx], y_skip[idx]

        # Zero gradients from the previous step
        optim.zero_grad()

        # Forward pass
        out = word2vec(batch_x)

        # Compute the loss
        loss = loss_f(out, batch_y)  # Convert one-hot to class indices

        # Backward pass and optimization
        loss.backward()
        optim.step()

        # Print loss every 10 iterations
        # if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.4f}")
