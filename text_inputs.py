import torch

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"

sentence = "the cat sat"
vocab = {"the": 0, "cat": 1, "sat": 2}

token_ids = [vocab[word] for word in sentence.split()]
print(token_ids)

ids = torch.tensor(token_ids, device=device)
embedding = torch.nn.Embedding(len(vocab), 4, device=device)
vectors = embedding(ids)

print(vectors.shape)

scores = vectors @ vectors.T
print(scores.shape)

weights = torch.softmax(scores, dim=1)
print(weights)
print(weights.sum(dim=1))

context = weights @ vectors
print(context.shape)
print(context[0])

future = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
masked_scores = scores.masked_fill(future, float("-inf"))
causal_weights = torch.softmax(masked_scores, dim=1)
print(causal_weights)
