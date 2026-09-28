import torch

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokens = torch.tensor([0, 1, 2], device=device)  # the cat sat
inputs = tokens[:-1]
targets = tokens[1:]

print("inputs:", inputs)
print("targets:", targets)

# A one-token-context language model: embedding followed by token scores.
model = torch.nn.Sequential(
    torch.nn.Embedding(3, 8),
    torch.nn.Linear(8, 3),
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    optimizer.zero_grad()
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    predictions = model(inputs).argmax(dim=1)

print("predictions:", predictions)
print("loss:", loss.item())
