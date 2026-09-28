import torch

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"

x = torch.tensor([[0.], [1.], [2.], [3.]], device=device)
print(x.shape)

y = 2 * x + 1
print(y)

model = torch.nn.Linear(1, 1, device=device)
prediction = model(x)
print(prediction)

loss = ((prediction - y) ** 2).mean()
print(loss)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(500):
    optimizer.zero_grad()
    prediction = model(x)
    loss = ((prediction - y) ** 2).mean()
    loss.backward()
    optimizer.step()

print(model(x))
print(((model(x) - y) ** 2).mean())
print("weight:", model.weight.item())
print("bias:", model.bias.item())
