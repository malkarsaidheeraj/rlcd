import torch

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = torch.nn.Sequential(
    torch.nn.Linear(1, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1),
).to(device)

x = torch.tensor([[-2.], [-1.], [0.], [1.], [2.]], device=device)

print(model)
print("after first layer:", model[0](x).shape)
print("final output:", model(x).shape)

target = x.abs()
print(target)

optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

for step in range(1000):
    optimizer.zero_grad()
    prediction = model(x)
    loss = ((prediction - target) ** 2).mean()
    loss.backward()
    optimizer.step()

print("trained predictions:", model(x))
print("final loss:", ((model(x) - target) ** 2).mean())

val_x = torch.tensor([[-1.5], [-0.5], [0.5], [1.5]], device=device)
val_y = val_x.abs()

with torch.no_grad():
    val_pred = model(val_x)
    val_loss = ((val_pred - val_y) ** 2).mean()

print("validation predictions:", val_pred)
print("validation loss:", val_loss)
