import torch

a, b = torch.nn.Parameter(torch.ones((2, 3))), torch.ones((2, 3))
optimizer = torch.optim.Adam([a])
c = torch.cat([a, b], dim=1)
linear = torch.nn.Linear(6, 1)
loss = (linear(c) - 12).mean()
print(loss)
loss.backward()
print(a.grad, b.grad, c.grad)
optimizer.step()
print(a, b)
print(c)
c = torch.cat([a, b], dim=1)
print(c)

o, i, *_ = 1, 2, 3, 4, 5
print(o, _, i)

print(torch.ones(1, 4, dtype=torch.int64).dtype)
print(torch.LongTensor([1, 2, 3]).dtype)
