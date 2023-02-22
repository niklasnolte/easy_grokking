# %%
import torch
import tqdm
from torch.nn import functional as F
torch.manual_seed(1)
# %%
P = 53
TRAIN_FRAC = .8
EPOCHS = 1000
HIDDEN_DIM = 128
WEIGHT_DECAY = 1e-3
LR = 1e-3

class Model(torch.nn.Module):
  def __init__(self, hidden_dim, P):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.P = P
    self.emb = torch.nn.Embedding(P, hidden_dim)
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear(2*hidden_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, P),
    )

  def forward(self, x):
    # x = [num1, num2]
    emb = self.emb(x).view(-1, 2*self.hidden_dim)
    return self.decoder(emb)

# %%

# data is X = [num1, num2], Y = num1 + num2 % P
all_nums = torch.arange(P)
X = torch.cartesian_prod(all_nums, all_nums)
Y = X.sum(axis=1) % P
# Y mean 0 std 1
# %%
# train and tests with randperm
perm = torch.randperm(X.shape[0])
train_size = int(X.shape[0] * TRAIN_FRAC)
train_idx = perm[:train_size]
test_idx = perm[train_size:]
X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

# %%

model = Model(HIDDEN_DIM, P)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
bar = tqdm.tqdm(range(EPOCHS))
for epoch in bar:
  optimizer.zero_grad()
  Yhat = model(X_train)
  loss = F.cross_entropy(Yhat, Y_train)
  loss.backward()
  optimizer.step()
  acc = Yhat.argmax(axis=1).eq(Y_train).float().mean()
  with torch.no_grad():
    Yhat_test = model(X_test)
    test_loss = F.cross_entropy(Yhat_test, Y_test)
    test_acc = Yhat_test.argmax(axis=1).eq(Y_test).float().mean()
  bar.set_description(f"loss: {loss.item():.3e}, test_loss: {test_loss.item():.3e}, acc: {acc.item():.3f}, test_acc: {test_acc.item():.3f}")
# %%
