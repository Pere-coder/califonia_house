import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X, y = data.data, data.target
print(f'values{data.data[:1]}, label{data.target[:1]}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


class RegressionModel(nn.Module):
  def __init__(self, input_dim):
    super(RegressionModel, self).__init__()
    self.fc = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
  def forward(self, x):
    return self.fc(x)

model = RegressionModel(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training loop
for epoch in range(2000):
  model.train()
  optimizer.zero_grad()
  predictions = model(X_train_tensor)
  loss = criterion(predictions, y_train_tensor)
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 50 == 0:
    print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
  test_predictions = model(X_test_tensor)
  test_loss = criterion(test_predictions, y_test_tensor)
  print(f'Test Loss: {test_loss.item():.4f}')




torch.save(model.state_dict(), 'regression_model.pth')
print("Model saved as 'regression_model.pth'")