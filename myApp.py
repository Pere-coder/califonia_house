import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


X_data = [[-1.6602e-01,  6.7167e-01, -1.9754e-01,  1.6349e-02,  1.6869e-01,
          -2.2855e-02, -6.4084e-01,  6.3246e-01],
         [ 1.7473e-01,  1.8666e+00,  9.7463e-02, -1.9997e-01, -2.8375e-01,
          -5.2193e-02, -6.8754e-01,  7.2757e-01],
         [-2.7559e-01,  1.1404e-01, -2.7428e-01,  5.8955e-02,  2.3051e+00,
           5.4946e-02,  9.3764e-01, -1.2598e+00],
         [-1.2837e-01,  1.8666e+00,  6.1787e-02,  1.2718e-01, -5.1521e-01,
           1.7366e-02,  4.6596e-01, -1.0295e+00],
         [ 5.9709e-01,  4.3269e-01,  3.2285e-01, -1.8101e-01, -5.7111e-01,
          -3.1126e-02, -1.3133e+00,  1.3233e+00],
         [ 6.5757e-01, -5.2324e-01,  2.2897e-01, -1.0475e-01, -3.1432e-01,
          -5.2151e-02, -5.6145e-01, -1.0340e-01],
         [-4.7992e-01,  1.4683e+00, -5.8370e-02, -9.1494e-02, -4.9512e-01,
          -1.1912e-02,  1.0077e+00, -1.3248e+00],
         [ 2.0101e-01, -1.0012e+00, -1.9899e-01, -7.4950e-02,  1.4684e+00,
          -4.6905e-02, -8.2297e-01,  8.8275e-01],
         [-7.4526e-01, -2.8426e-01, -6.9457e-01, -1.6292e-01, -6.6457e-01,
           4.5899e-02, -1.3507e+00,  1.2282e+00],
         [-6.0486e-01, -9.2155e-01,  7.5352e-02,  2.8884e-02,  5.3569e-03,
          -4.7675e-02,  1.4560e+00, -5.3390e-01],
         [ 3.5132e-01, -2.1165e+00,  8.5590e-01,  2.5502e-01,  6.3116e+00,
          -5.3526e-03,  1.4420e+00, -8.9433e-01],
         [-1.0000e+00, -2.0460e-01,  1.1017e+00,  1.5343e+00, -8.9079e-01,
          -7.6484e-02,  1.5447e+00, -2.0352e-01],
         [-8.5202e-02, -1.3995e+00,  1.1111e-01, -1.8701e-01,  7.6350e-01,
          -6.5608e-02, -1.1452e+00,  1.2131e+00],
         [-8.5472e-01, -1.5588e+00, -5.1602e-01, -2.0886e-02, -6.5322e-01,
          -1.2680e-01,  5.1266e-01, -8.3377e-02],
         [ 2.2226e+00, -1.1605e+00,  1.3133e+00, -6.8615e-02, -1.1605e-01,
           7.4462e-02, -9.3038e-01,  9.2781e-01],
         [-5.3499e-01, -3.6392e-01, -8.8574e-01, -1.6688e-01, -8.6197e-01,
           2.9309e-01, -8.6500e-01,  6.8753e-01],
         [ 6.0632e-01, -1.2494e-01,  3.8896e-01, -1.8077e-01, -2.1824e-01,
          -2.2720e-02, -5.5678e-01, -1.1842e-01],
         [-5.2125e-01,  1.2293e+00,  1.4447e-01,  1.4066e-01, -5.3006e-01,
          -7.1639e-02,  3.2586e-01,  1.3187e-01],
         [ 3.5423e+00,  5.9201e-01,  1.1581e+00, -1.8696e-01, -6.7505e-01,
          -3.3765e-02,  1.0777e+00, -1.5000e+00],
         [ 2.6310e-01, -2.8426e-01,  4.4324e-01, -9.2659e-02,  4.1849e-01,
          -4.1930e-03, -1.3460e+00,  1.2932e+00]],
X = torch.tensor(X_data)
y_data = [[1.8450],
         [3.2030],
         [1.8210],
         [1.6550],
         [1.6530],
         [3.4250],
         [1.3880],
         [2.0610],
         [0.9030],
         [1.1750],
         [1.5130],
         [0.8460],
         [1.9980],
         [0.5420],
         [3.0890],
         [1.3750],
         [2.8020],
         [0.7700],
         [5.0000],
         [2.1400]]
y = torch.tensor(y_data)


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
input_dim = 8   
model = RegressionModel(input_dim)
model.load_state_dict(torch.load('regression_model.pth'))
model.eval()

model.eval()
with torch.no_grad():
  predictions = model(X)


y_original = y.detach().numpy().flatten()
pred = predictions.detach().numpy().flatten()
y_pos = range(len(y_original))

plt.scatter(y_pos, y_original, c='b', label='Expected values')
plt.scatter(y_pos, pred, c='r', label='Predicted values')

plt.plot(y_pos, y_original, 'b-', alpha=0.3)
plt.plot(y_pos, pred, 'r-', alpha=0.3)

plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Real vs Predicted Values')
plt.legend()



st.write("""
# PYTORCH MODEL TRAINED TO PREDICT HOUSE PRICES.
This is a **real vs predicted** scatter plot for a simple linear regression model using PYTORCH.The model was trained using carlifornia housing dataset.
""")

st.pyplot(plt)