import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

SEQ_LEN = 150

df = pd.read_csv('GOOG.csv')
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
print('Original Data Shape:', df.shape)
print('Original Data Feature:', df.columns)


class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len].values
        y = self.data[index + 1:index + self.seq_len + 1].values
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TransformerPredictor(nn.Module):
    def __init__(self, d_model, nhead, num_layers, seq_len):
        super(TransformerPredictor, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        # The fully connected layer now outputs 'seq_len' values instead of 1
        self.fc = nn.Linear(d_model, seq_len)

    def forward(self, x):
        # Assuming the target is the same as the input, but shifted
        tgt = x

        # Passing both source and target to the transformer
        out = self.transformer(x, tgt)
        print(out)

        # Pass through the fully connected layer
        out = self.fc(out)

        return out


# Generate dataset uses previous class
dataset = StockDataset(df['Close'], seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the model, loss function, and optimizer
model = TransformerPredictor(d_model=SEQ_LEN, nhead=10, num_layers=4, seq_len=SEQ_LEN)  # Adjust parameters as needed
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Store predictions and actual values
all_predictions = []
all_actuals = []

# We'll turn off gradients for validation, saves memory and computations
with torch.no_grad():
    for data, target in dataloader:
        output = model(data)
        all_predictions.append(output)
        all_actuals.append(target)

# Convert lists of tensors to tensors
all_predictions = torch.cat(all_predictions)
all_actuals = torch.cat(all_actuals)

# If your output is (num_samples, sequence_length, num_features) and you are interested in the last prediction of the sequence:
if len(all_predictions.shape) == 3 and all_predictions.shape[1] > 1:
    # Only take the last prediction in the sequence
    predictions = all_predictions[:, -1, :]
    actuals = all_actuals[:, -1, :]
else:
    predictions = all_predictions
    actuals = all_actuals

# Reshape predictions and actuals to be a flat array
predictions = predictions.view(-1, 1).numpy()[SEQ_LEN - 1::SEQ_LEN]
actuals = actuals.view(-1, 1).numpy()[SEQ_LEN - 1::SEQ_LEN]

# Invert the scaling
predictions_inverse_scaled = scaler.inverse_transform(predictions)
actuals_inverse_scaled = scaler.inverse_transform(actuals)

# The predictions are aligned with the actuals, no need for padding
# Plotting
plt.figure(figsize=(15, 5))
plt.plot(actuals_inverse_scaled, label='Actual Prices')
plt.plot(predictions_inverse_scaled, label='Predicted Prices', color='orange')
plt.legend()
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()

