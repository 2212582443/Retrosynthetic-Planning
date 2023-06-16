import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
import argparse
import matplotlib.pyplot as plt

torch.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="./data")
parser.add_argument("--save_dir", type=str, default="./model")
parser.add_argument("--save_freq", type=int, default=10)

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

with open(f"{args.data_root}/train.pkl", "rb") as file:
    train_set = pickle.load(file)

with open(f"{args.data_root}/test.pkl", "rb") as file:
    test_set = pickle.load(file)

train_packed_fp = train_set["packed_fp"]
train_fp = np.unpackbits(train_packed_fp, axis=1)
train_fp = torch.tensor(train_fp, dtype=int)
train_values = train_set["values"].squeeze()


test_packed_fp = test_set["packed_fp"]
test_fp = np.unpackbits(test_packed_fp, axis=1)
test_fp = torch.tensor(test_fp, dtype=int)
test_values = test_set["values"].squeeze()


class CustomDataset(Dataset):
    def __init__(self, fp, values):
        self.fp = fp
        self.values = values

    def __len__(self):
        return len(self.fp)

    def __getitem__(self, idx):
        return self.fp[idx], self.values[idx]


# Define the network architecture
class CostPredictionNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(CostPredictionNetwork, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.mean(dim=1)  # Average the embeddings
        out = F.leaky_relu(self.fc1(embedded))
        out = F.relu(self.fc2(out))
        return out.squeeze()


# Define the hyperparameters
input_dim = 2048
embedding_dim = 128
hidden_dim = 64
learning_rate = 0.001
batch_size = 64
num_epochs = 100

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the network and move it to the device
net = CostPredictionNetwork(input_dim, embedding_dim, hidden_dim).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
# optimizer = optim.Adam(net.parameters(), lr=learning_rate)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

train_dataset = CustomDataset(train_fp, train_values)
test_dataset = CustomDataset(test_fp, test_values)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

train_losses = []
test_losses = []
train_r2_scores = []
test_r2_scores = []

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = net(batch_data)

        # Compute loss
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Collect true and predicted values for R-squared calculation
        y_true.extend(batch_labels.tolist())
        y_pred.extend(outputs.tolist())

    # Calculate R-squared for the training set
    train_r2 = r2_score(y_true, y_pred)
    train_r2_scores.append(train_r2)
    train_losses.append(running_loss / len(train_loader))
    # Print training loss and R-squared for this epoch
    print(
        f"Epoch {epoch + 1} - Training Loss: {running_loss / len(train_loader)}, R^2: {train_r2:.4f}"
    )

    if (epoch + 1) % args.save_freq == 0:
        torch.save(net.state_dict(), f"{args.save_dir}/model_epoch_{epoch + 1}.pth")

    # Evaluate on the test set
    net.eval()
    with torch.no_grad():
        test_loss = 0.0
        y_true = []
        y_pred = []

        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = net(batch_data)
            test_loss += criterion(outputs, batch_labels).item()

            # Collect true and predicted values for R-squared calculation
            y_true.extend(batch_labels.tolist())
            y_pred.extend(outputs.tolist())

        # Calculate R-squared for the test set
        test_r2 = r2_score(y_true, y_pred)
        test_r2_scores.append(test_r2)
        test_losses.append(test_loss / len(test_loader))

        # Print test loss and R-squared for this epoch
        print(
            f"Epoch {epoch + 1} - Test Loss: {test_loss / len(test_loader)}, R^2: {test_r2:.4f}"
        )

plt.figure(figsize=(10, 5))

# MSE curve
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Train")
plt.plot(range(1, num_epochs + 1), test_losses, label="Test")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.title("Mean Squared Error")

# R-squared curve
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_r2_scores, label="Train")
plt.plot(range(1, num_epochs + 1), test_r2_scores, label="Test")
plt.xlabel("Epoch")
plt.ylabel("R^2")
plt.legend()
plt.title("R-squared")

# Save the figure
plt.tight_layout()
plt.savefig("curves.png")
