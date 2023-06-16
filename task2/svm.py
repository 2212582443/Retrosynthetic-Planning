import numpy as np
import argparse
import torch
import pickle
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="./data")

args = parser.parse_args()

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

input_dim = 2048  # Dimensionality of the one-hot input vector
embedding_dim = 32  # Dimensionality of the learned dense representation
embedding = nn.Embedding(input_dim, embedding_dim)

# Apply the embedding layer to the input features
train_fp = embedding(train_fp)
test_fp = embedding(test_fp)


pca = PCA(n_components=32)
train_fp = pca.fit_transform(train_fp)
test_fp = pca.transform(test_fp)

print("reduce dim complete")

svm = SVR()
svm.fit(train_fp, train_values)

predicted_values = svm.predict(test_fp)

# Calculate and print the mean squared error (MSE)
mse = mean_squared_error(test_values, predicted_values)
r2 = r2_score(test_values, predicted_values)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2: {r2:.4f}")
