import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

import extractData
import pandas as pd


class RNN:
    def __init__(self):
        self.data = extractData.Extract(r"data/Kevin data/2024_04_29_vs_snortsnort")
        self.data.delta()
        self.data.homogenise(method="window", size=10)
        self.df = self.data.df
        # init fields for later
        self.X_train = self.X_test = self.y_train = self.y_test = None  # dataframe into torch tensors
        self.model = self.criterion = self.optim = None  # neural network model, criterion and optimiser (respectively)

    def drop(self, drop_list=()):
        if drop_list:
            for drop in drop_list:
                for col in self.df.columns:
                    if drop in col:
                        self.df.drop(col, axis=1, inplace=True)

    def select(self, select_list=()):
        if select_list:
            for select in select_list:
                for col in self.df.columns:
                    if select not in col:
                        self.df.drop(col, axis=1, inplace=True)

    def defineModel(self, hidden_size=64, num_layers=1, output_size=1):
        self.model = RNNModule(len(self.df.columns) - 1, hidden_size=hidden_size, num_layers=num_layers,
                               output_size=output_size)
        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def dataprep(self, sequence_length=10):
        scaler = StandardScaler()
        self.df.dropna(inplace=True)

        # separate the target and scale X
        X = self.df.drop('Stress', axis=1)
        y = pd.DataFrame(self.df['Stress'])  # convert to dataframe, since its a series

        X_scaled = scaler.fit_transform(X)

        n_features = len(X.columns)

        # trim dataframe so that there are no remainders after reshaping
        num_sequences = len(X.index) // sequence_length
        X_trimmed = X_scaled[:num_sequences * sequence_length]
        y_trimmed = y[:num_sequences * sequence_length]

        # reshape
        X_rnn = X_trimmed.reshape(-1, sequence_length, n_features)
        y_rnn = y_trimmed[::sequence_length]
        # last label per sequence, i.e. seq length of 10, grabs the 10th value. Similar to "sampling" method from extractData.

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_rnn, y_rnn, test_size=0.2,
                                                                                random_state=42)

        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train.values, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float32)

    def train(self, epochs, batch_size, weighted=False):
        loss_values = []
        if weighted:
            class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=self.y_train.squeeze().numpy())
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            print(f"Class weights: {class_weights}")
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

        for epoch in range(epochs):
            self.model.train()
            permutation = torch.randperm(self.X_train.size(0))

            epoch_loss = 0
            for i in range(0, self.X_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = self.X_train[indices], self.y_train[indices]

                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass and optimization
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                epoch_loss += loss.item()
            loss_values.append(epoch_loss)

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
        return loss_values

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test)
            y_pred_binary = (y_pred > 0.5).float()
        print(y_pred_binary)
        print(classification_report(self.y_test.numpy(), y_pred_binary.numpy()))

    def plot(self, epochs, loss_values):
        plt.plot(range(1, epochs + 1), loss_values, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()


class RNNModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward pass through RNN
        out, _ = self.rnn(x, h0)

        # Take the output of the last time step
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)
        return torch.sigmoid(out)


if __name__ == '__main__':
    epochs = 20
    rnn = RNN()
    rnn.drop(("MA", "delta"))
    rnn.defineModel()
    rnn.dataprep()
    loss = rnn.train(epochs=epochs, batch_size=32)
    rnn.evaluate()
    rnn.plot(epochs, loss)
