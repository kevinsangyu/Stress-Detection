import extractData
import pandas as pd
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np


class LSTM():
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

    def defineModel(self, hidden_dim=50, output_dim=1):
        # input_dim = number of features - target feature
        self.model = LSTMmodule(len(self.df.columns) - 1, hidden_dim, output_dim)
        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def dataprep(self, sequence_length=10):
        scaler = MinMaxScaler()
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
        X_lstm = X_trimmed.reshape(-1, sequence_length, n_features)
        y_lstm = y_trimmed[::sequence_length]
        # last label per sequence, i.e. seq length of 10, grabs the 10th value. Similar to "sampling" method from extractData.

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train.values, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float32)

    def train(self, epochs=20, batch=32, weighted=False):
        train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
        loss_values = []
        # returns each epoch's loss values for plotting purposes
        if weighted:
            class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=self.y_train.squeeze().numpy())
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            print(f"Class weights: {class_weights}")
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optim.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optim.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
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


class LSTMmodule(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMmodule, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)
    

if __name__ == '__main__':
    epochs = 20
    lstm = LSTM()
    lstm.drop(("MA", "delta"))
    lstm.defineModel()
    lstm.dataprep()
    loss = lstm.train(weighted=True)
    lstm.evaluate()
    lstm.plot(epochs, loss)

    lstm = LSTM()
    lstm.drop(("MA", "delta"))
    lstm.defineModel()
    lstm.dataprep()
    loss = lstm.train()
    lstm.evaluate()
    lstm.plot(epochs, loss)
