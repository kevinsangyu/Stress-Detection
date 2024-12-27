import pandas as pd
from sklearn.metrics import classification_report

import extractData
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TCN:
    def __init__(self):
        self.data = extractData.Extract(r"data/Kevin data/2024_04_29_vs_snortsnort")
        self.data.delta()
        self.data.homogenise(method="window", size=10)
        self.data.combine_df()
        self.df = self.data.df
        # init fields for later
        self.X_train = self.X_test = self.y_train = self.y_test = None  # dataframe into torch tensors
        self.model = self.criterion = self.optim = None  # neural network model, criterion and optimiser (respectively)

    def drop(self, drop_list=()):
        if drop_list:
            for drop in drop_list:
                for col in self.df.columns:
                    if drop in col and col != "Stress":
                        self.df.drop(col, axis=1, inplace=True)

    def select(self, select_list=()):
        if select_list:
            for select in select_list:
                for col in self.df.columns:
                    if select not in col and col != "Stress":
                        self.df.drop(col, axis=1, inplace=True)

    def defineModel(self):
        self.model = TCNModule(input_size=len(self.df.columns)-1, num_classes=1, num_channels=[32, 64], kernel_size=3,
                               dropout=0.2)
        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def dataprep(self, seq_length=10):
        self.df.dropna(inplace=True)
        scaler = StandardScaler()

        X = self.df.drop("Stress", axis=1)
        y = pd.DataFrame(self.df["Stress"])

        X_scaled = scaler.fit_transform(X)

        # trim dataframe so that there are no remainders after reshaping
        num_features = len(X.columns)
        num_sequences = len(X.index) // seq_length
        X_trimmed = X_scaled[:num_sequences * seq_length]
        y_trimmed = y[:num_sequences * seq_length]

        X_tcn = X_trimmed.reshape(-1, seq_length, num_features)
        y_tcn = y_trimmed.values[:len(X_tcn)]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_tcn, y_tcn, test_size=0.2,
                                                                                random_state=42)
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32).permute(0, 2, 1)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32).permute(0, 2, 1)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

    def train(self, epochs, batch_size):
        self.model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(self.X_train.size(0))
            epoch_loss = 0

            for i in range(0, self.X_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = self.X_train[indices], self.y_train[indices]

                self.optim.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optim.step()

                epoch_loss += loss.item()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            predictions = (outputs > 0.5).float()
        # print(predictions)
        print(classification_report(self.y_test.numpy(), predictions.numpy()))


class TCNBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_size, output_size, kernel_size, padding=(kernel_size - 1) * dilation,
                                     dilation=dilation)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(output_size, output_size, kernel_size, padding=(kernel_size - 1) * dilation,
                                     dilation=dilation)
        self.residual = torch.nn.Conv1d(input_size, output_size, kernel_size=1) if input_size != output_size else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)

        if self.residual is not None:
            x = self.residual(x)

        if x.size(2) != out.size(2):
            if x.size(2) < out.size(2):
                pad_size = out.size(2) - x.size(2)
                x = torch.nn.functional.pad(x, (0, pad_size))
            else:
                x = x[:, :, :out.size(2)]

        return x + out  # Residual connection


class TCNModule(torch.nn.Module):
    def __init__(self, input_size, num_classes, num_channels, kernel_size=3, dropout=0.2):
        super(TCNModule, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        self.network = torch.nn.Sequential(*layers)
        self.fc = torch.nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # TCN expects input in shape [batch_size, input_size, sequence_length]
        x = self.network(x)
        x = x[:, :, -1]  # Use the last time step's output
        return torch.sigmoid(self.fc(x))


if __name__ == '__main__':
    tcn = TCN()
    tcn.drop(("MA", "delta"))
    tcn.defineModel()
    tcn.dataprep()
    tcn.train(epochs=20, batch_size=32)
    tcn.evaluate()
