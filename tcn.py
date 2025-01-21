import pandas as pd
from sklearn.metrics import classification_report, precision_score
from sklearn.utils.class_weight import compute_class_weight
import extractData
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


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

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        # self.criterion = torch.nn.BCELoss()  defined later, because weights.
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
        y_tcn = y_trimmed.values[::seq_length]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_tcn, y_tcn, test_size=0.2,
                                                                                random_state=42)
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32).permute(0, 2, 1)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32).permute(0, 2, 1)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

    def train(self, epochs, batch_size, weighted=False, custom_weight=()):
        self.model.train()
        if weighted:
            if custom_weight:
                class_weights = custom_weight
            else:
                class_weights = compute_class_weight('balanced', classes=np.array([0.0, 1.0]), y=self.y_train.numpy().flatten())
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            print(f"Class weights: {class_weights}")
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights[1] / class_weights[0]))
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
        return predictions


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
        out = self.network(x)
        # Check if residual connection needs adjustment
        if x.size(1) != out.size(1) or x.size(2) != out.size(2):
            x = torch.nn.functional.pad(x, (0, out.size(2) - x.size(2)))
        out = self.fc(out.mean(dim=2))
        return out


def calc_precision(epochs=20, plot=False, weight=20):
    results = {}

    all = TCN()
    all.defineModel()
    all.dataprep()
    all.train(epochs=epochs, batch_size=32, weighted=True, custom_weight=(1, weight))
    all_pred = all.evaluate()
    results["all"] = precision_score(all.y_test, all_pred)

    raw = TCN()
    raw.drop(("MA", "delta"))
    raw.defineModel()
    raw.dataprep()
    raw.train(epochs=epochs, batch_size=32, weighted=True, custom_weight=(1, weight))
    raw_pred = raw.evaluate()
    results["raw"] = precision_score(raw.y_test, raw_pred)

    delta = TCN()
    delta.select('delta')
    delta.defineModel()
    delta.dataprep()
    delta.train(epochs=epochs, batch_size=32, weighted=True, custom_weight=(1, weight))
    delta_pred = delta.evaluate()
    results["delta"] = precision_score(delta.y_test, delta_pred)

    MA = TCN()
    MA.select('MA')
    MA.defineModel()
    MA.dataprep()
    MA.train(epochs=epochs, batch_size=32, weighted=True, custom_weight=(1, weight))
    MA_pred = MA.evaluate()
    results["MA"] = precision_score(MA.y_test, MA_pred)

    if plot:
        plt.bar(range(len(results)), list(results.values()), align='center')
        plt.xticks(range(len(results)), list(results.keys()))
        plt.title("Precision scores of different data selections")
        plt.ylabel("Precision Score")
        plt.xlabel(f"Data selections, class_weight = {weight}, epochs = {epochs} for weighted selections")
        plt.show()
    return results


def calc_epochs(limit):
    indexes = []
    epochs = []
    for epoch in range(10, limit, 10):
        results = calc_precision(plot=False, weight=50, epochs=epoch)
        average_precision = sum(results.values()) / len(results.values())
        indexes.append(epoch)
        epochs.append(average_precision)
    plt.title("Optimal epoch for precision")
    plt.xlabel("Epochs")
    plt.ylabel("Precision Score")
    plt.plot(indexes, epochs)
    plt.show()


def custom_weights(limit):
    indexes = []
    weights = []
    for weight in range(10, limit, 10):
        results = calc_precision(plot=False, weight=weight, epochs=20)
        average_precision = sum(results.values()) / len(results.values())
        indexes.append(weight)
        weights.append(average_precision)
    plt.title("Optimal class weight for precision")
    plt.xlabel("Class weights")
    plt.ylabel("Precision Score")
    plt.plot(indexes, weights)
    plt.show()


if __name__ == '__main__':
    calc_precision(epochs=30, plot=True, weight=50)
    # custom_weights(100)
    # calc_epochs(50)
