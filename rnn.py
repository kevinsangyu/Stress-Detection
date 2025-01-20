import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

import extractData
import pandas as pd


class RNN:
    def __init__(self):
        self.data = extractData.Extract(r"data/Kevin data/2024_04_29_vs_snortsnort")
        self.data.delta()
        self.data.homogenise(method="window", size=10)
        self.data.combine_df()
        self.df = self.data.df
        # init fields for later
        self.X_train = self.X_test = self.y_train = self.y_test = None  # dataframe into torch tensors
        self.model = self.criterion = self.optim = None  # neural network model, criterion and optimiser (respectively)

        self.test_data = extractData.Extract(r"data/Kevin data/2024_06_24_finals")
        self.test_data.delta()
        self.test_data.homogenise(method="window", size=10)
        self.test_data.combine_df()
        self.testdf = self.test_data.df

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def drop(self, drop_list=()):
        if drop_list:
            for drop in drop_list:
                for col in self.testdf.columns:
                    if drop in col:
                        self.testdf.drop(col, axis=1, inplace=True)
                for col in self.df.columns:
                    if drop in col:
                        self.df.drop(col, axis=1, inplace=True)

    def select(self, select_list=()):
        if select_list:
            for select in select_list:
                for col in self.testdf.columns:
                    if select not in col and col != "Stress":
                        self.testdf.drop(col, axis=1, inplace=True)
                for col in self.df.columns:
                    if select not in col and col != "Stress":
                        self.df.drop(col, axis=1, inplace=True)

    def defineModel(self, hidden_size=64, num_layers=1, output_size=1):
        self.model = RNNModule(len(self.df.columns) - 1, hidden_size=hidden_size, num_layers=num_layers,
                               output_size=output_size)
        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def dataprep(self, sequence_length=10):
        scaler = StandardScaler()
        testscaler = StandardScaler()
        self.df.dropna(inplace=True)
        self.testdf.dropna(inplace=True)

        # separate the target and scale X
        X = self.df.drop('Stress', axis=1)
        y = pd.DataFrame(self.df['Stress'])  # convert to dataframe, since its a series
        Xtest = self.testdf.drop('Stress', axis=1)
        ytest = pd.DataFrame(self.testdf['Stress'])

        X_scaled = scaler.fit_transform(X)
        Xtest_scaled = testscaler.fit_transform(Xtest)

        n_features = len(X.columns)
        test_n_features = len(Xtest.columns)

        # trim dataframe so that there are no remainders after reshaping
        num_sequences = len(X.index) // sequence_length
        X_trimmed = X_scaled[:num_sequences * sequence_length]
        y_trimmed = y[:num_sequences * sequence_length]
        test_num_sequences = len(Xtest.index) // sequence_length
        Xtest_trimmed = Xtest_scaled[:test_num_sequences * sequence_length]
        ytest_trimmed = ytest[:test_num_sequences * sequence_length]

        # reshape
        X_rnn = X_trimmed.reshape(-1, sequence_length, n_features)
        y_rnn = y_trimmed[::sequence_length]
        Xtest_rnn = Xtest_trimmed.reshape(-1, sequence_length, test_n_features)
        ytest_rnn = ytest_trimmed[::sequence_length]
        # last label per sequence, i.e. seq length of 10, grabs the 10th value. Similar to "sampling" method from extractData.

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_rnn, y_rnn, test_size=0.2,
        #                                                                         random_state=42)
        self.X_train = X_rnn
        self.y_train = y_rnn
        self.X_test = Xtest_rnn
        self.y_test = ytest_rnn

        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train.values, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float32)

    def train(self, epochs, batch_size, weighted=False, custom_weight=()):
        loss_values = []
        if weighted:
            # class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=self.y_train.squeeze().numpy())
            class_weights = torch.tensor(custom_weight, dtype=torch.float32)
            print(f"Class weights: {class_weights}")
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
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

            # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
        return loss_values

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test)
            y_pred_binary = (y_pred > 0.5).float()
        print(classification_report(self.y_test.numpy(), y_pred_binary.numpy()))
        return y_pred_binary

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


def calc_precision(epochs=20, plot=False, weight=40):
    results = {}

    all = RNN()
    all.defineModel()
    all.dataprep()
    all.train(epochs=epochs, batch_size=32, weighted=True, custom_weight=(1, weight))
    all_pred = all.evaluate()
    results["all"] = precision_score(all.y_test, all_pred)

    raw = RNN()
    raw.drop(("MA", "delta"))
    raw.defineModel()
    raw.dataprep()
    raw.train(epochs=epochs, batch_size=32, weighted=True, custom_weight=(1, weight))
    raw_pred = raw.evaluate()
    results["raw"] = precision_score(raw.y_test, raw_pred)

    delta = RNN()
    delta.select('delta')
    delta.defineModel()
    delta.dataprep()
    delta.train(epochs=epochs, batch_size=32, weighted=True, custom_weight=(1, weight))
    delta_pred = delta.evaluate()
    results["delta"] = precision_score(delta.y_test, delta_pred)

    MA = RNN()
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
        results = calc_precision(plot=False, weight=40, epochs=epoch)
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
        results = calc_precision(plot=False, weight=weight)
        average_precision = sum(results.values()) / len(results.values())
        indexes.append(weight)
        weights.append(average_precision)
    plt.title("Optimal class weight for precision")
    plt.xlabel("Class weights")
    plt.ylabel("Precision Score")
    plt.plot(indexes, weights)
    plt.show()


if __name__ == '__main__':
    calc_precision(plot=True, weight=40, epochs=10)
    # custom_weights(1000)
    # calc_epochs(100)
