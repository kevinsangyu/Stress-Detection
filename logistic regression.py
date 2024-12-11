import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import extractData


class LRegression:
    def __init__(self):
        self.data = extractData.Extract(r"data/Kevin data/2024_04_29_vs_snortsnort")
        self.data.delta()
        self.data.homogenise(method="window", size=10)
        # combine all sensors into 1 dataframe
        # todo move this block of code into extractData and make it return the dataframe...
        self.df = pd.DataFrame([])
        for i in self.data.iterable:
            self.df = pd.concat([self.df.reset_index(drop=True), i.df.reset_index(drop=True)], axis=1)
        stress_df = pd.DataFrame(self.data.STRESS).rename(columns={0: 'Stress'})
        self.df = pd.concat([self.df.reset_index(drop=True), stress_df.reset_index(drop=True)], axis=1)

        self.X_train = self.X_test = self.y_train = self.y_test = pd.DataFrame([])

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

    def scalesplit(self):
        scaler = MaxAbsScaler()
        self.df.dropna(inplace=True)  # dropping null values after selecting/dropping Moving Averages or Deltas

        # separate the target and scale X
        X = self.df.drop('Stress', axis=1)
        y = self.df['Stress']

        X_scaled = scaler.fit_transform(X)

        # split data, first 2 sets training, last set testing. i.e. 66:33 split.
        train_size = int(0.6 * len(X_scaled))
        print(f"X length: {len(X_scaled)}, y length: {len(y)}")
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

    def predict(self):
        # Class imbalance issue. 97% of the data is under no stress, and the model achieves a high accuracy by just
        # not predicting. So we need to balance the classes out. The below just balances the weights, making
        # misclassified stressed parts heavily penalised.
        # this drops the accuracy from 0.97 to 0.69
        model = LogisticRegression(class_weight='balanced')
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        if y_pred.max() == 0.0:
            print("Maximum is 0, no prediction was made.")
            return
        y_predlist = y_pred.tolist()
        print(f'{len([i for i, x in enumerate(y_predlist) if x == 1])} stressed out of {len(y_predlist)}')
        print(f"Timestamps: {[i+len(self.y_train) for i, x in enumerate(y_predlist) if x == 1]}")
        differences = []
        for i in range(0, len(y_predlist)-1):
            differences.append(y_predlist[i+1]-y_predlist[i])
        print(f"Differences: {differences}")

        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        print(f'Accuracy: {accuracy:.2f}')
        print('Classification Report:')
        print(report)


if __name__ == '__main__':
    lr = LRegression()
    lr.drop(("MA", "delta"))
    lr.scalesplit()
    lr.predict()
