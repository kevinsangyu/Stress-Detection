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
        self.df = pd.DataFrame([])
        for i in self.data.iterable:
            self.df = pd.concat([self.df.reset_index(drop=True), i.df.reset_index(drop=True)], axis=1)
        self.df.dropna(inplace=True)
        self.X_train = self.X_test = self.y_train = self.y_test = pd.DataFrame([])

    def drop(self, drop_list=()):
        if drop_list:
            for drop in drop_list:
                for col in self.df.columns:
                    if drop in col:
                        self.df.drop(col, axis=1, inplace=True)
        print(self.df.to_string())

    def select(self, select_list=()):
        if select_list:
            for select in select_list:
                for col in self.df.columns:
                    if select not in col:
                        self.df.drop(col, axis=1, inplace=True)
        print(self.df.to_string())

    def scalesplit(self):
        scaler = MaxAbsScaler()
        X = self.df
        y = pd.DataFrame(self.data.STRESS).rename(columns={0: 'Stress'})

        X_scaled = scaler.fit_transform(X)

        # split data, first 2 sets training, last set testing.
        train_size = int(0.6 * len(X_scaled))
        print(f"X length: {len(X_scaled)}, y length: {len(y)}")
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

    def predict(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        print(f'Accuracy: {accuracy:.2f}')
        print('Classification Report:')
        print(report)


if __name__ == '__main__':
    lr = LRegression()
    # lr.drop(("MA", "delta"))
    # lr.scalesplit()
    # lr.predict()
