from os import path
import pandas
from math import floor


class Data(object):
    def __init__(self, f_path : str):
        if f_path == r"tags.csv":
            self.df = []
            return
        self.file_path = f_path
        self.df = pandas.read_csv(self.file_path, skiprows=2, header=None)
        with open(self.file_path) as file:
            self.init_time = file.readline().strip().split(",")
            self.sampling = file.readline().strip().split(",")

    def __str__(self):
        return f"Data object with sampling: {self.sampling}, init time: {self.init_time} and df:\n {self.df.describe()}"


class Extract(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.ACC = Data(path.join(self.dir_path, r"ACC.csv"))
        self.BVP = Data(path.join(self.dir_path, r"BVP.csv"))
        self.EDA = Data(path.join(self.dir_path, r"EDA.csv"))
        self.HR = Data(path.join(self.dir_path, r"HR.csv"))
        # self.IBI = Data(path.join(self.dir_path, r"IBI.csv"))
        self.TEMP = Data(path.join(self.dir_path, r"TEMP.csv"))
        self.TAGS = [0 for i in range(0, len(self.HR.df))]
        self.iterable = [self.ACC, self.BVP, self.EDA, self.HR, self.TEMP]  # TAGS and IBI excluded.
        self.get_tags()

    def get_tags(self):
        init_time = floor(float(self.HR.init_time[0]))
        with open(path.join(self.dir_path, r"tags.csv")) as fo:
            tags = fo.readlines()
        for tag in tags:
            tag_time = floor(float(tag.strip()))
            self.TAGS[tag_time-init_time] = 1

    def common_denominator(self):
        small = 500.0
        for i in self.iterable:
            if float(i.sampling[0]) < small:
                small = float(i.sampling[0])
        return small

    def homogenise(self, method, size=10):
        if method == "sampling":
            for i in self.iterable:
                sample = int(float(i.sampling[0]))
                i.df = i.df.iloc[::sample, :]
        elif method == "window":
            for i in self.iterable:
                sample = int(float(i.sampling[0]))
                if i == self.ACC:
                    i.df['MA0'] = i.df.rolling(window=size)[0].mean()
                    i.df['MA1'] = i.df.rolling(window=size)[1].mean()
                    i.df['MA2'] = i.df.rolling(window=size)[2].mean()
                    if '0_delta' in i.df.columns:
                        i.df['MA0_delta'] = i.df.rolling(window=size)['0_delta'].mean()
                        i.df['MA1_delta'] = i.df.rolling(window=size)['1_delta'].mean()
                        i.df['MA2_delta'] = i.df.rolling(window=size)['2_delta'].mean()
                    i.df = i.df.iloc[::sample, :]
                else:
                    i.df['MA'] = i.df.rolling(window=size)[0].mean()
                    if 'delta' in i.df.columns:
                        i.df['MA_delta'] = i.df.rolling(window=size)[0].mean()
                    i.df = i.df.iloc[::sample, :]

    def delta(self):
        for i in self.iterable:
            if i == self.ACC:
                i.df[['0_delta']] = i.df[[0]].pct_change().fillna(0)
                i.df[['1_delta']] = i.df[[1]].pct_change().fillna(0)
                i.df[['2_delta']] = i.df[[2]].pct_change().fillna(0)
            else:
                i.df[['delta']] = i.df[[0]].pct_change().fillna(0)


if __name__ == '__main__':
    e = Extract(r"data/Season4VSBallTorture/Kevin/")
    for i in e.iterable:
        print(i.df.head())
    e.delta()
    print("---------------------Delta'd")
    for i in e.iterable:
        print(i.df.head())
    e.homogenise(method="window", size=10)
    print("-----------------Homogenised")
    for i in e.iterable:
        print(i.df.head())

    # print([i for i, x in enumerate(e.TAGS) if x == 1])
    # print(f"Tags length: {len(e.TAGS)}")
    # print(f"HR length: {len(e.HR.df)}")
