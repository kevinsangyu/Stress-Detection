from os import path
import pandas
from math import floor
import matplotlib.pyplot as plt


class Data(object):
    def __init__(self, f_path: str):
        if f_path == r"tags.csv":
            self.df = []
            return
        self.file_path = f_path
        self.df = pandas.read_csv(self.file_path, skiprows=2, header=None)
        if f_path.split(r'/')[-1] != r'ACC.csv' and f_path.split(r'/')[-1] != r'IBI.csv':
            self.df.rename(columns={0: f_path.split(r'/')[-1][:-4]}, inplace=True)
        elif f_path.split(r'/')[-1] == r'ACC.csv':
            self.df.rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=True)
        with open(self.file_path) as file:
            self.init_time = file.readline().strip().split(",")
            self.sampling = file.readline().strip().split(",")

    def __str__(self):
        return f"Data object with sampling: {self.sampling}, init time: {self.init_time} and df:\n {self.df.describe()}"


class Extract(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.ACC = Data(path.join(self.dir_path, r"ACC.csv").replace("\\","/"))
        self.BVP = Data(path.join(self.dir_path, r"BVP.csv").replace("\\","/"))
        self.EDA = Data(path.join(self.dir_path, r"EDA.csv").replace("\\","/"))
        self.HR = Data(path.join(self.dir_path, r"HR.csv").replace("\\","/"))
        # self.IBI = Data(path.join(self.dir_path, r"IBI.csv"))
        self.TEMP = Data(path.join(self.dir_path, r"TEMP.csv").replace("\\","/"))
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
                    i.df['MAx'] = i.df.rolling(window=size)['x'].mean()
                    i.df['MAy'] = i.df.rolling(window=size)['y'].mean()
                    i.df['MAz'] = i.df.rolling(window=size)['z'].mean()
                    if 'x_delta' in i.df.columns:
                        i.df['MAx_delta'] = i.df.rolling(window=size)['x_delta'].mean()
                        i.df['MAy_delta'] = i.df.rolling(window=size)['y_delta'].mean()
                        i.df['MAz_delta'] = i.df.rolling(window=size)['z_delta'].mean()
                    i.df = i.df.iloc[::sample, :]
                else:
                    i.df[f'{i.file_path.split(r"/")[-1][:-4]}-MA'] = i.df.rolling(window=size)[i.df.columns[0]].mean()
                    if f'{i.file_path.split(r"/")[-1][:-4]}-delta' in i.df.columns:
                        i.df[f'{i.file_path.split(r"/")[-1][:-4]}-MA_delta'] = i.df.rolling(window=size)[f'{i.file_path.split(r"/")[-1][:-4]}-delta'].mean()
                    i.df = i.df.iloc[::sample, :]

    def delta(self):
        for i in self.iterable:
            if i == self.ACC:
                i.df[['x_delta']] = i.df[['x']].pct_change().fillna(0)
                i.df[['y_delta']] = i.df[['y']].pct_change().fillna(0)
                i.df[['z_delta']] = i.df[['z']].pct_change().fillna(0)
            else:
                i.df[[f'{i.file_path.split(r"/")[-1][:-4]}-delta']] = i.df[[i.df.columns[0]]].pct_change().fillna(0)


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
        print(i.file_path)
        print(i.df.head(20))


    # print([i for i, x in enumerate(e.TAGS) if x == 1])
    # print(f"Tags length: {len(e.TAGS)}")
    # print(f"HR length: {len(e.HR.df)}")
