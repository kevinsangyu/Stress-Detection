from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import extractData
import matplotlib.pyplot as plt
import pandas as pd


class Kmeans_single():
    def __init__(self):
        prep = MaxAbsScaler()
        kmeans = KMeans(n_clusters=3, random_state=0)  # 3 clusters: at rest, physical stress, mental stress.

        data = extractData.Extract(r"data/Kevin data/2024_04_22_vs_pegasus")
        data.delta()
        data.homogenise(method="window", size=10)

        data.HR.df.dropna(inplace=True)
        scaled_hr_data = prep.fit_transform(data.HR.df)
        kmeans.fit(scaled_hr_data)

        color_dict = {}

        for cluster in range(kmeans.n_clusters):
            print(f"Cluster {cluster}")
            cluster_data = data.HR.df[kmeans.labels_ == cluster]
            for i in range(cluster_data.shape[0]):
                if cluster == 0:
                    color_dict[cluster_data.index[i]] = 'red'
                elif cluster == 1:
                    color_dict[cluster_data.index[i]] = 'green'
                elif cluster == 2:
                    color_dict[cluster_data.index[i]] = 'blue'

        plt.figure(figsize=(8, 6))
        for i in data.HR.df.index:
            for column in data.HR.df.columns:
                plt.plot(i, data.HR.df[column][i], 'o', color=color_dict.get(i, 'black'))
        plt.show()


class Kmeans_multiple():
    def __init__(self):
        scaler = StandardScaler()
        data = extractData.Extract(r"data/Kevin data/2024_04_22_vs_pegasus")
        data.delta()
        data.homogenise(method="window", size=10)

        df = pd.DataFrame([])
        for i in data.iterable:
            df = pd.concat([df, i.df], axis=1)
        df.dropna(inplace=True)

        scaled_data = scaler.fit_transform(df)
        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(scaled_data)

        color_dict = {}

        for cluster in range(kmeans.n_clusters):
            cluster_data = df[kmeans.labels_ == cluster]
            for i in range(cluster_data.shape[0]):
                if cluster == 0:
                    color_dict[cluster_data.index[i]] = 'red'
                elif cluster == 1:
                    color_dict[cluster_data.index[i]] = 'green'
                elif cluster == 2:
                    color_dict[cluster_data.index[i]] = 'blue'

        print(df.to_string())

        plt.figure(figsize=(8, 6))
        for i in df.index:
            for column in df.columns:
                plt.plot(i, df[column][i], 'o', color=color_dict.get(i, 'black'))
        plt.show()

        # plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=kmeans.labels_)
        # plt.show()


if __name__ == '__main__':
    test = Kmeans_multiple()
