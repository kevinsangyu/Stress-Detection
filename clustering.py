from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import extractData
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.tree import _tree, DecisionTreeClassifier, plot_tree
import numpy as np


class Kmeans_single():
    def __init__(self):
        prep = MaxAbsScaler()

        data = extractData.Extract(r"data/Kevin data/2024_04_29_vs_snortsnort")
        data.delta()
        data.homogenise(method="window", size=10)

        for column in data.HR.df.columns:
            print(column)
            df = pd.DataFrame(data.HR.df[column])
            df.dropna(inplace=True)
            scaled_hr_data = prep.fit_transform(df)
            kmeans = KMeans(n_clusters=3, random_state=0)  # 3 clusters: at rest, physical stress, mental stress.
            kmeans.fit(scaled_hr_data)
            df['Cluster'] = kmeans.labels_
            print(df.head(20))

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

            plt.figure(figsize=(8, 6))
            plt.title(column)
            for i in df.index:
                for column in df.columns:
                    if column == 'Cluster':
                        continue
                    plt.plot(i, df[column][i], 'o', color=color_dict.get(i, 'black'))
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster 0'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Cluster 1'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster 2')]

            plt.legend(handles=legend_elements)

            X = df.drop(columns='Cluster')
            y = df['Cluster']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            clf = DecisionTreeClassifier(random_state=0)
            clf.fit(X_train, y_train)

            plt.figure(figsize=(12, 8))
            plot_tree(clf, feature_names=X.columns, class_names=['Cluster 0', 'Cluster 1', 'Cluster 2'], filled=True)
            plt.show()

            feature_importances = clf.feature_importances_
            important_features = sorted(zip(X.columns, feature_importances), key=lambda x: x[1], reverse=True)

            for feature, importance in important_features:
                print(f'{feature}: {importance}')

        plt.show()


class Kmeans_multiple():
    def __init__(self, feature="", excludelist=()):
        scaler = StandardScaler()
        self.data = extractData.Extract(r"data/Kevin data/2024_04_29_vs_snortsnort")
        self.data.delta()
        self.data.homogenise(method="window", size=10)
        self.data.combine_df()
        self.df = self.data.df

        if not feature:
            # use all features if no specific one is provided, removing the excluded features.
            cols = [col for col in self.df.columns]
            for col in cols.copy():
                for exclude in excludelist:
                    if exclude in col:
                        cols.remove(col)
                        break
            df = self.df[cols]
            df.drop("Stress", axis=1, inplace=True)
        else:
            # grab the MA, delta and raw features of given feature
            # also works in grabbing just MA values or just delta values if you use 'MA' or 'delta'
            cols = [col for col in self.df.columns if feature in col]
            for col in cols.copy():
                for exclude in excludelist:
                    if exclude in col:
                        cols.remove(col)
                        break
            df = self.df[cols]
        print(f"Selected columns: {cols}")
        df = df.dropna()

        scaled_data = scaler.fit_transform(df)
        algorithm = KMeans(n_clusters=3, random_state=0)
        algorithm.fit(scaled_data)
        df['Cluster'] = algorithm.labels_

        color_dict = {}

        for cluster in range(algorithm.n_clusters):
            cluster_data = df[algorithm.labels_ == cluster]
            for i in range(cluster_data.shape[0]):
                if cluster == 0:
                    color_dict[cluster_data.index[i]] = 'red'
                elif cluster == 1:
                    color_dict[cluster_data.index[i]] = 'green'
                elif cluster == 2:
                    color_dict[cluster_data.index[i]] = 'blue'

        plt.figure(figsize=(8, 6))
        for i in df.index:
            # for column in df.columns:
            plt.plot(i, df['HR'][i], 'o', color=color_dict.get(i, 'black'))
        plt.show()

        X = df.drop(columns='Cluster')
        y = df['Cluster']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)

        plt.figure(figsize=(12, 8))
        plot_tree(clf, feature_names=X.columns, class_names=['Cluster 0', 'Cluster 1', 'Cluster 2'], filled=True)
        plt.show()


class Dbscan():
    def __init__(self, drop_list=(), select_list=()):
        data = extractData.Extract(r"data/Kevin data/2024_04_29_vs_snortsnort")
        data.delta()
        data.homogenise(method="window", size=10)

        df = pd.DataFrame([])
        for i in data.iterable:
            df = pd.concat([df, i.df], axis=1)
        print(df.to_string())
        df.dropna(inplace=True)
        if drop_list:
            for drop in drop_list:
                for col in df.columns:
                    if drop in col:
                        df.drop(col, axis=1, inplace=True)
        if select_list:
            for select in select_list:
                for col in df.columns:
                    if select not in col:
                        df.drop(col, axis=1, inplace=True)

        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(df)

        self.nearest(reduced_data)
        epsilon = int(input("Enter elbow value: "))
        # epsilon = 30

        scan = DBSCAN(eps=epsilon, min_samples=2)
        labels = scan.fit_predict(reduced_data)
        print(f"{len(set(labels))} Clusters")

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(df, labels)

        cluster_colors = ['red', 'green', 'blue', 'purple', 'brown', 'black']  # Replace with your cluster colors
        noise_color = 'grey'  # Color for noise points

        # Plot each time series, coloring the segments by the cluster
        plt.figure(figsize=(12, 6))
        for col in df.columns:
            plt.plot(df[col], color='black', alpha=0.5)
            for cluster_label in np.unique(labels):
                if cluster_label == -1:
                    color = noise_color  # Noise
                else:
                    color = cluster_colors[cluster_label]  # Cluster colors

                # Plot only the segments that belong to the current cluster
                cluster_indices = np.where(labels == cluster_label)[0]
                plt.plot(df.index[cluster_indices],
                         df[col].iloc[cluster_indices], 'o',
                         label=f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise',
                         color=color)

            plt.title(f'Time Series: {col}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.show()
            plot_tree(clf, feature_names=df.columns, class_names=[f'Cluster {i}' for i in np.unique(labels)],
                      filled=True, fontsize=10)
            plt.show()

    def nearest(self, reduced_data):
        k = 2  # Choose the same value as min_samples in DBSCAN
        nbrs = NearestNeighbors(n_neighbors=k).fit(reduced_data)
        distances, indices = nbrs.kneighbors(reduced_data)

        # Sort distances (k-th nearest neighbor distance for each point)
        distances = np.sort(distances[:, k - 1])

        # Plot the k-distance plot
        plt.plot(distances)
        plt.ylabel('k-distance')
        plt.xlabel('Data Points sorted by distance')
        plt.title('k-distance Graph to determine optimal eps')
        plt.show()


if __name__ == '__main__':
    # test = Dbscan(drop_list=('MA', 'delta'))
    # test = Kmeans_single()
    test = Kmeans_multiple(feature="", excludelist=("MA", "delta"))
    # test = Dbscan(select_list=('delta'))

