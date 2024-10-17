from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.model_selection import train_test_split
import extractData
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.tree import _tree, DecisionTreeClassifier, plot_tree


class Kmeans_single():
    def __init__(self):
        prep = MaxAbsScaler()

        data = extractData.Extract(r"data/Kevin data/2024_04_22_vs_pegasus")
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
        algorithm = KMeans(n_clusters=3, random_state=0)
        algorithm.fit(scaled_data)
        # identify what the clusters are discriminated by (centroids)
        # decision tree to characterise the clusters

        # try diff clustering: dbscan: https://scikit-learn.org/stable/modules/clustering.html

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

        print(df.to_string())

        plt.figure(figsize=(8, 6))
        for i in df.index:
            # for column in df.columns:
            plt.plot(i, df['HR'][i], 'o', color=color_dict.get(i, 'black'))
        plt.show()


if __name__ == '__main__':
    test = Kmeans_single()
