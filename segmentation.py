import matplotlib.pyplot as plt
from claspy.segmentation import BinaryClaSPSegmentation
from sktime.annotation.plotting.utils import plot_time_series_with_change_points, plot_time_series_with_profiles
from sktime.datasets import load_electric_devices_segmentation
import ruptures as rpt
import extractData
import pandas as pd


class CLASP(object):
    def __init__(self, method="sampling", delta_switch=True):
        self.data = extractData.Extract(r"data/Kevin data/2024_04_29_vs_snortsnort")
        self.data.delta()
        self.data.homogenise(method="window", size=10)
        self.data.combine_df()
        self.df = self.data.df
        print(f"Available features: {self.df.columns}")

    def segment(self, feature="HR", exclude="no feature should have this string.!@#)(*%"):
        if not feature:
            # use all features if no specific one is provided
            ts = self.df
            cols = [col for col in self.df.columns]
        else:
            # grab the MA, delta and raw features of given feature
            # also works in grabbing just MA values or just delta values if you use 'MA' or 'delta'
            cols = [col for col in self.df.columns if feature in col and exclude not in col]
            print(f"Selected columns: {cols}")
            ts = self.df[cols]
        ts.dropna(inplace=True)

        clasp = BinaryClaSPSegmentation(n_segments=5, validation=None)
        found_cps = clasp.fit_predict(time_series=ts.values)
        print("The found change points are", found_cps)

        indices = [i for i, x in enumerate(self.data.TAGS) if x == 1]
        print("True change points/tags: ", indices)
        clasp.plot(gt_cps=pd.Series(indices),
                   heading=f"Segmentation of Volleyball game",
                   ts_name=feature, font_size=18)
        plt.show()

    def test(self):
        from claspy.data_loader import load_tssb_dataset
        dataset, window_size, true_cps, time_series = load_tssb_dataset(names=("CricketX",)).iloc[0, :]
        clasp = BinaryClaSPSegmentation()
        clasp.fit_predict(time_series)
        print(clasp.window_size)


class RUPTURES(object):
    def __init__(self):
        self.data = extractData.Extract(r"data/Kevin data/2024_04_29_vs_snortsnort")
        self.data.delta()
        self.data.homogenise(method="window", size=10)
        self.data.combine_df()
        self.df = self.data.df
        print(f"Available features: {self.df.columns}")

    def segment(self, feature="HR", exclude="no feature should have this string.!@#)(*%"):
        if not feature:
            # use all features if no specific one is provided
            ts = self.df
            cols = [col for col in self.df.columns]
        else:
            # grab the MA, delta and raw features of given feature
            # also works in grabbing just MA values or just delta values if you use 'MA' or 'delta'
            cols = [col for col in self.df.columns if feature in col and exclude not in col]
            print(f"Selected columns: {cols}")
            ts = self.df[cols]
        ts.dropna(inplace=True)

        # select model and algorithm then predict
        model = "rbf"
        algo = rpt.Dynp(model=model, min_size=10).fit(ts.values)
        print("Segmenting...")
        predicted = algo.predict(n_bkps=6)

        # display
        indices = [i for i, x in enumerate(self.data.TAGS) if x == 1]
        fig, axarr = rpt.display(ts.values, predicted, indices, figsize=(10, 6))

        keep_axes = []
        for i in range(len(axarr)):
            if not feature:
                # to increase visibility, delete the MA and delta features from the graph, if using all features.
                if "MA" in cols[i] or "delta" in cols[i]:
                    fig.delaxes(axarr[i])
                    continue
            axarr[i].title.set_text(cols[i])
            keep_axes.append(axarr[i])
        # reposition remaining/kept axes
        if not feature:
            for i, ax in enumerate(keep_axes):
                ax.set_position([
                    0.05,  # x
                    0.9 - i * 0.12,  # y
                    0.90,  # width
                    0.08  # height
                ])
        plt.show()


if __name__ == '__main__':
    # rup = RUPTURES()
    # rup.segment(feature="")
    clsp = CLASP()
    # clsp.test()
    clsp.segment(feature="EDA", exclude="MA")
