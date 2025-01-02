import matplotlib.pyplot as plt
from sktime.annotation.clasp import ClaSPSegmentation
from sktime.annotation.plotting.utils import plot_time_series_with_change_points, plot_time_series_with_profiles
from sktime.datasets import load_electric_devices_segmentation
import ruptures as rpt
import extractData


class CLASP(object):
    def __init__(self, method="sampling", delta_switch=True):
        self.data = extractData.Extract(r"data/Kevin data/2024_04_22_vs_pegasus")
        homogenise_method = method
        delta = delta_switch
        if delta:
            self.data.delta()
        self.data.homogenise(method=homogenise_method, size=10)

        self.ts = self.data.HR.df
        if homogenise_method == "window":
            if delta:
                self.ts = self.ts['MA_delta']
            else:
                self.ts = self.ts['MA']
        else:
            if delta:
                self.ts = self.ts['delta']
            else:
                self.ts = self.ts[0]

    def segment(self):
        n_cps = 5
        period_length = 10  # ts.size // n_cps
        print(f"n_cps: {n_cps}, period_length: {period_length}")
        clasp = ClaSPSegmentation(period_length=period_length, n_cps=n_cps)
        found_cps = clasp.fit_predict(self.ts)
        print("The found change points are", found_cps.to_numpy())

        indices = [i for i, x in enumerate(self.data.TAGS) if x == 1]
        print("True change points/tags")
        _ = plot_time_series_with_profiles(
            "Volleyball game",
            self.ts,
            clasp.profiles,
            indices,
            found_cps,
        )
        plt.show()

    def test(self):
        ts, period_size, true_cps = load_electric_devices_segmentation()
        _ = plot_time_series_with_change_points("Electric Devices", ts, true_cps)
        clasp = ClaSPSegmentation(period_length=period_size, n_cps=5)
        found_cps = clasp.fit_predict(ts)
        profiles = clasp.profiles
        scores = clasp.scores
        print("The found change points are", found_cps.to_numpy())
        _ = plot_time_series_with_profiles(
            "Electric Devices",
            ts,
            profiles,
            true_cps,
            found_cps,
        )
        plt.show()


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
        algo = rpt.Binseg(model=model, min_size=10).fit(ts.values)
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
    rup = RUPTURES()
    rup.segment(feature="delta", exclude="MA")
    # clsp = CLASP(delta_switch=True)
    # clsp.segment()
