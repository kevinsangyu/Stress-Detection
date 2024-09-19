import matplotlib.pyplot as plt
from sktime.annotation.clasp import ClaSPSegmentation
from sktime.annotation.plotting.utils import plot_time_series_with_change_points, plot_time_series_with_profiles
from sktime.datasets import load_electric_devices_segmentation
import ruptures as rpt
import extractData


class CLASP(object):
    def __init__(self):
        self.data = extractData.Extract(r"data/Kevin data/2024_04_22_vs_pegasus")
        homogenise_method = "sampling"
        delta = True
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
        self.data = extractData.Extract(r"data/Kevin data/2024_04_22_vs_pegasus")
        homogenise_method = "sampling"
        self.data.homogenise(method=homogenise_method, size=10)

        self.ts = self.data.HR.df
        if homogenise_method == "window":
            self.ts = self.ts['MA']
        else:
            self.ts = self.ts[0]

    def segment(self):
        model = "rbf"
        print("f1")
        algo = rpt.Dynp(model=model, min_size=10).fit(self.ts.values)
        print("f2")
        predicted = algo.predict(n_bkps=6)
        print("f3")
        indices = [i for i, x in enumerate(self.data.TAGS) if x == 1]
        # rpt.show.display(self.ts.values, predicted, figsize=(10, 6))
        rpt.display(self.ts.values, indices, predicted)
        plt.show()


if __name__ == '__main__':
    # rup = RUPTURES()
    # rup.segment()
    clsp = CLASP()
    clsp.segment()
