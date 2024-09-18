import matplotlib.pyplot as plt
import json


class HeartRateGrapher(object):
    def __init__(self, file_loc):
        self.f = open(file_loc)
        self.data = json.load(self.f)
        self.hr = []
        self.time = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def playthrough(self, speed_multiplier=1, x_view_lim=30):
        # time is tracked through ms passed since epoch, so we keep a track of the starting time
        start_time = self.data[0]["start_time"]
        # this is for bugs since sometimes the time data is wrong (from samsung side)

        # metadata
        peak = int(self.data[0]["heart_rate"])
        trough = peak

        print("Play through commencing")
        for i in range(len(self.data)):
            self.ax.clear()
            t = (self.data[i]["start_time"] - start_time) / 1000
            interval = 0.01
            if i != len(self.data)-1:
                # the pause function on the bottom draws then waits, so we have to calculate the interval between
                # the current measurement and the next measurement
                interval = (self.data[i+1]["start_time"]-self.data[i]["start_time"]) / 1000
                if interval > 5 or interval < 0:
                    print(f"{self.data[i+1]['start_time']}-{self.data[i]['start_time']} = {self.data[i+1]['start_time']-self.data[i]['start_time']}")
                    print("Gap between time is bugged, skipping")
                    continue

            # updating list of data
            heartrate = int(self.data[i]["heart_rate"])
            self.hr.append(heartrate)
            self.time.append(t)
            if len(self.time) > x_view_lim:
                self.hr.pop(0)
                self.time.pop(0)

            # updating metadata
            if peak < heartrate:
                peak = heartrate
            if trough > heartrate:
                trough = heartrate

            # updating labels
            self.ax.set_title(f"low: {trough}, high: {peak}\nbpm, sec: {heartrate}, {t}", loc='right')
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Heart Rate")

            # plotting
            if t > x_view_lim:
                plt.xlim(t - x_view_lim, t + 1)
            self.ax.plot(self.time, self.hr)

            # waiting
            plt.pause(interval / speed_multiplier)
        print("Play through complete")
        plt.show()

    def graph(self):
        start_time = self.data[0]["start_time"]
        prev = 0

        # plot labels
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Heart Rate")

        # metadata
        peak = int(self.data[0]["heart_rate"])
        trough = peak

        print("Graphing commenced")
        for i in self.data:
            t = (i["start_time"] - start_time) / 1000
            interval = t - prev
            if interval > 5 or interval < 5 and interval < 0:
                print("Gap between time is bugged, skipping")
                continue
            prev = t

            # updating list of data
            heartrate = int(i["heart_rate"])
            self.hr.append(heartrate)
            self.time.append(t)

            # updating metadata
            if peak < heartrate:
                peak = heartrate
            if trough > heartrate:
                trough = heartrate

        self.ax.set_title(f"low: {trough}, high: {peak}", loc='right')
        self.ax.plot(self.time, self.hr)
        print("Graphing concluded.")
        plt.show()


if __name__ == '__main__':
    grapher = HeartRateGrapher("cougars.json")
    grapher.playthrough(speed_multiplier=1, x_view_lim=60)
    # grapher.graph()
