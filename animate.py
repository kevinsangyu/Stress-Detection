from os.path import join
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from extractData import Extract


class Animator(object):
    def __init__(self, dataframes):
        self.dataframes = dataframes

    def hr(self):
        data = self.dataframes.HR.df[0].tolist()
        duration = [i for i in range(len(data))]
        self.anim(data, duration, label="Heart Rate")

    def bvp(self):
        data = self.dataframes.BVP.df[0].tolist()
        duration = [i for i in range(len(data))]
        self.anim(data, duration, label="Blood Volume Pressure")

    def temp(self):
        data = self.dataframes.TEMP.df[0].tolist()
        duration = [i for i in range(len(data))]
        self.anim(data, duration, label="Temperature")

    def eda(self):
        data = self.dataframes.EDA.df[0].tolist()
        duration = [i for i in range(len(data))]
        self.anim(data, duration, label="ElectroDermal Activity")

    def tags(self):
        data = self.dataframes.TAGS
        duration = [i for i in range(len(data))]
        self.anim(data, duration, label="Tags")

    def anim_all(self):
        self.hr()
        self.bvp()
        self.temp()
        self.eda()
        self.tags()

    def plot(self, data, duration):
        fig, ax = plt.subplots()
        ax.plot(duration, data)
        ax.set(xlabel="Time (s)", ylabel="HeartRate", title="Vs Ball Torture, OH HR")
        plt.show()

    def anim(self, data, duration, intv=1, label='Def Label', window=200):
        print("Attempting plot...")
        fig, ax = plt.subplots()
        line, = ax.plot(duration, data)
        total_num = len(data)

        def init():
            if label == "Blood Volume Pulse":
                ax.set_ylim(-600, 600)
            ax.set(xlabel="Time (s)", ylabel=label)
            return line,

        def update(num):
            start = max(0, num - window)
            line.set_data(duration[start:num], data[start:num])
            ax.set_xlim((num/intv)-20, (num/intv)+5)
            ax.set(ylabel=f'{label} - {data[num]}', xlabel=f"Time (s) - {duration[num]}")
            if num % (total_num // 100) == 0:
                print(f"Update iteration {num} of {total_num} - {num/total_num*100//1}%")
            return line,

        ani = animation.FuncAnimation(
            fig, update, frames=len(data), init_func=init, blit=False, interval=(1000/intv)
        )

        output_path = join(r'E:\UserData\Coding\python\stress\output', f'{label}.mp4')
        ani.save(output_path, writer='ffmpeg')
        print("Animated successfully saved.")


if __name__ == '__main__':
    e = Extract(r"data/Kevin data/2024_06_24_finals")
    e.homogenise(method="window")
    a = Animator(e)
    print(len(e.TAGS))
    print([i for i, n in enumerate(e.TAGS) if n == 1])
    a.tags()
