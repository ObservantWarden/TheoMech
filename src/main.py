import sys

from MyPlotter import MyPlotter

plt = MyPlotter(4, 1)

try:
    plt.draw_plot()
except KeyboardInterrupt:
    plt.close()
    sys.exit(0)
