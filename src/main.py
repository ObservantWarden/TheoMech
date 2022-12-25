import sys

from MyPlotter import MyPlotter, close

if __name__ == '__main__':
    plt = MyPlotter()

    try:
        plt.draw_plot()
    except KeyboardInterrupt:
        close()
        sys.exit(0)
