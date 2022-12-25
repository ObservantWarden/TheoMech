import sys

import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Вычисляем относительную позицию стрелки
def rotation_2d(x, y, angle):
    return [x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)]


def close():
    plt.close()
    sys.exit(0)


class MyPlotter:

    # Задаём константы, просчитываем траектории
    def __init__(self):
        r = 4
        # Константы и некоторые объекты, которые мы нарисуем
        self.__circle = None
        self.__drawn_arrow = None
        self.__point = None
        time = sp.Symbol('t')
        self.__radius = 1 + sp.sin(5 * time)
        self.__arrow_coordinates = [np.array([-0.2 * r, 0, -0.2 * r]),
                                    np.array([0.1 * r, 0, -0.1 * r])]

        self.__angular_velocity = time

        # Функции
        x_formula = self.__radius * sp.cos(time)
        y_formula = self.__radius * sp.sin(time)
        velocity_formula = [sp.diff(x_formula, time), sp.diff(y_formula, time)]

        time_range = np.linspace(0, 40, 1000)

        self.__x_values = np.zeros_like(time_range)
        self.__y_values = np.zeros_like(time_range)
        self.__velocity_values = [np.empty(1000), np.empty(1000)]
        self.__position = np.zeros_like(time_range)
        self.__radius_values = np.zeros_like(time_range)

        # Рассчитываем траекторию и скорость в каждый момент времени
        for moment in np.arange(len(time_range)):
            self.__radius_values[moment] = sp.Subs(self.__radius, time, time_range[moment])
            self.__x_values[moment] = sp.Subs(x_formula, time, time_range[moment])
            self.__y_values[moment] = sp.Subs(y_formula, time, time_range[moment])
            self.__velocity_values[0][moment] = sp.Subs(velocity_formula[0], time, time_range[moment])
            self.__velocity_values[1][moment] = sp.Subs(velocity_formula[1], time, time_range[moment])

    def draw_plot(self):
        # Настройки осей и маштаба
        self.__fig = plt.figure()
        r = 4

        main_plot = self.__fig.add_subplot(1, 1, 1)

        main_plot.axis('equal')
        main_plot.set(xlim=[r, 12 * r], ylim=[-r, 3 * r])

        # Рисуем траекторию и опору для колеса
        main_plot.plot(self.__x_values, self.__y_values)
        main_plot.plot([self.__x_values.min(), self.__x_values.max()], [0, 0], 'black')

        # Добавляем точку
        self.__point = main_plot.plot(self.__x_values[0], self.__y_values[0], marker='o', color='red')[0]

        # Добавляем прямую
        self.__velocity_line = main_plot.plot(
            [self.__x_values[0], self.__x_values[0] + self.__velocity_values[0][0]],
            [self.__y_values[0], self.__y_values[0] + self.__velocity_values[1][0]], 'r')[0]

        # Вычисляем положение головы стрелки
        rotated_arrow = rotation_2d(*self.__arrow_coordinates,
                                    math.atan2(self.__velocity_values[1][0],
                                               self.__velocity_values[0][0]))

        # Рисуем стрелку
        self.__drawn_arrow = main_plot.plot(rotated_arrow[0] + self.__x_values[0] + self.__velocity_values[0][0],
                                            rotated_arrow[1] + self.__y_values[0] + self.__velocity_values[1][0])[0]

        # Рисуем покатушку
        self.__circle = main_plot.plot(
           self.__x_values, self.__y_values, 'g')[0]

        anim = FuncAnimation(self.__fig, self.animation, frames=1000, interval=2, repeat=True)
        plt.show()

        # Функция анимации

    def animation(self, i):
        self.__point.set_data(self.__x_values[i], self.__y_values[i])
        self.__velocity_line.set_data([self.__x_values[i], self.__x_values[i] + self.__velocity_values[0][i]],
                                      [self.__y_values[i], self.__y_values[i] + self.__velocity_values[1][i]])
        inner_rotated_arrow = rotation_2d(*self.__arrow_coordinates,
                                          math.atan2(self.__velocity_values[1][i], self.__velocity_values[0][i]))
        self.__drawn_arrow.set_data(inner_rotated_arrow[0] + self.__x_values[i] + self.__velocity_values[0][i],
                                    inner_rotated_arrow[1] + self.__y_values[i] + self.__velocity_values[1][i])
