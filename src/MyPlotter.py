import sys

import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Вычисляем относительную позицию стрелки
def rotation_2d(x, y, angle):
    return [x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)]


class MyPlotter:

    # Задаём константы, просчитываем траектории
    def __init__(self, radius, angular_velocity):
        # Константы и некоторые объекты, которые мы нарисуем
        self.__velocity_line = None
        self.__circle = None
        self.__drawn_arrow = None
        self.__point = None
        self.__radius = radius
        self.__arrow_coordinates = [np.array([-0.2 * self.__radius, 0, -0.2 * self.__radius]),
                                  np.array([0.1 * self.__radius, 0, -0.1 * self.__radius])]

        self.__angular_velocity = angular_velocity
        time = sp.Symbol('t')

        # Функции
        x_formula = radius * (angular_velocity * time - sp.sin(angular_velocity * time))
        y_formula = radius * (1 - sp.cos(angular_velocity * time))
        velocity_formula = [sp.diff(x_formula, time), sp.diff(y_formula, time)]
        position_formula = self.__radius * angular_velocity * time

        time_range = np.linspace(0, 10, 1000)

        self.__x_values = np.zeros_like(time_range)
        self.__y_values = np.zeros_like(time_range)
        self.__velocity_values = [np.empty(1000), np.empty(1000)]
        self.__position = [np.empty(1000), np.empty(1000)]

        # Рассчитываем траекторию и скорость в каждый момент времени
        for moment in np.arange(len(time_range)):
            self.__x_values[moment] = sp.Subs(x_formula, time, time_range[moment])
            self.__y_values[moment] = sp.Subs(y_formula, time, time_range[moment])
            self.__velocity_values[0][moment] = sp.Subs(velocity_formula[0], time, time_range[moment])
            self.__velocity_values[1][moment] = sp.Subs(velocity_formula[1], time, time_range[moment])
            self.__position[0][moment] = sp.Subs(position_formula, time, time_range[moment])

    def draw_plot(self):
        # Настройки осей и маштаба
        self.__fig = plt.figure()

        main_plot = self.__fig.add_subplot(1, 1, 1)

        main_plot.axis('equal')
        main_plot.set(xlim=[self.__radius, 12 * self.__radius], ylim=[-self.__radius, 3 * self.__radius])

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
        pi = np.linspace(0, 6.28, 100)
        self.__circle = main_plot.plot(
            self.__position[0][0] + self.__radius * np.cos(pi),
            self.__radius + self.__radius * np.sin(pi), 'g')[0]

        anim = FuncAnimation(self.__fig, self.animation, frames=1000, interval=2, repeat=True)
        plt.show()

        # Функция анимации

    def animation(self, i):
        pi = np.linspace(0, 6.28, 100)
        self.__point.set_data(self.__x_values[i], self.__y_values[i])
        self.__velocity_line.set_data([self.__x_values[i], self.__x_values[i] + self.__velocity_values[0][i]],
                                    [self.__y_values[i], self.__y_values[i] + self.__velocity_values[1][i]])
        inner_rotated_arrow = rotation_2d(self.__arrow_coordinates[0], self.__arrow_coordinates[1],
                                          math.atan2(self.__velocity_values[1][i], self.__velocity_values[0][i]))
        self.__drawn_arrow.set_data(inner_rotated_arrow[0] + self.__x_values[i] + self.__velocity_values[0][i],
                                  inner_rotated_arrow[1] + self.__y_values[i] + self.__velocity_values[1][i])
        self.__circle.set_data(self.__position[0][i] + self.__radius * np.cos(pi),
                             self.__radius + self.__radius * np.sin(pi))

    def close(self):
        plt.close()
        sys.exit(0)
