import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def rotation(x_values, y_values, angle):
    rotating_x = x_values * np.cos(angle) - y_values * np.sin(angle)
    rotating_y = x_values * np.sin(angle) + y_values * np.cos(angle)
    return rotating_x, rotating_y


Centre, Line = None, None
Circle1, Circle2 = None, None
A, B = None, None
Ax, Ay, Bx, By = None, None, None, None
Spr = None


def Animation(moment):
    Centre.set_data(x_shift + move[moment], y_shift)
    Circle1.set_data(inner_x_values + move[moment], inner_y_values)
    Circle2.set_data(outer_x_values + move[moment], outer_y_values)
    A.set_data(Ax[moment], Ay[moment])
    B.set_data(Bx[moment], By[moment])
    animated_x_distance = Bx[moment] - Ax[moment]
    animated_y_distance = By[moment] - Ay[moment]
    animated_stretch = np.sqrt(animated_x_distance ** 2 + animated_y_distance ** 2)
    animated_alpha = np.pi + np.arctan2(animated_y_distance, animated_x_distance)
    animated_rotated_x, animated_rotated_y = rotation(ball_x_values * animated_stretch, ball_y_values, animated_alpha)
    Spr.set_data(animated_rotated_x + Bx[moment], animated_rotated_y + By[moment])
    return [Circle1, Circle2, Centre, Spr, A, B]


if __name__ == "__main__":
    main_plot = plt.figure(figsize=[6.5, 6.5])
    subplot = main_plot.add_subplot(1, 1, 1)
    subplot.set(xlim=[-20, 20], ylim=[-20, 20])

    steps = 1000
    t = np.linspace(0, 10, steps)

    phi_angle_func = np.sin(t)
    psi_angle_func = np.cos(1.2 * t)

    first_radius, second_radius = 5, 4
    x_lower, x_upper = -15, 15
    y_lower = y_upper = -first_radius
    x_shift, y_shift = 0, 0

    move = first_radius * psi_angle_func
    betta = np.linspace(0, 6.28, 1000)
    inner_x_values = first_radius * np.sin(betta) + x_shift
    inner_y_values = first_radius * np.cos(betta) + y_shift
    outer_x_values = second_radius * np.sin(betta) + x_shift
    outer_y_values = second_radius * np.cos(betta) + y_shift

    Ax = first_radius * np.sin(psi_angle_func) + move + x_shift
    Ay = first_radius * np.cos(psi_angle_func) + y_shift

    pipe_radius = (first_radius - second_radius) / 2 + second_radius
    Bx = pipe_radius * np.sin(phi_angle_func) + move + x_shift
    By = pipe_radius * np.cos(phi_angle_func) + y_shift

    n = 15
    b = 1 / (n - 2)
    sh = 0.4
    ball_x_values = np.zeros(n)
    ball_y_values = np.zeros(n)
    ball_x_values[0] = 0
    ball_x_values[n - 1] = 1
    ball_y_values[0] = 0
    ball_y_values[n - 1] = 0
    for i in range(n - 2):
        ball_x_values[i + 1] = b * (i + 1) - b / 2
        ball_y_values[i + 1] = sh * (-1) ** i

    x_distance = Bx[0] - Ax[0]
    y_distance = By[0] - Ay[0]
    stretch = np.sqrt(x_distance ** 2 + y_distance ** 2)
    alpha_angle = np.pi + np.arctan2(y_distance, x_distance)
    rotated_x, rotated_y = rotation(ball_x_values * stretch, ball_y_values, alpha_angle)

    Centre = subplot.plot(x_shift + move[0], y_shift, 'white', marker='o', ms=10, mec="c")[0]
    Line = subplot.plot([x_lower, x_upper], [y_lower, y_upper], 'black')
    Circle1 = subplot.plot(inner_x_values + move[0], inner_y_values, color="c")[0]
    Circle2 = subplot.plot(outer_x_values + move[0], outer_y_values, color="c")[0]

    Spr = subplot.plot(rotated_x + Bx[0], rotated_y + By[0], 'red')[0]

    A = subplot.plot(Ax[0], Ay[0], 'k', marker='o', ms=4)[0]
    B = subplot.plot(Bx[0], Ay[0], 'k', marker='o', ms=8)[0]

    a = FuncAnimation(main_plot, Animation, frames=steps, interval=10)
    plt.show()
