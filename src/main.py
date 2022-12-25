import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import math


def rotation(x_values, y_values, angle):
    rotating_x = x_values * np.cos(angle) - y_values * np.sin(angle)
    rotating_y = x_values * np.sin(angle) + y_values * np.cos(angle)
    return rotating_x, rotating_y


def equation_system(inner_y_value, inner_t_value, inner_m_value, bigger_m_value, inner_r_value, inner_g_value,
                    inner_c_value):
    yt = np.zeros_like(inner_y_value)

    yt[0] = inner_y_value[2]
    yt[1] = inner_y_value[3]

    a = (inner_y_value[0] + inner_y_value[1]) / 2

    a11 = 1
    a12 = np.cos(inner_y_value[0])
    a21 = np.cos(inner_y_value[0])
    a22 = 1 + 2 * (bigger_m_value / m_value)

    b1 = -(inner_g_value / inner_r_value) * \
         np.sin(inner_y_value[0]) - 2 * (inner_c_value / inner_m_value) * (1 - np.cos(a)) * np.sin(a)
    b2 = inner_y_value[2] ** 2 * \
         np.sin(inner_y_value[0]) - 2 * (inner_c_value / inner_m_value) * (1 - np.cos(a)) * np.sin(a)

    yt[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    yt[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)

    return yt


def Animation(moment):
    centre.set_data(x_shift + move[moment], y_shift)
    inner_circle.set_data(inner_x_values + move[moment], inner_y_values)
    outer_circle.set_data(outer_x_values + move[moment], outer_y_values)
    A.set_data(Ax[moment], Ay[moment])
    B.set_data(Bx[moment], By[moment])

    distance_x = Bx[moment] - Ax[moment]
    distance_y = By[moment] - Ay[moment]
    animated_stretch = np.sqrt(distance_x ** 2 + distance_y ** 2)
    animated_alpha = np.pi + np.arctan2(distance_y, distance_x)
    rotating_x, rotating_y = rotation(ball_x_values * animated_stretch, ball_y_values, animated_alpha)

    spr.set_data(rotating_x + Bx[moment], rotating_y + By[moment])
    return [inner_circle, outer_circle, centre, spr, A, B]


if __name__ == "__main__":
    plot = plt.figure(figsize=[6.5, 6.5])
    subplot = plot.add_subplot(1, 1, 1)
    subplot.set(xlim=[-20, 20], ylim=[-20, 20])

    m_value = 0.1
    big_m_value = 1
    r_value = 10
    start_t_value = 0
    c_value = 0
    g_value = 10

    start_phi_angle = math.pi / 3
    start_psi_angle = 0
    start_phi_derivative = 0
    start_psi_derivative = 0

    y_start = [start_phi_angle, start_psi_angle, start_phi_derivative, start_psi_derivative]
    steps = 1001
    t_finish = 20
    t = np.linspace(0, t_finish, 1001)

    y_values = odeint(equation_system, y_start, t, (m_value, big_m_value, r_value, g_value, c_value))

    phi_angle_func = y_values[:, 0]
    psi_angle_func = y_values[:, 1]
    phi_pr = y_values[:, 2]
    psi_pr = y_values[:, 3]

    phi_ppr = np.zeros((len(t)))
    psi_ppr = np.zeros((len(t)))
    for i in range(len(t)):
        Res = equation_system(y_values[i, :], t[i], m_value, big_m_value, r_value, g_value, c_value)
        phi_ppr[i] = Res[2]
        psi_ppr[i] = Res[3]

    N = m_value * (g_value * np.cos(phi_angle_func) + r_value * (
            phi_pr ** 2 - psi_ppr * np.sin(phi_angle_func))) + 2 * r_value * c_value * (
                1 - np.cos((phi_angle_func + psi_angle_func) / 2)) * np.cos((phi_angle_func + psi_angle_func) / 2)

    # графики зависимости координат и реакции от времени

    plot_for_graphs = plt.figure(figsize=[13, 7])
    subplot_for_graph = plot_for_graphs.add_subplot(2, 2, 1)
    subplot_for_graph.plot(t, phi_angle_func, color='blue')
    subplot_for_graph.set_title("phi(t)")
    subplot_for_graph.set(xlim=[0, t_finish])
    subplot_for_graph.grid(True)

    subplot_for_graph = plot_for_graphs.add_subplot(2, 2, 2)
    subplot_for_graph.plot(t, psi_angle_func, color='red')
    subplot_for_graph.set_title('psi(t)')
    subplot_for_graph.set(xlim=[0, t_finish])
    subplot_for_graph.grid(True)

    subplot_for_graph = plot_for_graphs.add_subplot(2, 2, 3)
    subplot_for_graph.plot(t, N, color='black')
    subplot_for_graph.set_title('N(t)')
    subplot_for_graph.set(xlim=[0, t_finish])
    subplot_for_graph.grid(True)

    first_radius = r_value
    second_radius = r_value - 1
    x_lower = -15
    x_upper = 15
    y_lower = y_upper = -first_radius
    x_shift = 0
    y_shift = 0

    move = first_radius * psi_angle_func
    betta = np.linspace(0, 6.28, 100)
    inner_x_values = first_radius * np.sin(betta) + x_shift
    inner_y_values = first_radius * np.cos(betta) + y_shift
    outer_x_values = second_radius * np.sin(betta) + x_shift
    outer_y_values = second_radius * np.cos(betta) + y_shift

    pipe_radius = (first_radius - second_radius) / 2 + second_radius

    Ax = first_radius * np.sin(psi_angle_func) + move + x_shift
    Ay = first_radius * np.cos(psi_angle_func) + y_shift

    Bx = pipe_radius * np.sin(phi_angle_func) + move + x_shift
    By = -pipe_radius * np.cos(phi_angle_func) + y_shift

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

    centre = subplot.plot(x_shift + move[0], y_shift, 'white', marker='o', ms=10, mec="c")[0]
    line = subplot.plot([x_lower, x_upper], [y_lower, y_upper], 'black')[0]
    inner_circle = subplot.plot(inner_x_values + move[0], inner_y_values, color="c")[0]
    outer_circle = subplot.plot(outer_x_values + move[0], outer_y_values, color="c")[0]

    spr = subplot.plot(rotated_x + Bx[0], rotated_y + By[0], 'red')[0]

    A = subplot.plot(Ax[0], Ay[0], 'k', marker='o', ms=4)[0]
    B = subplot.plot(Bx[0], Ay[0], 'k', marker='o', ms=8)[0]

    a = FuncAnimation(plot, Animation, frames=steps, interval=10)
    plt.show()
