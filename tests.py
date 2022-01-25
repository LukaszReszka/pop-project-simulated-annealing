from matplotlib import pyplot as plt
from matplotlib import cm
import simulated_annealing as sa
import cooling_schedule as cs
import annealing_with_tabu as at
import numpy as np
from functions import *

def show_chart(x_vals, y_vals, color='r', title="chart"):
    plt.plot(x_vals, y_vals, color=color, marker='.', label="sym")
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

def dim3_chart(func, description):
    x1 = np.linspace(-10, 10, 100)
    x2 = np.linspace(-10, 10, 100)
    r_min,r_max = -10, 10
    x1, x2 = np.meshgrid(x1, x2)
    results = func([x1, x2])
    figure = plt.figure(figsize = (9,9))
    axis = figure.gca(projection = '3d')
    axis.contour3D(x1, x2, results, 15)
    axis.set_title(description)
    axis.plot_surface(x1,x2, results, cmap = cm.rainbow)
    axis.view_init(elev = 21,azim = 42)
    axis.set_xlabel('X') 
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    plt.contour(x1, x2, results,15)
    plt.show()

def print_data(points, q_val):
    row_num = 1
    for point, q in zip(points, q_val):
        i = 0
        label = ""
        row = ""
        for part in point:
            label += f"x{i} "
            row += f"{part} "
            i += 1
        print(f"row {row_num} : {label+row} : q {q}")
        row_num += 1

def results(func, cooling_func, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data):
    alg = sa.SimulatedAnnealing(func, x_dim)
    points, q_val = alg.run_algorithm(init_min, init_max, max_iter, cooling_func)
    if show_data:
        description = "Simulated Annealing"
        print(description)
        print_data(points, q_val)
    if plotting:
        if x_dim == 1:
            show_chart([point[0] for point in points], q_val, color='r', title=description)
        if x_dim == 2:
            show_chart([point[0] for point in points], [point[1] for point in points], color='r', title=description)

    alg2 = at.AnnealingWithTabu(func, x_dim)
    points2, q_val2 = alg2.run_algorithm(init_min, init_max, max_iter, tabu_size, cooling_func)
    if show_data:
        description2 = "Simulated Annealing with tabu"
        print(description2)
        print_data(points2, q_val2)
    if plotting:
        if x_dim == 1:
            show_chart([point2[0] for point2 in points2], q_val2, color='r', title=description2)
        if x_dim == 2:
            show_chart([point2[0] for point2 in points2], [point2[1] for point2 in points2], color='r', title=description2)

    return [q_val[-1], q_val2[-1]]
                            


def run_all(func, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data):
    col = cs.StandardCoolingSchedule(t_0, param)
    optimum_result1 = results(func, col.standard_cooling, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data)
    col = cs.LinearCoolingSchedule(t_0, param)
    optimum_result2 = results(func, col.linear_cooling, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data)
    col = cs.GeometricCoolingSchedule(t_0, param)
    optimum_result3 = results(func, col.geometric_cooling, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data)
    col = cs.LogarithmicCoolingSchedule(t_0, param)
    optimum_result4 = results(func, col.logarithmic_cooling, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data)
    optimum_results = [optimum_result1, optimum_result2, optimum_result3, optimum_result4]
    for result in optimum_results:
        print(result)






