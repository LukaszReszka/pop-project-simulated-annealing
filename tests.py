from matplotlib import pyplot as plt
from matplotlib import cm
import simulated_annealing as sa
import cooling_schedule as cs
import annealing_with_tabu as at
import numpy as np
from functions import *

def show_chart(x_vals, y_vals, color='r', title="chart", label="projection of convergence patch in 2d", x_label="x values", y_label="y values"):
    plt.plot(x_vals, y_vals, color=color, marker='.', label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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

def results(func, cooling_func, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data, func_name=""):
    alg = sa.SimulatedAnnealing(func, x_dim)
    points, q_val = alg.run_algorithm(init_min, init_max, max_iter, cooling_func)
    if show_data:
        description = "Simulated Annealing " + func_name
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
        description2 = "Simulated Annealing with tabu " + func_name
        print(description2)
        print_data(points2, q_val2)
    if plotting:
        if x_dim == 1:
            show_chart([point2[0] for point2 in points2], q_val2, color='r', title=description2)
        if x_dim == 2:
            show_chart([point2[0] for point2 in points2], [point2[1] for point2 in points2], color='r', title=description2)
        
    

    return [q_val[-1], q_val2[-1]]
                            


def run_all(func, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data, logging=False, func_name=""):
    if logging:
        print(1)
    col = cs.StandardCoolingSchedule(t_0, param)
    optimum_result1 = results(func, col.standard_cooling, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data, func_name=func_name+" StandardCooling")
    if logging:
        print(2)
    col = cs.LinearCoolingSchedule(t_0, param)
    optimum_result2 = results(func, col.linear_cooling, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data, func_name=func_name+" Linear Cooling")
    if logging:
        print(3)
    col = cs.GeometricCoolingSchedule(t_0, param)
    optimum_result3 = results(func, col.geometric_cooling, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data, func_name=func_name+" Geometric Cooling")
    if logging:
        print(4)
    col = cs.LogarithmicCoolingSchedule(t_0, param)
    optimum_result4 = results(func, col.logarithmic_cooling, x_dim, t_0, param, init_min, init_max, max_iter, tabu_size, plotting, show_data, func_name=func_name+" Logharytmic Cooling")
    optimum_results = [optimum_result1, optimum_result2, optimum_result3, optimum_result4]
    return optimum_results


def show_sample_functions():
    # display of a few functions used to test alghoritms 
    dim3_chart(shubert_f, 'shubert function')
    dim3_chart(rosenbrock_f, 'rosenbrock function')
    dim3_chart(func2, 'function 2')
    dim3_chart(bird_f, 'bird function')

def display_visual_results():
    run_all(func0, func_name="func0", x_dim=1, t_0=2.0, param=0.1, init_min=-5, init_max=5, max_iter=100, tabu_size=2, plotting=True, show_data=True)
    run_all(func1, func_name="func1", x_dim=1, t_0=2.0, param=0.1, init_min=-5, init_max=5, max_iter=100, tabu_size=2, plotting=True, show_data=True)
    run_all(func2, func_name="func2", x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=2, plotting=True, show_data=True)
    run_all(rosenbrock_f, func_name="rosenbrock_f", x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=2, plotting=True, show_data=True)
    run_all(shubert_f,func_name="shubert_f", x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=2, plotting=True, show_data=True)
    run_all(bird_f, func_name="bird_f", x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=20, plotting=True, show_data=True)
    run_all(cec_func, func_name="cec_func", x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=100, tabu_size=2, plotting=True, show_data=True)
    run_all(cec_func, func_name="cec_func", x_dim=10, t_0=10.0, param=0.1, init_min=-10, init_max=10, max_iter=100, tabu_size=2, plotting=False, show_data=True)

def get_test_data(func):
    t_0_data = [1.0, 2.0, 10.0, 50.0, 100.0, 200.0]
    param_data = [0.1, 0.2, 0.5]
    init_data = [5, 10, 20, 50, 100, 200, 500, 1000, 4000]
    tabu_size_data = [2, 10, 20]
    functions = [func]
    return [t_0_data, param_data, init_data, tabu_size_data, functions]

def get_test_data_temp(func, temp):
    t_0_data = temp
    param_data = [0.1]
    init_data = [100]
    tabu_size_data = [5]
    functions = [func]
    return [t_0_data, param_data, init_data, tabu_size_data, functions]

def get_test_data_tabu(func, tabu):
    t_0_data = [1.0]
    param_data = [0.1]
    init_data = [100]
    tabu_size_data = tabu
    functions = [func]
    return [t_0_data, param_data, init_data, tabu_size_data, functions]

def get_quality_results(func, get_test_data, logging, max_iter=100, temp=None, tabu=None):
    if not temp and not tabu:
        t_0_data, param_data, init_data, tabu_size_data, functions = get_test_data(func)
    elif not tabu:
        t_0_data, param_data, init_data, tabu_size_data, functions = get_test_data(func, temp)
    else:
        t_0_data, param_data, init_data, tabu_size_data, functions = get_test_data(func, tabu)
    standard_cool_res = [[], []]
    linear_cool_res = [[], []]
    geometric_cool_res = [[], []]
    logarithmic_cool_res = [[], []]
    for t_0 in t_0_data:
        for param in param_data:
            for init in init_data:
                for tabu_size in tabu_size_data:
                    for func in functions:
                        dim = 0
                        if func != cec_func:
                            dim = 2
                        else:
                            dim = 10
                        if logging:
                            print("t_0 ", t_0, " param ", param, "init ", init, "tabu_size ", tabu_size)
                        st, li, ge, lo = run_all(func, x_dim=dim, t_0=t_0, param=param, init_min=-init, init_max=init, max_iter=max_iter, tabu_size=tabu_size, plotting=False, show_data=False, logging=logging)
                        standard_cool_res[0].append(st[0])
                        standard_cool_res[1].append(st[1])
                        linear_cool_res[0].append(li[0])
                        linear_cool_res[1].append(li[1])
                        geometric_cool_res[0].append(ge[0])
                        geometric_cool_res[1].append(ge[1])
                        logarithmic_cool_res[0].append(lo[0])
                        logarithmic_cool_res[1].append(lo[1])

    return [standard_cool_res, linear_cool_res, geometric_cool_res, logarithmic_cool_res]

def show_chart_all(y1, y2, y3, y4, y5, y6, y7, y8, title="chart", x=None):
    if x == None:
        x = [i for i in range(1, len(y1) + 1)]
    plt.scatter(x, y1, color='red', marker='.', label="no tabu - standard cool")
    plt.scatter(x, y2, color='orange', marker='.', label="tabu - standard cool")
    plt.scatter(x, y3, color='blue', marker='.', label="no tabu - linear cool")
    plt.scatter(x, y4, color='cyan', marker='.', label="tabu - linear cool")
    plt.scatter(x, y5, color='forestgreen', marker='.', label="no tabu - geometric cool")
    plt.scatter(x, y6, color='limegreen', marker='.', label="tabu - geometric cool")
    plt.scatter(x, y7, color='violet', marker='.', label="no tabu - logarytmic cool")
    plt.scatter(x, y8, color='purple', marker='.', label="tabu - logarytmic cool")
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

def show_chart_all_sep(ys, title="chart", x=None, x_label="temperature"):
    if x == None:
        x = [i for i in range(1, len(ys[0]) + 1)]
    colors = ['red', 'orange', 'blue', 'cyan', 'forestgreen', 'limegreen', 'violet', 'purple']
    labels = ["no tabu - standard cool", "tabu - standard cool", "no tabu - linear cool", "tabu - linear cool",
                "no tabu - geometric cool", "tabu - geometric cool", "no tabu - logarytmic cool", "tabu - logarytmic cool"
                    ]

    for y, color, label in zip(ys, colors, labels):
        show_chart(x, y, color=color, title="best solution found", label=label, x_label=x_label, y_label="q values")


def best_efficiency(st, li, ge, lo):
    evaluation = [0,0,0,0,0,0,0,0]
    for i in range(0, len(st[0])):
        l = [st[0][i], st[1][i], li[0][i], li[1][i], ge[0][i], ge[1][i], lo[0][i], lo[1][i]]
        best = max(l)
        for j in range(0, len(l)):
            if l[j] == best:
                evaluation[j] += 1
                break
    
    return evaluation

def percentage(evaluation):
    percentage = []
    for ev in evaluation:
        percentage.append(ev / sum(evaluation))
    labels = ["standard_cooling", "standard_cooling_tabu", "linear_cooling", "linear_cooling_tabu",
                "geometric_cooling", "geometric_cooling_tabu", "logharitmic_cooling", "logharitmic__cooling_tabu"]
    plt.bar(labels, percentage, color='orange')
    plt.xlabel('Type')
    plt.ylabel('Percentage of best solution found')
    plt.title('Efficience for each cooling sheadule')
    plt.show()
    
    return percentage
    
def test_efficiency(func, func_name, show_chart=True, logging=True):
    st, li, ge, lo = get_quality_results(func, get_test_data, logging=logging)
    if show_chart:
        show_chart_all(st[0], st[1], li[0], li[1], ge[0], ge[1], lo[0], lo[1], func_name)
    ef = best_efficiency(st, li, ge, lo)
    return percentage(ef)

def test_efficiency_for_temp(func, func_name, show_chart=True, logging=True, temps=None, write_to_file=True):
    st, li, ge, lo = get_quality_results(func, get_test_data_temp, logging=logging, temp=temps)
    results = [st[0], st[1], li[0], li[1], ge[0], ge[1], lo[0], lo[1]]
    for i in range(0, 100):
        st, li, ge, lo = get_quality_results(func, get_test_data_temp, logging=logging, temp=temps)
        new_results = [st[0], st[1], li[0], li[1], ge[0], ge[1], lo[0], lo[1]]
        for j in range(0, len(results)):
            for k in range(0, len(results[0])):
                results[j][k] = min(results[j][k], new_results[j][k])

    if show_chart:
        show_chart_all_sep(results, func_name, x=temps)

    if write_to_file:
        f = open(f"{func_name}_res.txt", "a")
        f.write(f"temperature : shema_nums \n")
        f.write(f"T : 1 : 2 : 3 : 4 : 5 : 6 : 7 : 8 \n")
        print(results)
        for i in range(0, len(results[0])):
            f.write(f"t:{temps[i]} ")
        f.write("\n")
        for i in range(0, len(results)):
            for j in range(0, len(results[0])):
                f.write(f"{results[i][j]} ")
            f.write(f"\n")
        
        f.write("------------------------------------------------------")
        f.write('\n')
        f.close()

    return results

def test_efficiency_for_tabu(func, func_name, show_chart=True, logging=True, tabus=None, write_to_file=True):
    st, li, ge, lo = get_quality_results(func, get_test_data_tabu, logging=logging, tabu=tabus)
    results = [st[0], st[1], li[0], li[1], ge[0], ge[1], lo[0], lo[1]]
    for i in range(0, 100):
        st, li, ge, lo = get_quality_results(func, get_test_data_tabu, logging=logging, tabu=tabus)
        new_results = [st[0], st[1], li[0], li[1], ge[0], ge[1], lo[0], lo[1]]
        for j in range(0, len(results)):
            for k in range(0, len(results[0])):
                results[j][k] = min(results[j][k], new_results[j][k])

    if show_chart:
        show_chart_all_sep(results, func_name, x=tabus, x_label="tabu size")

    if write_to_file:
        f = open(f"{func_name}_res_tabu.txt", "a")
        f.write(f"tabu : shema_nums \n")
        print(results)
        for i in range(0, len(results[0])):
            f.write(f"t:{tabus[i]} ")
        f.write("\n")
        for i in range(0, len(results)):
            for j in range(0, len(results[0])):
                f.write(f"{results[i][j]} ")
            f.write(f"\n")
        
        f.write("------------------------------------------------------")
        f.write('\n')
        f.close()

    return results

def test_efficiency_for_temp_all():
    test_efficiency_for_temp(rosenbrock_f, func_name="rosenbrock", temps=[i for i in range(1, 200, 5)], show_chart=True, logging=True, write_to_file=True)
    test_efficiency_for_temp(shubert_f, func_name="shubert", temps=[i for i in range(1, 200, 5)], show_chart=True, logging=True, write_to_file=True)
    test_efficiency_for_temp(bird_f, func_name="bird", temps=[i for i in range(1, 200, 5)], show_chart=True, logging=True, write_to_file=True)
    test_efficiency_for_temp(cec_func, func_name="cec", temps=[i for i in range(1, 200, 15)], show_chart=True, logging=True, write_to_file=True)
 
def test_efficiency_for_tabu_all():
    test_efficiency_for_tabu(rosenbrock_f, func_name="rosenbrock", tabus=[i for i in range(1, 100, 4)], show_chart=True, logging=True, write_to_file=True)
    test_efficiency_for_tabu(shubert_f, func_name="shubert", tabus=[i for i in range(1, 100, 4)], show_chart=True, logging=True, write_to_file=True)
    test_efficiency_for_tabu(bird_f, func_name="bird", tabus=[i for i in range(1, 100, 4)], show_chart=True, logging=True, write_to_file=True)
    test_efficiency_for_tabu(cec_func, func_name="cec", tabus=[2, 5, 10, 20, 30, 40, 50, 75, 100], show_chart=True, logging=True, write_to_file=True)

def run_old_tests():
    test_efficiency(rosenbrock_f, func_name="rosenbrock function", show_chart=True, logging=True)
    test_efficiency(bird_f, func_name="rbird function", show_chart=True, logging=True)
    test_efficiency(shubert_f, func_name="shubert function", show_chart=True, logging=True)
    test_efficiency(cec_func, func_name="cec_func")