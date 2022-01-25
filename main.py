import logging
from tests import *
from functions import *
import pandas as pd

def show_sample_functions():
    # display of a few functions used to test alghoritms 
    dim3_chart(shubert_f, 'shubert function')
    dim3_chart(rosenbrock_f, 'rosenbrock function')
    dim3_chart(func2, 'function 2')
    dim3_chart(bird_f, 'bird function')

def display_visual_results():
    run_all(func0, x_dim=1, t_0=2.0, param=0.1, init_min=-5, init_max=5, max_iter=100, tabu_size=2, plotting=True, show_data=True)
    run_all(func1, x_dim=1, t_0=2.0, param=0.1, init_min=-5, init_max=5, max_iter=100, tabu_size=2, plotting=True, show_data=True)
    run_all(func2, x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=2, plotting=True, show_data=True)
    run_all(rosenbrock_f, x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=2, plotting=True, show_data=True)
    run_all(shubert_f, x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=2, plotting=True, show_data=True)
    run_all(bird_f, x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=2, plotting=True, show_data=True)
    run_all(cec_func, x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=100, tabu_size=2, plotting=True, show_data=True)
    run_all(cec_func, x_dim=10, t_0=10.0, param=0.1, init_min=-10, init_max=10, max_iter=100, tabu_size=2, plotting=False, show_data=True)

def get_test_data(func):
    t_0_data = [1.0, 2.0, 10.0, 50.0, 100.0, 200.0]
    param_data = [0.1, 0.2, 0.5]
    init_data = [5, 10, 20, 50, 100, 200, 500, 1000, 4000]
    tabu_size_data = [2, 10, 20]
    #functions = [func1, func2, func3, rosenbrock_f, bird_f, cec_func]
    functions = [func]
    return [t_0_data, param_data, init_data, tabu_size_data, functions]

def get_quality_results(func, get_test_data, logging, max_iter=100):
    t_0_data, param_data, init_data, tabu_size_data, functions = get_test_data(func)
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

def show_chart_all(y1, y2, y3, y4, y5, y6, y7, y8, title="chart"):
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

def best_efficiency(st, li, ge, lo):
    # best = max(population, key=lambda genome: evaluate_genome(genome, func))
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


def main():
    print("Annealing")
    # show_sample_functions()
    # display_visual_results()
    # test_efficiency(rosenbrock_f, func_name="rosenbrock")
    test_efficiency(bird_f, func_name="rbird function")
    # test_efficiency(cec_func, func_name="cec_func")
    # st, li, ge, lo = get_quality_results(rosenbrock_f, get_test_data, logging=True)
    # show_chart_all(st[0], st[1], li[0], li[1], ge[0], ge[1], lo[0], lo[1], "rosenbrock_f")
    # ef1 = best_efficiency(st, li, ge, lo)
    # print(ef1)
    # print(percentage(ef1))

main()