from tests import *
from functions import *

def show_sample_functions():
    # display of a few functions used to test alghoritms 
    dim3_chart(shubert_f, 'shubert function')
    dim3_chart(rosenbrock_f, 'rosenbrock function')
    dim3_chart(func2, 'function 2')
    dim3_chart(bird_f, 'bird function')

def display_visual_results():
    run_all(func0, x_dim=1, t_0=2.0, param=0.1, init_min=-5, init_max=5, max_iter=100, tabu_size=2, plotting=False, show_data=True)
    run_all(func1, x_dim=1, t_0=2.0, param=0.1, init_min=-5, init_max=5, max_iter=100, tabu_size=2, plotting=False, show_data=True)
    run_all(func2, x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=2, plotting=True, show_data=True)
    run_all(rosenbrock_f, x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=2, plotting=True, show_data=True)
    run_all(shubert_f, x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=2, plotting=True, show_data=True)
    run_all(bird_f, x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=1000, tabu_size=2, plotting=True, show_data=True)
    run_all(cec_func, x_dim=2, t_0=2.0, param=0.1, init_min=-100, init_max=100, max_iter=100, tabu_size=2, plotting=True, show_data=True)
    run_all(cec_func, x_dim=10, t_0=10.0, param=0.1, init_min=-10, init_max=10, max_iter=100, tabu_size=2, plotting=False, show_data=True)

def get_test_data():
    t_0_data = [1.0, 2.0, 10.0, 50.0, 100.0, 200.0]
    param_data = [0.1, 0.2, 0.5]
    init_data = [5, 10, 20, 50, 100, 200, 500, 1000, 4000]
    tabu_size_data = [2, 10, 20, 50]
    #functions = [func1, func2, func3, rosenbrock_f, bird_f, cec_func]
    functions = [rosenbrock_f, bird_f, cec_func]
    return [t_0_data, param_data, init_data, tabu_size_data, functions]

def get_quality_results(get_test_data, max_iter=200):
    t_0_data, param_data, init_data, tabu_size_data, functions = get_test_data()
    standard_cool_res = []
    linear_cool_res = []
    geometric_cool_res = []
    logarithmic_cool_res = []
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
                        st, li, ge, lo = run_all(func, x_dim=dim, t_0=t_0, param=param, init_min=-init, init_max=init, max_iter=max_iter, tabu_size=tabu_size, plotting=False, show_data=False)
                        standard_cool_res.append(st)
                        linear_cool_res.append(li)
                        geometric_cool_res.append(ge)
                        logarithmic_cool_res.append(lo)

    return [standard_cool_res, linear_cool_res, geometric_cool_res, logarithmic_cool_res]

def main():
    print("Annealing")
    show_sample_functions()
    display_visual_results()
    # print(get_quality_results(get_test_data)) # tu będzie jeszcze porównanie, efektywności między metodami, w zależności od dostarczanych wyników 

main()