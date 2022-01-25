from matplotlib import pyplot as plt
import simulated_annealing as sa
import cooling_schedule as cs
import annealing_with_tabu as at
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

def main():
    print("Annealing")
    show_sample_functions()
    display_visual_results()

main()