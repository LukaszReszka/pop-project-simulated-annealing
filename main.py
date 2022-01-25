from tests import *
from functions import *

def main():
    print("Annealing")
    show_sample_functions()
    display_visual_results()
    test_efficiency(rosenbrock_f, func_name="rosenbrock", show_chart=True, logging=True)
    test_efficiency(bird_f, func_name="rbird function", show_chart=True, logging=True)
    test_efficiency(shubert_f, func_name="shubert function", show_chart=True, logging=True)
    test_efficiency(cec_func, func_name="cec_func")


main()