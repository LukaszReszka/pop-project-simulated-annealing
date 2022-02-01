from tests import *
from functions import *

def main():
    print("Annealing")
    show_sample_functions()
    display_visual_results()

    test_efficiency_for_temp_all()
    test_efficiency_for_tabu_all()

    # run_old_tests()



main()