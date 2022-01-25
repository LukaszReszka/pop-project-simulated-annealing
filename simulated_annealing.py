import math
import random
from typing import Callable

Q_FUNC = Callable[[list[float]], float]
COOLING_FUNC = Callable[[int], float]
FLOAT_VECT = list[float]
TUPLE_POINTS_AND_Q = tuple[list[list[float]], list[float]]


class SimulatedAnnealing:
    def __init__(self, q_function: Q_FUNC, x_dimension: int, neighbor_radius: float = 1.0,
                 minimize_func: bool = True) -> None:
        self.q = q_function
        self.dim = x_dimension
        self.minimize = minimize_func
        self.n_radius = neighbor_radius
        self.squared_radius = neighbor_radius ** 2

                      
    def run_algorithm(self, init_range_min: int, init_range_max: int, max_iter: int,
                      cooling_func: COOLING_FUNC) -> TUPLE_POINTS_AND_Q:
        points = []
        q_val = []
        accept_worse_point = True
        x = self._init_x(init_range_min, init_range_max)

        for iteration in range(max_iter):
            y = self._select_neighbour(x)
            q_y = self.q(y)
            q_x = self.q(x)
                               
            if (self.minimize is True and q_y < q_x) or (self.minimize is False and q_y > q_x):
                x = y

            elif accept_worse_point: 
                t_i = cooling_func(iteration) 
                if t_i <= 0: 
                    accept_worse_point = False
                elif random.uniform(0, 1) < math.exp(-((abs(q_y - q_x)) / t_i)):  # p_a < exp(-|q_y - q_x|/T)
                    x = y

            points.append(x) 
            q_val.append(self.q(x)) 

        return points, q_val

    def _init_x(self, range_min: int, range_max: int) -> FLOAT_VECT:
        x = []
        for i in range(self.dim):
            x.append(random.uniform(range_min, range_max))

        return x

    def _select_neighbour(self, x: FLOAT_VECT) -> FLOAT_VECT:  
        is_point_inside_neighborhood = False                   
        y = []

        while not is_point_inside_neighborhood:
            sum_to_check = 0

            for i in range(self.dim):
                x_i = random.uniform(x[i] - self.n_radius, x[i] + self.n_radius)
                y.append(x_i)
                sum_to_check += ((x[i] - x_i) ** 2)

            if sum_to_check <= self.squared_radius:  # check if (x[0] - x_0)^2 + ... + (x[n] - x_n))^2 <= radius^2
                is_point_inside_neighborhood = True
            else:
                y = []

        return y
