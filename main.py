import simulated_annealing as sa
import cooling_schedule as cs
import annealing_with_tabu as at


def func(x: list[float]) -> float:
    return x[0] ** 2


lin = cs.LinearCoolingSchedule(2.0, 0.1)

alg = sa.SimulatedAnnealing(func, 1)
points, q_val = alg.run_algorithm(-5, 5, 100, lin.linear_cooling)

alg2 = at.AnnealingWithTabu(func, 1)
points2, q_val2 = alg2.run_algorithm(-5, 5, 100, 1, lin.linear_cooling)

print("Simulated annealing:\n")
print(points)
print(q_val)
print("\n")

print("Annealing with tabu:\n")
print(points2)
print(q_val2)
print("\n")

