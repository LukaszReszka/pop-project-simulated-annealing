import math


class StandardCoolingSchedule:  # T_i = T_0 * exp(-i * alfa), 0 < alfa < 1
    def __init__(self, t_0: float, alfa: float) -> None:
        self.t_0 = t_0
        self.alfa = alfa

    def standard_cooling(self, i: int) -> float:
        return self.t_0 * math.exp(-i * self.alfa)


class LinearCoolingSchedule:  # T_i = T_0 - i * delta_t, delta_t >= 0
    def __init__(self, t_0: float, delta_t: float) -> None:
        self.t_0 = t_0
        self.delta_t = delta_t

    def linear_cooling(self, i: int) -> float:
        return self.t_0 - i * self.delta_t


class GeometricCoolingSchedule:  # T_i = T_0 * a^i, 0 < a < 1
    def __init__(self, t_0: float, a: float) -> None:
        self.t_0 = t_0
        self.a = a

    def geometric_cooling(self, i: int) -> float:
        return self.t_0 * (self.a ** i)


class LogarithmicCoolingSchedule:  # T_i = a*T_0/ln(i+2)
    def __init__(self, t_0: float, a: float) -> None:
        self.t_0 = t_0
        self.a = a

    def logarithmic_cooling(self, i: int) -> float:
        return self.a * self.t_0 / math.log(i + 2)
