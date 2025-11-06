import pandas as pd
from random import random
import numpy as np
from price_simulation import sim_stock_price


def get_d1(S_0, K, r, T, deviation):
    return (np.log(S_0 / K) + (r +(deviation ** 2 / 2)) * T) / (deviation * np.sqrt(T))

##Estima la integral de funciong entre -inf y a con 5000 simulaciones
def monte_carlo_inf_a(g, a, n_sim):
    suma = 0
    for _ in range(n_sim):
        u = random()
        suma += g(a - (1/u - 1)) / (u**2)
    return suma / n_sim

def funciong(x):
    return np.exp((-x**2)/2)

def norm_cdf(x):
    estim_monte_carlo = monte_carlo_inf_a(funciong, x, n_sim=1000)
    return 1 / np.sqrt(2 * np.pi) * estim_monte_carlo

def delta_call_long(S0, K, r, T, deviation):
    x = get_d1(S0, K, r, T, deviation)
    return norm_cdf(x)

def delta_put_long(S0, K, r, T, deviation):
    x = get_d1(S0, K, r, T, deviation)
    return norm_cdf(x) - 1

def main():
    K = 50
    S_0 = 49
    deviation = 0.20
    T = 0.3846 # años, equiv a 20 semanas
    n_sim = 1000
    r = 0.05
    BSM_price = 2.40

    d1 = get_d1(S_0, K, r, T, deviation)
    Delta = delta_call_long(S_0, K, r, T, deviation)

    print(f"d1: {d1}")
    print(f"N(d1) (Delta call): {Delta}")

if __name__ == '__main__':
    main()
