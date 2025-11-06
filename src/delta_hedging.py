import pandas as pd
from random import random
import numpy as np
from price_simulation import sim_stock_price


def get_d1(S_0, K, r, tao, deviation):
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

def delta_call(S0, K, r, tao, sigma):
    x = get_d1(S0, K, r, tao, sigma)
    return norm_cdf(x)

def delta_put(S0, K, r, tao, sigma):
    x = get_d1(S0, K, r, tao, sigma)
    return norm_cdf(x) - 1

def delta_hedging_single_sim(S0, K, Delta0, r, sigma, T, amt_options, n_steps):
    """
    t | S(t) | Delta | Acciones compradas | Costo acciones compradas | Costo acumulado | Costo de Interes  
    """
    columns = ['t', 'S(t)', 'Delta', 'Acciones compradas', 'Costo acciones compradas', 'Costo acumulado', 'Costo de Interes']
    df = pd.DataFrame(columns=columns)
    dt = T / n_steps
    S = np.empty(n_steps + 1, dtype=float)
    Deltas = np.empty(n_steps + 1, dtype=float)
    shares_purchased = np.empty(n_steps + 1, dtype=float)
    cost_s_p = np.empty(n_steps + 1, dtype=float)
    cum_cost = np.empty(n_steps + 1, dtype=float)
    interest_cost = np.empty(n_steps + 1, dtype=float)

    S[0] = S0
    Deltas[0] = Delta0
    shares_purchased[0] = Deltas[0] * amt_options 
    cost_s_p[0] = shares_purchased[0] * S[0] / 1000 # in millions
    cum_cost[0] = cost_s_p[0]
    interest_cost[0] = 0

    for k in range(1, n_steps + 1):
        S[k] = sim_stock_price(r, sigma, S[k-1], dt)
        tao = T - k*dt
        Deltas[k] = delta_call(S0, K, r, tao, sigma)
        shares_purchased[k] = (Deltas[k] - Deltas[k-1]) * amt_options
        cost_s_p[k] = shares_purchased[k] * S[k] / 1000 # in millions
        cum_cost[k] = cost_s_p[k-1] * (1 + r * dt) + cost_s_p[k]
        interest_cost[k] = cum_cost[k-1] * r * dt
    return df

def main():
    K = 50
    S_0 = 49
    deviation = 0.20
    T = 0.3846 # años, equiv a 20 semanas
    n_sim = 1000
    r = 0.05
    amt_options = 100_000 # cantidad de opciones que se firmaron en el contrato
    BSM_price = 2.40


    Delta = delta_call_long(S_0, K, r, T, deviation)

    print(f"N(d1) (Delta call): {Delta}")

if __name__ == '__main__':
    main()
