import pandas as pd
import numpy as np
from math import erf
from price_simulation import sim_stock_price


def get_d1(S_0, K, r, T, deviation):
    return (np.log(S_0 / K) + (r +(deviation ** 2 / 2)) * T) / (deviation * np.sqrt(T))

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def delta_call_long(S0, K, r, T, deviation):
    x = get_d1(S0, K, r, T, deviation)
    return norm_cdf(x)

def delta_put_long(S0, K, r, T, deviation):
    x = get_d1(S0, K, r, T, deviation)
    return norm_cdf(x) - 1.0

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
