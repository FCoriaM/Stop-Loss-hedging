import pandas as pd
import numpy as np
from price_simulation import sim_stock_price


def get_hedge_preformance(Cs, BSM_price):
    std_dev = np.std(Cs, ddof=1)
    return std_dev / BSM_price

def sim_stock_price(r, sigma, S, dt):
    eps = np.random.normal(0,1)
    dS = r * S * dt + sigma * S * np.sqrt(dt) * eps
    S_new = max(S + dS, 1e-12)
    return S_new

def stop_loss_single_sim(K, S_0, r, deviation, delta_t, n_steps):
    S = np.empty(n_steps + 1, dtype=float)
    S[0] = S_0
    t = 0
    hedging_cost = 0
    for t in range(1, n_steps + 1):
        S[t] = sim_stock_price(r, deviation, S[t-1], delta_t)
        if K < S[t] and S[t-1] < K:
            # buy stock
            hedging_cost += S[t]
        elif S[t] < K and K < S[t-1]:
            # sell stock
            hedging_cost -= S[t]
    
    if S[n_steps] > K:
        hedging_cost -= K

    return S, hedging_cost

def montecarlo_stop_loss(K, S_0, r, deviation, delta_t, n_steps, n_sim):
    # C = hedging costs array
    C = np.empty(n_sim, dtype=float)
    for i in range(n_sim):
        _, C[i] = stop_loss_single_sim(K, S_0, r, deviation, delta_t, n_steps)
    return C
        

def main():
    K = 50
    S_0 = 49
    deviation = 0.20
    T = 0.3846 # años, equiv a 20 semanas
    n_sim = 1000
    r = 0.05
    BSM_price = 2.40

    n_steps = [4, 5, 10, 20, 40, 80]
    deltas_t = [T/n_step for n_step in n_steps]
    hedges_performances = []


    for delta_t, n_step in zip(deltas_t, n_steps):
        Cs = montecarlo_stop_loss(K, S_0, r, deviation, delta_t, n_step, n_sim)    
        performance = get_hedge_preformance(Cs, BSM_price)
        hedges_performances.append(performance)
        
    dts_weeks = [round(dt * 52, 2) for dt in deltas_t]


    print("\nTabla 19.1 - Performance de la cobertura Stop-Loss")
    print("--------------------------------------------------")
    print(f"{'Δt (semanas)':>14} | {'Performance':>12}")
    print("---------------|--------------")
    for dt, perf in zip(dts_weeks, hedges_performances):
        print(f"{dt:>14.2f} | {perf:>12.5f}")

if __name__ == '__main__':
    main()