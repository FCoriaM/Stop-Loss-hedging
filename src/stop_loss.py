import pandas as pd
import numpy as np
from price_simulation import sim_stock_price


def get_hedge_preformance():
    pass

def stop_loss_sim(K, S_0, media, deviation, delta_t):
    
    prev_S = S_0
    hedging_cost = S_0 if S_0 > 0 else 0
    for _ in range(20):
        epsilon, delta_s, new_S, t = sim_stock_price(media, deviation, prev_S, delta_t)
        if K < new_S and prev_S < K:
            # buy stock
            hedging_cost += new_S - K
        elif new_S < K and K < prev_S:
            # sell stock
            hedging_cost += K - new_S

        prev_S = new_S

    return hedging_cost


def main():
    pass

if __name__ == '__main__':
    pass