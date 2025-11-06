import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_plot(df: pd.DataFrame):
    ax = df.plot(x='t', y='Stock price at t', kind='line')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock price')
    ax.set_title('Stock price variation Graph')
    plt.savefig('stock_price_simulation.png')
    plt.show()    

def sim_stock_price(mu, sigma, S, dt):
    eps = np.random.normal(0,1)
    dS = 0.05 * S * dt + sigma * S * np.sqrt(dt) * eps
    S_new = max(S + dS, 1e-12)
    return S_new

def simulate_single_path(S0, mu, sigma, dt, n_steps):
    
    t = np.arange(n_steps + 1) * dt
    S = np.empty(n_steps + 1, dtype=float)
    S[0] = S0
    for k in range(1, n_steps + 1):
        S[k] = sim_stock_price(mu, sigma, S[k-1], dt)
    return t, S

def main():
    T = 0.3846 # años, equiv a 20 semanas
    n_steps = 20            # rebalanceo semanal
    interval = T / n_steps  # Δt
    media = 0.13
    deviation = 0.20
    initial_stock_price = 49
    r = 0.05
    K = 50
    BSM_price = 2.40

if __name__ == '__main__':
    main()

