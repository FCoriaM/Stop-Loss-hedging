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

def get_DeltaS(media, deviation, S, Delta_t, epsilon):
    DeltaS = media * S * Delta_t + deviation * S * np.sqrt(Delta_t) * epsilon
    return DeltaS

def montecarlo(Delta_t, media, deviation, stock_price):
    column_names = ['t', 'Stock price at t', 'Random sample for epsilon', 'Delta S in period']
    stock_simulation_df = pd.DataFrame(columns=column_names)
    S = stock_price
    t = 0
    for _ in range(100):
        epsilon = np.random.normal(loc=0,scale=1)
        Delta_S = get_DeltaS(media, deviation, S, Delta_t, epsilon)
        new_row_data = {
                        't': t,
                        'Stock price at t': S, 
                        'Random sample for epsilon': epsilon, 
                        'Delta S in period': Delta_S
                        }
        stock_simulation_df.loc[len(stock_simulation_df)] = new_row_data

        S += Delta_S
        t += Delta_t
    return stock_simulation_df

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

