import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_plot(df: pd.DataFrame):
    ax = df.plot(x='t', y='Stock price at t', kind='line')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock price')
    ax.set_title('Stock price variation Graph')
    # plt.savefig('stock_price_simulation.png')
    plt.show()    

def get_delta_s(media, deviation, S, delta_t, epsilon):
    delta_s = media * S * delta_t + deviation * S * np.sqrt(delta_t) * epsilon
    return delta_s

def sim_stock_price(media, deviation, S, delta_t):
    epsilon = np.random.normal(loc=0,scale=1)
    delta_s = get_delta_s(media, deviation, S, delta_t, epsilon)
    new_S += delta_s
    t += delta_t
    return epsilon, delta_s, new_S, t

def montecarlo(delta_t, media, deviation, stock_price):
    column_names = ['t', 'Stock price at t', 'Random sample for epsilon', 'Delta S in period']
    prices_df = pd.DataFrame(columns=column_names)
    S = stock_price
    t = 0
    for _ in range(100):
        epsilon, delta_s, S, t = sim_stock_price(media, deviation, delta_t)
        new_row_data = {
                        't': t,
                        'Stock price at t': S, 
                        'Random sample for epsilon': epsilon, 
                        'Delta S in period': delta_s
                        }
        prices_df.loc[len(prices_df)] = new_row_data
    return prices_df

def main():
    interval = 0.0192 # years, equiv to 1week
    media = 0.15
    deviation = 0.30
    initial_stock_price = 100
    df = montecarlo(delta_t=interval, media=media, deviation=deviation, stock_price=initial_stock_price)
    get_plot(df)

if __name__ == '__main__':
    main()

