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

def get_DeltaS(media, deviation, stock_price, Delta_t, epsilon):
    DeltaS = media * stock_price * Delta_t + deviation * stock_price * np.sqrt(Delta_t) * epsilon
    return DeltaS

def montecarlo(Delta_t, media, deviation, stock_price):
    column_names = ['t', 'Stock price at t', 'Random sample for epsilon', 'Delta S in period']
    stock_simulation_df = pd.DataFrame(columns=column_names)
    S = stock_price
    t = 0
    for _ in range(100):
        epsilon = np.random.normal(loc=0,scale=1)
        Delta_S = get_DeltaS(media, deviation, stock_price, Delta_t, epsilon)
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
    interval = 0.0192 # years, equiv to 1week
    media = 0.15
    deviation = 0.30
    initial_stock_price = 100
    df = montecarlo(Delta_t=interval, media=media, deviation=deviation, stock_price=initial_stock_price)
    get_plot(df)

if __name__ == '__main__':
    main()

