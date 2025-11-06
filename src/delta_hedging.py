import pandas as pd
from random import random
import numpy as np
from price_simulation import sim_stock_price


def get_d1(S_0, K, r, tau, deviation):
    return (np.log(S_0 / K) + (r +(deviation ** 2 / 2)) * tau) / (deviation * np.sqrt(tau))

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

def delta_call(S0, K, r, tau, sigma):
    x = get_d1(S0, K, r, tau, sigma)
    return norm_cdf(x)

def delta_put(S0, K, r, tau, sigma):
    x = get_d1(S0, K, r, tau, sigma)
    return norm_cdf(x) - 1

def delta_hedging_single_sim(S0, K, r, sigma, T, amt_options, n_steps):
    """
    Simula una estrategia de cobertura Delta-hedging paso a paso.
    Devuelve un DataFrame con el detalle de cada paso temporal.

    Columnas:
      t : tiempo actual
      S(t) : precio del subyacente
      Delta : Delta de la opción
      Acciones compradas : variación de posición en el subyacente
      Costo acciones compradas : flujo de caja por esa variación
      Costo acumulado : costo total acumulado hasta ese momento
      Costo de Interes : interés pagado sobre el costo acumulado previo
    """

    dt = T / n_steps
    df = pd.DataFrame(columns=[
        't', 'S(t)', 'Delta', 'Acciones compradas',
        'Costo acciones compradas', 'Costo acumulado', 'Costo de Interes'
    ])

    # Condiciones iniciales
    S = S0
    Delta_prev = delta_call(S, K, r, T, sigma)
    cum_cost = Delta_prev * S / 1000  # en millones
    interest_cost = 0

    # Agregar fila inicial
    df.loc[0] = [0, S, Delta_prev, Delta_prev * amt_options,
                 cum_cost, cum_cost, interest_cost]

    # Simulación paso a paso
    for k in range(1, n_steps + 1):
        # Simular nuevo precio
        S = sim_stock_price(r, sigma, S, dt)
        # Tiempo restante hasta el vencimiento
        tau = T - k * dt
        # Calcular nueva Delta
        Delta = delta_call(S, K, r, tau, sigma)
        # Calcular acciones a comprar/vender
        acciones = (Delta - Delta_prev) * amt_options
        # Flujo de caja asociado (en millones)
        costo_acciones = acciones * S / 1000
        # Interés del costo acumulado anterior
        interes = cum_cost * r * dt
        # Actualizar costo acumulado
        cum_cost = cum_cost * (1 + r * dt) + costo_acciones
        # Agregar fila al DataFrame
        df.loc[k] = [k, S, Delta, acciones,
                     costo_acciones, cum_cost, interes]
        # Actualizar para el siguiente paso
        Delta_prev = Delta

    return df

def main():
    K = 50
    S_0 = 49
    sigma = 0.20
    T = 0.3846 # años, equiv a 20 semanas
    n_sim = 1000
    r = 0.05
    amt_options = 100_000 # cantidad de opciones que se firmaron en el contrato
    BSM_price = 2.40
    n_steps = 20


    df = delta_hedging_single_sim(S_0, K, r, sigma, T, amt_options, n_steps)
    print(df.head(20))

if __name__ == '__main__':
    main()
