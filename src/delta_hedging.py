import pandas as pd
from random import random
import numpy as np
from price_simulation import sim_stock_price
from stop_loss import get_hedge_preformance
from scipy.stats import norm


def get_d1(S_0, K, r, tau, deviation):
    tau_eff = max(tau, 1e-12)
    return (np.log(S_0 / K) + (r +(deviation ** 2 / 2)) * tau_eff) / (deviation * np.sqrt(tau_eff))

##Estima la integral de funciong entre -inf y a con n_sim simulaciones
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
    return norm.cdf(x)

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
        'Week', 'Precio de la acción', 'Delta', 'Acciones compradas',
        'Costo acciones compradas ($000)', 'Costo acumulado ($000)', 'Costo de Interes ($000)'
    ])

    # Condiciones iniciales
    S = S0
    Delta_prev = delta_call(S, K, r, T, sigma)
    acciones_0 = Delta_prev * amt_options
    costo_0 = acciones_0 * S
    accum_cost = costo_0
    interes_0 = accum_cost * r * dt
    
    # Agregar fila inicial
    df.loc[0] = [0, S, Delta_prev, acciones_0,
                 costo_0, accum_cost, interes_0]

    # Simulación paso a paso
    for k in range(1, n_steps + 1):
        # Simular nuevo precio
        S = sim_stock_price(r, sigma, S, dt)
        # Tiempo restante hasta el vencimiento
        tau = T - k * dt
        # Calcular nueva Delta
        Delta = delta_call(S, K, r, tau, sigma)
        # Calcular acciones a comprar/vender
        cant_acciones = (Delta - Delta_prev) * amt_options
        # Flujo de caja asociado (en millones)
        costo_acciones = (cant_acciones * S)
        # Interés del costo acumulado anterior
        interes = accum_cost * r * dt
        # Actualizar costo acumulado
        accum_cost = accum_cost * (1 + r * dt) + costo_acciones
        # Agregar fila al DataFrame
        df.loc[k] = [k, S, Delta, cant_acciones,
                     costo_acciones, accum_cost, interes]
        # Actualizar para el siguiente paso
        Delta_prev = Delta

    # --- AJUSTE DE LIQUIDACIÓN FINAL (EN T) ---

    S_T = S  # S es el precio final de la acción
    shares_held_final = Delta * amt_options # Delta es Delta[n_steps]
    
    # Costo final total para la performance (en miles)
    costo_final_performance = accum_cost
    
    # Flujo de caja de liquidación (en miles)
    flujo_liquidacion = 0
    
    if S_T > K:
        # 1. Caso ITM (Option Exercised): Institución recibe el strike K a cambio de las acciones.
        # Es un ingreso, por lo que reduce el costo acumulado (costo_final_performance).
        flujo_liquidacion = K * amt_options
    else:
        # 2. Caso OTM (Option Expires): Institución vende las acciones remanentes del hedge.
        # También es un ingreso.
        flujo_liquidacion = shares_held_final * S_T

    # Actualizar el costo acumulado con el flujo final de liquidación
    costo_final_performance -= flujo_liquidacion
    
    # Crear la fila final de resumen para el DataFrame (Opcional, pero ayuda a la trazabilidad)
    df.loc[n_steps + 1] = ['Final', S_T, 
                            (1.0 if S_T > K else 0.0), # Delta teórico en T
                            -shares_held_final, # Acciones vendidas/entregadas
                            -flujo_liquidacion, # Ingreso
                            costo_final_performance, 0] # Costo final


    return df

def montecarlo_delta_hedging(S_0, K, r, sigma, T, n_steps, amt_options, n_sim):
    # C = hedging costs array
    Cs = np.empty(n_sim, dtype=float)
    for i in range(n_sim):
        df = delta_hedging_single_sim(S_0, K, r, sigma, T, amt_options, n_steps)
        Cs[i] = df['Costo acumulado'].loc[len(df)-1] 
    return Cs

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


    # df = delta_hedging_single_sim(S_0, K, r, sigma, T, amt_options, n_steps)
    # ## Table 19.2 - 19.3 Hull
    # print(df.head(21))

    Cs = montecarlo_delta_hedging(S_0, K, r, sigma, T, n_steps, amt_options, n_sim)
    performance = get_hedge_preformance(Cs, BSM_price * amt_options)
    print(f'\n\nPerformance: {performance}\n\n')
if __name__ == '__main__':
    main()
