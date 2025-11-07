import pandas as pd
import numpy as np
from scipy.stats import norm


def sim_stock_price(r, sigma, S, dt):
    eps = np.random.normal(0,1)
    dS = r * S * dt + sigma * S * np.sqrt(dt) * eps
    S_new = max(S + dS, 1e-12)
    return S_new

def get_hedge_preformance(Cs, BSM_price):
    std_dev = np.std(Cs, ddof=1)
    return std_dev / BSM_price

def get_d1(S_0, K, r, tau, deviation):
    tau_eff = max(tau, 1e-12)
    return (np.log(S_0 / K) + (r +(deviation ** 2 / 2)) * tau_eff) / (deviation * np.sqrt(tau_eff))


def delta_call(S0, K, r, tau, sigma):
    x = get_d1(S0, K, r, tau, sigma)
    return norm.cdf(x)

def delta_put(S0, K, r, tau, sigma):
    x = get_d1(S0, K, r, tau, sigma)
    return norm.cdf(x) - 1

def delta_hedging_single_sim(S0, K, r, sigma, T, amt_options, n_steps, tasa_comision):
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
    costo_0 = acciones_0 * S / 1000
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
        factor_comision = (1 + tasa_comision) if cant_acciones > 0 else (1 - tasa_comision)
        costo_acciones = cant_acciones * S * factor_comision / 1000
        # Interés del costo acumulado anterior
        interes = accum_cost * r * dt
        # Actualizar costo acumulado
        accum_cost = accum_cost * (1 + r * dt) + costo_acciones
        # Agregar fila al DataFrame
        df.loc[k] = [k, S, Delta, cant_acciones,
                     costo_acciones, accum_cost, interes]
        # Actualizar para el siguiente paso
        Delta_prev = Delta

    # Agregar fila final
    df.loc[n_steps + 1] = [n_steps + 1, S, 0 if S < K else 1, cant_acciones,
                 costo_acciones, accum_cost, interes]

    # --- AJUSTE DE LIQUIDACIÓN FINAL (EN T) ---

    S_T = S  # S es el precio final de la acción
    shares_held_final = Delta * amt_options # Delta es Delta[n_steps]
    
    # Costo final total para la performance (en miles)
    costo_final_performance = accum_cost * 1000
    
    # Flujo de caja de liquidación (en miles)
    flujo_liquidacion = 0
    
    factor_comision_liquidacion = 1 - tasa_comision

    if S_T > K:
        # 1. Caso ITM (Option Exercised): Institución recibe el strike K a cambio de las acciones.
        # Es un ingreso, por lo que reduce el costo acumulado (costo_final_performance).
        flujo_liquidacion = K * amt_options * factor_comision_liquidacion
    else:
        # 2. Caso OTM (Option Expires): Institución vende las acciones remanentes del hedge.
        # También es un ingreso.
        flujo_liquidacion = shares_held_final * S_T * factor_comision_liquidacion

    # Actualizar el costo acumulado con el flujo final de liquidación
    costo_final_performance -= flujo_liquidacion
    
    # Crear la fila final de resumen para el DataFrame (Opcional, pero ayuda a la trazabilidad)
    df.loc[n_steps + 1] = [n_steps + 1, S_T, 
                            (1.0 if S_T > K else 0.0), # Delta teórico en T
                            shares_held_final, # Acciones vendidas/entregadas
                            flujo_liquidacion, # Ingreso
                            costo_final_performance, 0] # Costo final


    return df

def montecarlo_delta_hedging(S_0, K, r, sigma, T, n_steps, amt_options, n_sim, tasa_comision):
    # C = hedging costs array
    Cs = np.empty(n_sim, dtype=float)
    for i in range(n_sim):
        df = delta_hedging_single_sim(S_0, K, r, sigma, T, amt_options, n_steps, tasa_comision)
        Cs[i] = df['Costo acumulado ($000)'].loc[len(df)-1] 
    return Cs

def print_dh_single_sim_table(df):
    """
    Imprime una tabla formateada del DataFrame de cobertura Delta-Hedging.
    Similar a la Tabla 19.2 del libro de Hull.
    """

    print("\n\n\nTabla 19.2 / 19.3 - Evolución de la estrategia Delta-Hedging\n")
    print(f"{'t':>9} | {'S(t)':>8} | {'Delta':>8} | {'Acciones':>10} | {'Costo acciones compradas':>12} | {'Costo acum.':>12} | {'Interés':>10}")
    print("-" * 100)

    for _, row in df.iterrows():
        print(f"{row['Week']:>9} | "
            f"{row['Precio de la acción']:>8.3f} | "
            f"{row['Delta']:>8.3f} | "
            f"{row['Acciones compradas']:>10.1f} | "
            f"{row['Costo acciones compradas ($000)']:>24.5f} | "
            f"{row['Costo acumulado ($000)']:>12.2f} | "
            f"{row['Costo de Interes ($000)']:>10.5f}")
    print('\n\n\n')

def print_dh_performances_table(dts_weeks, hedges_performances):
    print("\nTabla 19.4 - Performance de la cobertura Delta-hedging")
    print("--------------------------------------------------")
    print(f"{'Δt (semanas)':>14} | {'Performance':>12}")
    print("---------------|--------------")
    for dt, perf in zip(dts_weeks, hedges_performances):
        print(f"{dt:>14.2f} | {perf:>12.5f}")
    print("---------------|--------------\n\n")


def run_delta_hedging(S_0, K, r, sigma, T, n_steps, amt_options, n_sim, BSM_price, dh_hedges_performances, tasa_comision):
    Cs = montecarlo_delta_hedging(S_0, K, r, sigma, T, n_steps, amt_options, n_sim, tasa_comision)
    performance = get_hedge_preformance(Cs, BSM_price * amt_options)
    dh_hedges_performances.append(performance)

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


    ## Table 19.2 - 19.3 Hull    
    df = delta_hedging_single_sim(S_0, K, r, sigma, T, amt_options, n_steps, tasa_comision=0.0015)
    print_dh_single_sim_table(df)

    Cs = montecarlo_delta_hedging(S_0, K, r, sigma, T, n_steps, amt_options, n_sim, tasa_comision=0.0015)
    performance = get_hedge_preformance(Cs, BSM_price * amt_options)
    print(f'Performance: {performance}\n')


if __name__ == '__main__':
    main()

