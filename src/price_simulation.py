import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def simulate_single_path(S0, mu, sigma, dt, n_steps):
    
    rng = np.random.default_rng()
    t = np.arange(n_steps + 1) * dt
    S = np.empty(n_steps + 1, dtype=float)
    S[0] = S0
    for k in range(1, n_steps + 1):
        eps = rng.normal(0.0, 1.0)
        dS = mu * S[k-1] * dt + sigma * S[k-1] * np.sqrt(dt) * eps
        S[k] = max(S[k-1] + dS, 1e-12)
    return t, S

def stoploss_trades_and_position(S, K):
    """
    posicion en cada cierre t es 1{S[t] >= K}. Si cambia vs el cierre previo, hubo trade.
    opera al final del intervalo, a precio S[t].
    """
    n = len(S)
    pos = np.zeros(n, dtype=int)
    pos[0] = 1 if S[0] >= K else 0
    trades = []

    for t in range(1, n):
        pos[t] = 1 if S[t] >= K else 0
        if pos[t] != pos[t-1]:
            action = "buy" if pos[t] == 1 else "sell"
            trades.append({"t_index": t, "action": action, "price": float(S[t])})

    return trades, pos

def plot_stoploss_price_only(t, S, K, trades, outfile="figure_19_1_stoploss.png"):
    """
    precio vs tiempo, linea horizontal en K,
    y marcadores de trades al final de cada intervalo.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))

    ax1.plot(t, S, label="Precio del subyacente")
    ax1.axhline(K, linestyle="--", linewidth=1, label=f"Strike K={K}")

    
    if trades:
        buys_idx  = [e["t_index"] for e in trades if e["action"] == "buy"]
        buys_p    = [e["price"]   for e in trades if e["action"] == "buy"]
        sells_idx = [e["t_index"] for e in trades if e["action"] == "sell"]
        sells_p   = [e["price"]   for e in trades if e["action"] == "sell"]

        ax1.scatter(t[buys_idx],  buys_p,  marker="^", s=80, label="Compra (fin de intervalo)")
        ax1.scatter(t[sells_idx], sells_p, marker="v", s=80, label="Venta (fin de intervalo)")

    ax1.set_xlabel("Tiempo (años)")
    ax1.set_ylabel("Precio")
    ax1.set_title("Stop-loss en una trayectoria (Figura 19.1)")
    ax1.legend(loc="best")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()


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

    # simulacion de una trayectoria + stop-loss y figura 19.1 ===
    t, S = simulate_single_path(
        S0=initial_stock_price, mu=media, sigma=deviation,
        dt=interval, n_steps=n_steps
    )

    trades, pos = stoploss_trades_and_position(S, K)

    plot_stoploss_price_only(t, S, K, trades, outfile="figure_19_1_stoploss.png")


if __name__ == '__main__':
    main()

