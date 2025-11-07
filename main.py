import src.stop_loss, src.delta_hedging

def main():
    K = 50
    S_0 = 49
    sigma = 0.20
    T = 0.3846 # años, equiv a 20 semanas
    n_sim = 1000
    r = 0.05
    amt_options = 100_000 # cantidad de opciones que se firmaron en el contrato
    BSM_price = 2.40

    n_steps = [4, 5, 10, 20, 40, 80]
    deltas_t = [T/n_step for n_step in n_steps]
    sl_hedges_performances = []
    dh_hedges_performances = []

    for delta_t, n_step in zip(deltas_t, n_steps):
        src.stop_loss.run_stop_loss(K, S_0, r, sigma, delta_t, n_step, n_sim, BSM_price, sl_hedges_performances, tasa_comision=0.0015)
        src.delta_hedging.run_delta_hedging(S_0, K, r, sigma, T, n_step, amt_options, n_sim, BSM_price, dh_hedges_performances, tasa_comision=0.0015)    

    dts_weeks = [round(dt * 52, 2) for dt in deltas_t]

    # Table 19.1
    src.stop_loss.print_stop_loss_table(dts_weeks, sl_hedges_performances)
    # Table 19.2 / 19.3
    n = 20
    df = src.delta_hedging.delta_hedging_single_sim(S_0, K, r, sigma, T, amt_options, n, tasa_comision=0)
    src.delta_hedging.print_dh_single_sim_table(df)
    # Table 19.4
    src.delta_hedging.print_dh_performances_table(dts_weeks, dh_hedges_performances)


if __name__ == '__main__':
    main()



