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
    n_steps = 20

    n_steps = [4, 5, 10, 20, 40, 80]
    deltas_t = [T/n_step for n_step in n_steps]
    hedges_performances = []

    for delta_t, n_step in zip(deltas_t, n_steps):
        src.stop_loss.run_stop_loss(K, S_0, r, sigma, delta_t, n_step, n_sim, BSM_price, hedges_performances)    

    dts_weeks = [round(dt * 52, 2) for dt in deltas_t]

    src.stop_loss.print_stop_loss_table(dts_weeks, hedges_performances)



if __name__ == '__main__':
    main()



