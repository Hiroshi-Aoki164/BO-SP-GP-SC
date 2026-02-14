#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel execution script for BO_GPR.py
Multiprocessing support for Windows
"""

#%%
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from pystan import StanModel
import pickle
from sklearn.linear_model import LinearRegression
import math
import arviz
import support_function as sp
import multiprocessing as mp
import warnings
import os
warnings.filterwarnings("ignore")

#%%
def process_single_seed(seed_num, shared_data):
    try:
        print(f"Process {os.getpid()}: Seed {seed_num} started")

        X1 = shared_data['X1']
        X2 = shared_data['X2']
        Y = shared_data['Y']
        X_grid = shared_data['X_grid']
        y_grid = shared_data['y_grid']
        N_grid = shared_data['N_grid']
        f_result = shared_data['f_result']
        count_num = shared_data['count_num']
        N_first = shared_data['N_first']
        iter_ = shared_data['iter_']
        warmup = shared_data['warmup']
        eq = shared_data['eq']

        f_model = Path("./model")
        with open(f_model/'model_GPR.pickle', mode='br') as fi:
            model = pickle.load(fi)

        data_ = sp.generate_initial_data(N_first, seed=seed_num, x1_range=[-0.2, 0.2], x2_range=[-0.2, 0.2])
        X_act = pd.DataFrame(data_, columns = X_grid.columns)
        y_act = sp.cosines(X_act["X1"],X_act["X2"])
        sigma_ave = pd.DataFrame()

        f_plot, f_hist, f_dens, f_trace, \
            f_kakutoku, f_yave, f_ysd = sp.make_folder_BO_GPR(seed_num, f_result)

        D = len(X_act.T)

        for count in range(0,count_num):
            N = len(X_act)
            data = {
                    "N" : N,
                    "D" : D,
                    "X" : X_act,
                    "y" : y_act,
                    "N_grid" : N_grid,
                    "X_grid" : X_grid
              }


            fit = model.sampling(data = data,
                            iter = iter_,#2000
                            warmup = warmup,
                            seed = 123,
                            chains = 1,
                            n_jobs=1
            )

            para = ["rho","alpha","sd_e"]
            sp.plot_posterior_forest(fit,count,f_plot,para)
            sp.plot_posterior_hist(fit,count,f_hist,para)
            sp.plot_posterior_density(fit,count,f_dens,para)
            sp.plot_posterior_trace(fit,count,f_trace,para)

            summary = arviz.summary(fit)
            print(
                f"Process {os.getpid()}: Seed {seed_num}, Experiment {count}: count of r_hat<1.1: "
                + str(sum(summary["r_hat"] <= 1.10))
            )

            samples = fit.extract()

            #%%
            EI = sp.cal_EI(y_act, samples)
            PI = sp.cal_PI(y_act, samples)

            y_ave = sp.cal_ave(samples)
            y_sd = sp.cal_sd(samples)

            z_EI = sp.make_z_y(EI)
            z_PI = sp.make_z_y(PI)
            z_yave = sp.make_z_y(y_ave)
            z_ysd = sp.make_z_y(y_sd)


            # When using PI
            # EI = PI
            # z_EI = z_PI

            X_act, y_act = sp.add_act_data(X_act, N_first, EI, X_grid, eq=eq)
            # X_act, y_act = sp.add_act_data(X_act, N_first, PI, X_grid, eq=eq)


            title = "Experiment {}".format(count)
            level = None
            file_pass = f_kakutoku/"experiment_{}.png".format(count)
            sp.plot_contour(X1, X2, z_EI, title, file_pass, level, X_act, N_first)

            level = np.linspace(-3,1,16)
            file_pass = f_yave/"experiment_{}.png".format(count)
            sp.plot_contour(X1, X2, z_yave, title, file_pass, level, X_act, N_first)

            level = None
            file_pass = f_ysd/"experiment_{}.png".format(count)
            sp.plot_contour(X1, X2, z_ysd, title, file_pass, level, X_act, N_first)

            sigma_ave.loc[count,0] = y_sd.mean()[0]

        y_act = y_act.reset_index(drop=True)

        print(f"Process {os.getpid()}: Seed {seed_num} completed")

        return seed_num, {"y_act": y_act, "sigma_ave": sigma_ave}

    except Exception as e:
        print(f"Process {os.getpid()}: Seed {seed_num} error: {str(e)}")
        import traceback
        traceback.print_exc()
        return seed_num, None

def main():
    f_result = Path("./result/BO(GPR)")
    f_result.mkdir(parents=True, exist_ok=True)
    f_model = Path(("./model"))
    f_model.mkdir(parents=True, exist_ok=True)

    start = -0.2
    stop = 0.8
    num = 11
    eq = 3
    X1, X2, Y = sp.plt_data(start,stop,start,stop,num,eq=eq)
    X_grid, y_grid = sp.make_grid(X1, X2, Y)
    N_grid = num*num

    title = "actual y"
    level = np.linspace(-3,1,16)
    file_pass = "y_actual.png"
    sp.plot_contour(X1, X2, Y, title, file_pass, level)

    seed_num_all = 50
    count_num = 20
    N_first = 3
    iter_ = 1000
    warmup = 200
    try:
        with open(f_model/'model_GPR.pickle', mode='br') as fi:
            model = pickle.load(fi)
        print("Loaded existing model")
    except FileNotFoundError:
        print("Building model...")
        model = StanModel("stan/GPR.stan")
        with open(f_model/'model_GPR.pickle', mode='wb') as fo:
            pickle.dump(model, fo)
        print("Built and saved model")

    shared_data = {
        'X1': X1,
        'X2': X2,
        'Y': Y,
        'X_grid': X_grid,
        'y_grid': y_grid,
        'N_grid': N_grid,
        'f_result': f_result,
        'count_num': count_num,
        'N_first': N_first,
        'iter_': iter_,
        'warmup': warmup,
        'eq': eq,
    }

    num_processes = max(1, min(mp.cpu_count() - 1, seed_num_all))
    print(f"Parallel workers: {num_processes} processes")
    print(f"Total CPU cores: {mp.cpu_count()}")

    Y_end_all = pd.DataFrame()
    sigma_ave_all = pd.DataFrame()

    print("Starting parallel processing...")

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_single_seed,
                              [(seed_num, shared_data) for seed_num in range(seed_num_all)])

    print("Parallel processing complete; merging results...")

    for seed_num, result in results:
        if result is not None:
            Y_end_all = pd.concat([Y_end_all, result["y_act"]], axis=1)
            sigma_ave_all = pd.concat([sigma_ave_all, result["sigma_ave"]], axis=1)
        else:
            print(f"Seed {seed_num} result is invalid")

    print("Saving results...")
    Y_end_all.to_csv(f_result/"y_all.csv", encoding="shift-jis")
    sigma_ave_all.to_csv(f_result/"y_sigma.csv", encoding="shift-jis")

    print(f"All tasks completed. Results saved to {f_result}.")

if __name__ == "__main__":
    # Windows support
    mp.freeze_support()
    main()


